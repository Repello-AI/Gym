# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Miles hooks for the ipi_attacker environment.

- ``run``: the agent function (``--custom-agent-function-path ipi_attacker_miles.run``). One ``/run`` POST per
  sample, with the row's prompt as the attacker's input and the Miles session URL as ``policy_base_url``.
- ``reward_func``: the reward hook (``--custom-rm-path ipi_attacker_miles.reward_func``).
- ``drop_masked``: the sample filter (``--rollout-sample-filter-path ipi_attacker_miles.drop_masked``).
- ``normalize_unmasked``: the reward post-process (``--custom-reward-post-process-path
  ipi_attacker_miles.normalize_unmasked``); Miles runs it in place of its own GRPO normalization.

Only stdlib and httpx: importing miles pulls in torch, and this module must load on CPU-only machines too.

Env vars:
  NEMO_GYM_URL               base URL of the ipi_attacker agent server (default: http://localhost:12000).
  NEMO_GYM_RUN_TIMEOUT       wall-clock cap in seconds for one /run call (default: 3600).
  IPI_MASK_BREAKER_RATE      a rollout counts as mostly masked above this fraction (default: 0.5).
  IPI_MASK_BREAKER_ROLLOUTS  stop after this many mostly-masked rollouts in a row (default: 3).
"""

import asyncio
import logging
import math
import os
import random
import statistics
from typing import Any

import httpx


logger = logging.getLogger(__name__)

DEFAULT_GYM_URL = "http://localhost:12000"
DEFAULT_RUN_TIMEOUT_S = "3600"
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_S = (1.0, 5.0)
_mostly_masked_streak = 0  # consecutive rollouts over the breaker's masked rate


def run_body(base_url: str, prompt: Any, request_kwargs: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """The /run body: the row's fields at the top level, the prompt as input, and the session as the policy."""
    params: dict[str, Any] = {"input": prompt}
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    if request_kwargs.get("max_tokens") is not None:
        params["max_output_tokens"] = request_kwargs["max_tokens"]
    return {**metadata, "responses_create_params": params, "policy_base_url": f"{base_url}/v1"}


def _is_messages(prompt: Any) -> bool:
    return isinstance(prompt, list) and bool(prompt) and all(isinstance(m, dict) and "role" in m for m in prompt)


def _masked(failure_kind: str) -> dict[str, Any]:
    return {"reward": 0.0, "mask_sample": True, "failure_kind": failure_kind}


async def _post_run(url: str, body: dict[str, Any]) -> dict[str, Any]:
    # Only a refused connection is retried: any later failure may have started a paid episode.
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as client:
        for attempt in range(CONNECT_ATTEMPTS):
            try:
                response = await client.post(url, json=body)
                break
            except httpx.ConnectError:
                if attempt == CONNECT_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(random.uniform(*CONNECT_BACKOFF_S))
    response.raise_for_status()
    return response.json()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one episode through /run; return the reward and mask to merge into the sample's metadata.

    Never raises: Miles swallows an agent-function exception and trains the sample as reward 0, so every failure
    comes back masked instead.
    """
    if not _is_messages(prompt):
        logger.error(
            "ipi_attacker: the prompt is not a list of chat messages; drop --apply-chat-template, "
            "which renders it to text before this hook"
        )
        return _masked("ipi_attacker:prompt_not_messages")
    try:
        url = f"{os.getenv('NEMO_GYM_URL', DEFAULT_GYM_URL).rstrip('/')}/run"
        timeout_s = float(os.getenv("NEMO_GYM_RUN_TIMEOUT", DEFAULT_RUN_TIMEOUT_S))
        body = run_body(base_url, prompt, request_kwargs or {}, metadata or {})
        response = await asyncio.wait_for(_post_run(url, body), timeout=timeout_s)
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        logger.error(f"ipi_attacker /run returned {status}: {error.response.text[:500]}")
        return _masked(f"ipi_attacker:run_http_{status}")
    except Exception as error:
        logger.warning(f"ipi_attacker /run failed: {error!r}")
        return _masked("ipi_attacker:run_failed")
    if not isinstance(response, dict):
        return _masked("ipi_attacker:bad_response")
    # simple_agent reports a policy timeout as a 200 failure row with no reward.
    if "reward" not in response or response.get("_ng_failure_class"):
        return _masked(response.get("failure_kind") or "ipi_attacker:run_failed")
    reward = response["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        return _masked("ipi_attacker:bad_response")
    return {
        "reward": float(reward),
        "mask_sample": bool(response.get("mask_sample")),
        "failure_kind": response.get("failure_kind"),
    }


async def reward_func(args, samples, **kwargs):
    """The reward /run returned, stored in the sample's metadata by ``run``."""
    if isinstance(samples, list):
        return [sample.metadata.get("reward", 0.0) for sample in samples]
    return samples.metadata.get("reward", 0.0)


def _is_masked(rollout: list) -> bool:
    # No reward means ``run`` never returned (its process died), so the reward is not a real score.
    metadata = rollout[0].metadata or {}
    return bool(metadata.get("mask_sample")) or "reward" not in metadata


def drop_masked(args, data: list) -> None:
    """Leave masked samples out of the loss, and stop training once most samples keep coming back masked.

    A masked sample's placeholder reward becomes its group's unmasked mean, so the logged raw reward is not pulled
    down by failures; ``normalize_unmasked`` leaves it out of the advantages. A group entry is one rollout: a sample,
    or a list of session-branch samples that share one reward.
    """
    global _mostly_masked_streak
    total = masked = 0
    for group in data:
        rollouts = [entry if isinstance(entry, list) else [entry] for entry in group]
        valid = [rollout[0].reward for rollout in rollouts if not _is_masked(rollout)]
        baseline = sum(valid) / len(valid) if valid else 0.0
        for rollout in rollouts:
            total += 1
            if _is_masked(rollout):
                masked += 1
                for sample in rollout:
                    sample.remove_sample = True
                    sample.reward = baseline
    if masked:
        logger.warning(f"drop_masked: left {masked}/{total} masked rollouts out of the loss")
    rate = float(os.getenv("IPI_MASK_BREAKER_RATE", "0.5"))
    limit = int(os.getenv("IPI_MASK_BREAKER_ROLLOUTS", "3"))
    _mostly_masked_streak = _mostly_masked_streak + 1 if total and masked / total > rate else 0
    if _mostly_masked_streak >= limit:
        raise RuntimeError(
            f"drop_masked: over {rate:.0%} of samples masked {limit} rollouts in a row; "
            "check the Gym servers, API keys and judge before resuming"
        )


def normalize_unmasked(args, samples: list) -> tuple[list[float], list[float]]:
    """GRPO advantages per prompt group over unmasked rollouts only; a masked sample gets 0.

    Miles calls this in place of its own normalization, which counts masked samples in each group's std. A rollout's
    segments share one reward and count once, keyed as Miles keys them.
    """
    raw = [sample.get_reward_value(args) for sample in samples]
    groups: dict[Any, dict[Any, list[int]]] = {}
    for row, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("normalize_unmasked needs each sample's group_index")
        rollout = sample.rollout_id if sample.rollout_id is not None else sample.index
        groups.setdefault(sample.group_index, {}).setdefault(rollout, []).append(row)
    scale_by_std = args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization
    advantages = [0.0] * len(samples)
    for rollouts in groups.values():
        valid = [rows for rows in rollouts.values() if not samples[rows[0]].remove_sample]
        rewards = [raw[rows[0]] for rows in valid]
        if not rewards:
            continue
        mean = statistics.fmean(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        for rows, reward in zip(valid, rewards):
            advantage = reward - mean
            if scale_by_std and std > 0:
                advantage /= std + 1e-6
            for row in rows:
                advantages[row] = advantage
    return raw, advantages
