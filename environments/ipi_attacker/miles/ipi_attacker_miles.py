# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Miles hooks for the ipi_attacker environment.

- ``run``: the agent function (``--custom-agent-function-path ipi_attacker_miles.run``). One ``/run`` POST per
  sample, with the row's prompt as the attacker's input and the Miles session URL as ``policy_base_url``.
- ``reward_func``: the reward hook (``--custom-rm-path ipi_attacker_miles.reward_func``).
- ``drop_masked``: the sample filter (``--rollout-sample-filter-path ipi_attacker_miles.drop_masked``).

Only stdlib and httpx: importing miles pulls in torch, and this module must load on CPU-only machines too.

Env vars:
  NEMO_GYM_URL          base URL of the ipi_attacker agent server (default: http://localhost:12000).
  NEMO_GYM_RUN_TIMEOUT  wall-clock cap in seconds for one /run call (default: 3600).
"""

import asyncio
import logging
import os
import random
from typing import Any

import httpx


logger = logging.getLogger(__name__)

CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_S = (1.0, 5.0)


def run_body(base_url: str, prompt: Any, request_kwargs: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """The /run body: the row's fields at the top level, the prompt as input, and the session as the policy."""
    params: dict[str, Any] = {"input": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt}
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    if request_kwargs.get("max_tokens") is not None:
        params["max_output_tokens"] = request_kwargs["max_tokens"]
    return {**metadata, "responses_create_params": params, "policy_base_url": f"{base_url}/v1"}


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
    url = f"{os.getenv('NEMO_GYM_URL', 'http://localhost:12000').rstrip('/')}/run"
    timeout_s = float(os.getenv("NEMO_GYM_RUN_TIMEOUT", "3600"))
    body = run_body(base_url, prompt, request_kwargs or {}, metadata or {})
    try:
        response = await asyncio.wait_for(_post_run(url, body), timeout=timeout_s)
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        logger.error(f"ipi_attacker /run returned {status}: {error.response.text[:500]}")
        return _masked(f"ipi_attacker:run_http_{status}")
    except Exception as error:
        logger.warning(f"ipi_attacker /run failed: {error!r}")
        return _masked("ipi_attacker:run_failed")
    # simple_agent reports a policy timeout as a 200 failure row with no reward.
    if "reward" not in response or response.get("_ng_failure_class"):
        return _masked(response.get("failure_kind") or "ipi_attacker:run_failed")
    return {
        "reward": float(response["reward"]),
        "mask_sample": bool(response.get("mask_sample")),
        "failure_kind": response.get("failure_kind"),
    }


async def reward_func(args, samples, **kwargs):
    """The reward /run returned, stored in the sample's metadata by ``run``."""
    if isinstance(samples, list):
        return [sample.metadata.get("reward", 0.0) for sample in samples]
    return samples.metadata.get("reward", 0.0)


def _is_masked(rollout: list) -> bool:
    return bool((rollout[0].metadata or {}).get("mask_sample"))


def drop_masked(args, data: list) -> None:
    """Leave masked samples out of the loss, at their group's unmasked mean reward.

    ``remove_sample`` alone keeps a sample's reward in its group's normalization, so a masked placeholder 0 would
    shift its siblings' advantages. A group entry is one rollout: a sample, or a list of session-branch samples that
    share one reward.
    """
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
        logger.info(f"drop_masked: left {masked}/{total} masked rollouts out of the loss")
