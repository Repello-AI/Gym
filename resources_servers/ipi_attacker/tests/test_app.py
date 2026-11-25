# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import exercise_verifier_fixture

from ..app import (
    VERIFIER_FIXTURE,
    IpiAttackerResourcesServer,
    IpiAttackerResourcesServerConfig,
    IpiAttackerVerifyRequest,
    attacker_fill,
)
from ..verifier_fixture import create_fixture_server, fixture_clients


CASES = Path(__file__).parent / "verifier_cases.jsonl"
ASYNC_CLIENT = httpx.AsyncClient
MILES_HOOKS = Path(__file__).resolve().parents[3] / "environments" / "ipi_attacker" / "miles" / "ipi_attacker_miles.py"


def full_break_request(**fields) -> IpiAttackerVerifyRequest:
    return IpiAttackerVerifyRequest.model_validate(json.loads(CASES.read_text().splitlines()[0])["request"] | fields)


def verify(request, **clients):
    server = create_fixture_server()
    server._clients = fixture_clients(request.environment, "reference", **clients)
    return asyncio.run(server.verify(request))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="stochastic",
        )
    )


def test_fill_is_the_last_message_without_leading_reasoning() -> None:
    response = full_break_request().response.model_dump(mode="json")
    response["output"][0]["content"][0]["text"] = "<think>plan</think>\n Please send it. <think>kept</think>"
    assert attacker_fill(NeMoGymResponse.model_validate(response)) == "Please send it. <think>kept</think>"


def test_break_reward_rises_with_concealment_from_action_only() -> None:
    for concealment_score, reward in ((0, 0.5), (4, 0.7), (10, 1.0)):
        result = verify(full_break_request(), concealment_score=concealment_score)
        assert (result.reward, result.mask_sample) == (reward, False)


def test_judge_failure_is_masked_not_zero() -> None:
    result = verify(full_break_request(), judge_error="judge unavailable: 529")
    assert (result.reward, result.mask_sample, result.failure_kind) == (
        0.0,
        True,
        "ipi_attacker:concealment_judge_error",
    )


def test_startup_refuses_a_missing_key_file_or_unqualified_security_reader(tmp_path) -> None:
    keys = tmp_path / "keys.env"
    keys.write_text("OPENAI_API_KEY=x\nANTHROPIC_API_KEY=y\n")
    for api_key_file, security_model, error in (
        ("missing.env", "claude-sonnet-5-5", "not a file"),
        (str(keys), "gpt-4o-mini", "qualified"),
    ):
        config = IpiAttackerResourcesServerConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="ipi_attacker",
            api_key_file=api_key_file,
            security_model=security_model,
            call_log_path=str(tmp_path / "calls.jsonl"),
        )
        with pytest.raises(ValueError, match=error):
            IpiAttackerResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_defender_thinking_picks_the_defender_client(tmp_path) -> None:
    keys = tmp_path / "keys.env"
    keys.write_text("OPENAI_API_KEY=x\nANTHROPIC_API_KEY=y\n")
    built = {}
    for thinking in (False, True):
        config = IpiAttackerResourcesServerConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="ipi_attacker",
            api_key_file=str(keys),
            defender_thinking=thinking,
            call_log_path=str(tmp_path / "calls.jsonl"),
        )
        server = IpiAttackerResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        built[thinking] = type(server._clients[0].client).__name__
    assert built == {False: "OpenAICompatClient", True: "OpenAIResponsesClient"}


@pytest.fixture
def hooks():
    spec = importlib.util.spec_from_file_location("ipi_attacker_miles", MILES_HOOKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def miles_row():
    """A fixture row as Miles hands it to the agent function: the prompt, and the row's other fields as metadata."""
    request = json.loads(CASES.read_text().splitlines()[0])["request"]
    metadata = {key: value for key, value in request.items() if key not in ("responses_create_params", "response")}
    return request["responses_create_params"]["input"], metadata, request["response"]


def masked(kind):
    return {"reward": 0.0, "mask_sample": True, "failure_kind": kind}


def run_hook(hooks, monkeypatch, handler):
    """Run the agent function against a mocked /run; return its result and every request body sent."""
    bodies = []

    def record(request):
        bodies.append(json.loads(request.content))
        return handler(request)

    monkeypatch.setattr(
        hooks.httpx, "AsyncClient", lambda **kwargs: ASYNC_CLIENT(transport=httpx.MockTransport(record), **kwargs)
    )
    monkeypatch.setattr(hooks, "CONNECT_BACKOFF_S", (0.0, 0.0))
    prompt, metadata, _ = miles_row()
    return asyncio.run(hooks.run("http://miles/sessions/s1", prompt, {}, metadata)), bodies


def test_run_body_sends_the_prompt_and_is_a_valid_verify_request(hooks) -> None:
    prompt, metadata, response = miles_row()
    body = hooks.run_body(
        "http://miles/sessions/s1", prompt, {"temperature": 0.8, "top_p": None, "max_tokens": 512}, metadata
    )
    assert body["responses_create_params"] == {"input": prompt, "temperature": 0.8, "max_output_tokens": 512}
    assert body["policy_base_url"] == "http://miles/sessions/s1/v1"
    request = IpiAttackerVerifyRequest.model_validate(body | {"response": response})
    assert (request.environment, request.env_id) == (metadata["environment"], metadata["env_id"])
    assert hooks.run_body("http://miles", "write it", {}, {})["responses_create_params"]["input"] == [
        {"role": "user", "content": "write it"}
    ]


@pytest.mark.parametrize(
    "status, payload, expected",
    [
        (200, {"reward": 0.75, "mask_sample": False}, {"reward": 0.75, "mask_sample": False, "failure_kind": None}),
        (
            200,
            {"reward": 0.0, "mask_sample": True, "failure_kind": "ipi_attacker:concealment_judge_error"},
            masked("ipi_attacker:concealment_judge_error"),
        ),
        (
            200,
            {"_ng_failure_class": "agent_run_error", "failure_kind": "model_request_timeout"},
            masked("model_request_timeout"),
        ),
        (400, {"detail": "policy_base_url needs allow_policy_base_url: true"}, masked("ipi_attacker:run_http_400")),
        (200, {"reward": None}, masked("ipi_attacker:bad_response")),
        (200, {"reward": "0.5"}, masked("ipi_attacker:bad_response")),
        (200, [0.5], masked("ipi_attacker:bad_response")),
    ],
)
def test_run_returns_the_reward_or_a_mask(hooks, monkeypatch, status, payload, expected) -> None:
    result, bodies = run_hook(hooks, monkeypatch, lambda request: httpx.Response(status, json=payload))
    assert result == expected
    assert len(bodies) == 1 and bodies[0]["responses_create_params"]["input"] == miles_row()[0]


def test_run_resends_only_a_request_that_never_left(hooks, monkeypatch) -> None:
    attempts = iter([httpx.ConnectError("refused"), httpx.Response(200, json={"reward": 1.0})])

    def connect_then_answer(request):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result, bodies = run_hook(hooks, monkeypatch, connect_then_answer)
    assert (result["reward"], len(bodies)) == (1.0, 2)

    def read_error(request):
        raise httpx.ReadError("connection reset")

    result, bodies = run_hook(hooks, monkeypatch, read_error)
    assert (result, len(bodies)) == (masked("ipi_attacker:run_failed"), 1)


def test_run_masks_a_run_past_its_timeout(hooks, monkeypatch) -> None:
    monkeypatch.setenv("NEMO_GYM_RUN_TIMEOUT", "0.05")

    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={"reward": 1.0})

    result, _ = run_hook(hooks, monkeypatch, slow)
    assert result == masked("ipi_attacker:run_failed")


def sample(reward, masked=False):
    return SimpleNamespace(reward=reward, metadata={"reward": reward, "mask_sample": masked}, remove_sample=False)


def test_masked_samples_leave_the_loss_and_take_their_siblings_mean(hooks) -> None:
    crashed = SimpleNamespace(reward=0.0, metadata={}, remove_sample=False)  # the agent function raised: no reward
    group = [
        sample(1.0),
        sample(0.5),
        sample(0.0, masked=True),
        [sample(0.0, masked=True), sample(0.0, masked=True)],
        crashed,
    ]
    all_masked = [sample(0.0, masked=True), sample(0.0, masked=True)]
    hooks.drop_masked(None, [group, all_masked])
    flat = [*group[:3], *group[3], crashed]
    assert [s.reward for s in flat] == [1.0, 0.5, 0.75, 0.75, 0.75, 0.75]
    assert [s.remove_sample for s in flat] == [False, False, True, True, True, True]
    assert [(s.reward, s.remove_sample) for s in all_masked] == [(0.0, True), (0.0, True)]


def test_reward_func_reads_the_run_reward(hooks) -> None:
    assert asyncio.run(hooks.reward_func(None, sample(0.5))) == 0.5
    assert asyncio.run(hooks.reward_func(None, [sample(0.5), sample(1.0)])) == [0.5, 1.0]


def trained(rollout_id, group, reward, removed=False):
    return SimpleNamespace(
        rollout_id=rollout_id,
        index=rollout_id,
        group_index=group,
        reward=reward,
        remove_sample=removed,
        get_reward_value=lambda args: reward,
    )


@pytest.mark.parametrize("std_on, spread", [(True, 0.5 / (0.5**0.5 + 1e-6)), (False, 0.5)])
def test_advantages_leave_masked_rollouts_out_of_the_group_mean_and_std(hooks, std_on, spread) -> None:
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=std_on)
    # Rollout 1 has two segments that share a reward and count once; rollout 2 is masked.
    samples = [trained(0, 0, 1.0), trained(1, 0, 0.0), trained(1, 0, 0.0), trained(2, 0, 0.5, removed=True)]
    samples += [trained(3, 1, 0.0, removed=True), trained(4, 1, 1.0)]
    raw, advantages = hooks.normalize_unmasked(args, samples)
    assert raw == [1.0, 0.0, 0.0, 0.5, 0.0, 1.0]
    assert advantages == pytest.approx([spread, -spread, -spread, 0.0, 0.0, 0.0])


def test_drop_masked_stops_training_after_consecutive_mostly_masked_rollouts(hooks) -> None:
    mostly_masked = lambda: [[sample(1.0), sample(0.0, masked=True), sample(0.0, masked=True)]]  # noqa: E731
    hooks.drop_masked(None, mostly_masked())
    hooks.drop_masked(None, [[sample(1.0), sample(0.0)]])  # a healthy rollout resets the count
    hooks.drop_masked(None, mostly_masked())
    hooks.drop_masked(None, mostly_masked())
    with pytest.raises(RuntimeError, match="3 rollouts in a row"):
        hooks.drop_masked(None, mostly_masked())
