# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier fixture: the real verify() with a scripted defender and concealment judge set by each case."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from envgen.concealment import CRITERION

from nemo_gym.client import ServerClient


class FixtureDefender:
    """Makes the given tool calls one per turn, then replies with text."""

    model = "fixture_defender"

    def __init__(self, calls, final="Done, the update is sent."):
        self.calls, self.final, self.turns = list(calls), final, 0

    def turn(self, *_, **__):
        self.turns += 1
        if self.turns <= len(self.calls):
            call = self.calls[self.turns - 1]
            block = SimpleNamespace(
                type="tool_use", id=f"call_{self.turns}", name=call["name"], input=call["arguments"]
            )
            return SimpleNamespace(content=[block], stop_reason="tool_use"), {}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.final)], stop_reason="end_turn"), {}


class FixtureJudge:
    """A concealment judge that gives every reply `score` out of 10, or fails with `error`."""

    model = "fixture_judge"

    def __init__(self, score=10, error=None):
        self.score, self.error = score, error

    def chat(self, messages, **_):
        if self.error:
            raise RuntimeError(self.error)
        return SimpleNamespace(content=json.dumps({CRITERION: {"score": self.score, "explanation": "fixture"}}))


def fixture_clients(environment, defender="text", concealment_score=10, judge_error=None):
    """`defender` "reference" makes the environment's winning calls; "text" only replies."""
    calls = environment["controls"]["reference"]["calls"] if defender == "reference" else []
    return FixtureDefender(calls), FixtureJudge(concealment_score, judge_error), None


def create_fixture_server():
    from resources_servers.ipi_attacker.app import IpiAttackerResourcesServer, IpiAttackerResourcesServerConfig

    class FixtureServer(IpiAttackerResourcesServer):
        def build_clients(self):
            return ()

    config = IpiAttackerResourcesServerConfig(
        host="127.0.0.1", port=0, entrypoint="app.py", name="ipi_attacker", api_key_file=""
    )
    return FixtureServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_with_fixture_defender(server, request):
    """Verify with the defender and judge the case names in `fixture_defender` and `fixture_concealment_score`."""
    server._clients = fixture_clients(
        request.environment,
        getattr(request, "fixture_defender", "text"),
        getattr(request, "fixture_concealment_score", 10),
    )
    return await server.verify(request)
