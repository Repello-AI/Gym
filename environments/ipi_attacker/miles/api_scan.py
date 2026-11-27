# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send a few data rows through the running Gym servers with an API model as the attacker. Paid.

Posts the same /run body as the Miles hook, with an OpenAI-compatible endpoint as policy_base_url. Exits non-zero
unless every row is scored (not masked), the prompt reached the attacker, and the attacker read thousands of tokens.

Usage:
    python api_scan.py --data validation.jsonl --limit 5 --api-key-file keys.env
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from ipi_attacker_miles import DEFAULT_GYM_URL, run_body


MIN_INPUT_TOKENS = 1000


def api_key(path: str) -> str:
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip("\"'")
    raise SystemExit(f"no OPENAI_API_KEY in {path}")


async def scan_row(client: httpx.AsyncClient, url: str, row: dict, policy_base_url: str, key: str) -> bool:
    body = run_body(policy_base_url, row["prompt"], {}, row["metadata"]) | {"policy_api_key": key}
    response = await client.post(url, json=body)
    if response.status_code != 200:
        # A 422 echoes the request body, key included.
        print(f"{row['metadata']['env_id']}: HTTP {response.status_code} {response.text[:300].replace(key, '***')}")
        return False
    result = response.json()
    tokens = ((result.get("response") or {}).get("usage") or {}).get("input_tokens", 0)
    # Gym normalizes each message (adds fields), so compare role and content only.
    echo = (result.get("responses_create_params") or {}).get("input") or []
    echoed = [(m.get("role"), m.get("content")) for m in echo] == [(m["role"], m["content"]) for m in row["prompt"]]
    ok = "reward" in result and not result.get("mask_sample") and echoed and tokens >= MIN_INPUT_TOKENS
    print(
        f"{row['metadata']['env_id']}: reward={result.get('reward')} mask={result.get('mask_sample')} "
        f"failure_kind={result.get('failure_kind')} input_tokens={tokens} prompt_echoed={echoed} "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="a jsonl in Miles' row format")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--policy-base-url", default="https://api.openai.com", help="without /v1")
    parser.add_argument("--gym-url", default=os.getenv("NEMO_GYM_URL", DEFAULT_GYM_URL))
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()][: args.limit]
    if not rows:
        raise SystemExit(f"no rows to scan in {args.data} with --limit {args.limit}")
    key = api_key(args.api_key_file)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
        results = await asyncio.gather(
            *(scan_row(client, f"{args.gym_url.rstrip('/')}/run", row, args.policy_base_url, key) for row in rows)
        )
    print(f"{sum(results)}/{len(results)} rows passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
