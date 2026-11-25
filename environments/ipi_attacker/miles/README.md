# Training the ipi_attacker policy with Miles

Miles' example Gym adapter (`examples/experimental/nemo-gym/nemogym_agent_function.py`) breaks this environment twice:
it sends `/run` with `responses_create_params.input = []`, so the attacker never sees its prompt, and it keeps only
`reward`, so a judge or API failure trains as a real 0. The hooks here replace it. No Miles change is needed: Miles
imports hooks by dotted path, so this folder only has to be on `PYTHONPATH`.

| File | What it is |
|---|---|
| `ipi_attacker_miles.py` | The hooks: `run` (agent function), `reward_func` (reward hook), `drop_masked` (sample filter), `normalize_unmasked` (reward post-process) |
| `run.py` | A Miles launcher sized for a smoke run on one 80 GB GPU, with a data preflight |
| `env.yaml.example` | The Gym `env.yaml` for training behind Miles |
| `api_scan.py` | Sends a few rows through the running Gym servers with an API model as the attacker (paid) |
| `setup_pod.sh` | Pre-handoff test: sets up Gym and Miles in one GPU pod started from Miles' image, and runs the smoke |

## What the hooks do

- `run` sends the row's `prompt` as the attacker's input, the row's `metadata` (`environment`, `env_id`, ...) at the
  top level, and the Miles session URL as `policy_base_url`. It returns `reward`, `mask_sample` and `failure_kind`.
  It never raises: Miles would log the exception and train the sample as reward 0. A timeout, transport error,
  non-2xx response, or simple_agent's policy-timeout row (HTTP 200, no reward) comes back masked, with
  `failure_kind` `ipi_attacker:run_failed`, `ipi_attacker:run_http_<status>`, `ipi_attacker:bad_response` (a body
  that is not an object or a reward that is not a finite number), or the row's own kind. Only a refused connection
  is retried, since any later failure may have started a paid episode.
- `reward_func` reads `metadata["reward"]`.
- `drop_masked` sets `remove_sample` on each masked sample, including one with no reward at all (its agent function
  died, so Miles kept the recorded tokens without `run`'s result). The placeholder reward becomes the mean of the
  group's unmasked samples, so the logged raw reward is not pulled down by failures. It also stops training: when
  more than `IPI_MASK_BREAKER_RATE` (0.5) of a rollout's samples are masked for `IPI_MASK_BREAKER_ROLLOUTS` (3)
  rollouts in a row, it raises, since a dead key, judge or Gym server would otherwise train on nothing.
- `normalize_unmasked` computes the GRPO advantages; Miles calls a reward post-process in place of its own
  normalization. That normalization counts every sample in a group's std, even with `remove_sample`, so masked
  samples would scale the others' advantages with the failure rate. This hook takes each group's mean and std over
  unmasked rollouts only (as NeMo RL's `valid_mask` does), gives masked samples 0, counts a rollout's segments once,
  and honours `--disable-grpo-std-normalization`. Miles' `--length-penalty-coeff` does not apply through it.

Written against Miles `b55c2a533f`. The hook signatures are Miles' internal contract, so pin the Miles commit.

## What the GPU machine needs

These are not in git:

- `ipi_envs_rl-0.2.1.zip`, sha256 `a8f5e8ff51f56abfc0bd7420543c0491225cd637665ad492fe13be84a1792839`.
- `ipi-attacker-data.zip` (or its folder) with `train.jsonl` and `validation.jsonl`.
- A keys file with `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` (plus `SARVAM_API_KEY` or `OPENROUTER_API_KEY` for
  those defenders).

The machine also needs outbound access to GitHub, PyPI, Hugging Face (the model downloads on first use) and the
OpenAI and Anthropic APIs.

## Setup on one GPU box

The Gym servers run on the host (Gym needs Python >= 3.13.14); Miles runs in its docker image with `--network host`,
so both reach each other on localhost.

1. Gym, on the host, at the PR branch of the Gym fork:

   ```bash
   git clone -b feat/ipi-attacker-own-envs https://github.com/Repello-AI/Gym.git && cd Gym
   uv venv --python 3.13 && source .venv/bin/activate && uv sync
   mkdir -p resources_servers/ipi_attacker/vendor
   cp /path/to/ipi_envs_rl-0.2.1.zip resources_servers/ipi_attacker/vendor/
   shasum -a 256 resources_servers/ipi_attacker/vendor/ipi_envs_rl-0.2.1.zip   # must match the sha in uv.lock
   cp environments/ipi_attacker/miles/env.yaml.example env.yaml                 # set ipi_attacker_api_key_file
   ```

   The keys file holds `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` (plus `SARVAM_API_KEY` or `OPENROUTER_API_KEY` for
   those defenders).

2. Start the Gym servers (in tmux; the first start builds each server's venv):

   ```bash
   gym env start --environment ipi_attacker --model-type vllm_model \
       '++ipi_attacker_agent.responses_api_agents.simple_agent.port=12000' \
       '++ipi_attacker_agent.responses_api_agents.simple_agent.allow_policy_base_url=true' \
       '++ipi_attacker_agent.responses_api_agents.simple_agent.max_model_retries=0'
   ```

   Ready when the startup table lists `ipi_attacker_agent` on 12000. `max_model_retries=0`: a retried policy call
   would record a second generation in the Miles session. Defender reasoning is off by default; add
   `'++ipi_attacker_resources_server.resources_servers.ipi_attacker.defender_thinking=true'` to turn it on.

3. Optional, paid, no GPU: check the Gym chain with an API model as the attacker. Set `policy_model_name` in
   `env.yaml` to an OpenAI model (e.g. `gpt-5.6-luna`), start Gym, then:

   ```bash
   cd environments/ipi_attacker/miles
   python api_scan.py --data /path/to/ipi-attacker-data/validation.jsonl --limit 5 --api-key-file /path/to/keys.env
   ```

   It exits non-zero unless every row is scored (not masked), the prompt reached the attacker and the attacker read
   thousands of tokens. Restart Gym with `policy_model_name: model` before training.

4. Miles, in its image, with the Gym checkout and the data mounted:

   ```bash
   docker run --rm --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
       --network=host -v /path/to/Gym:/root/Gym -v /path/to/ipi-attacker-data:/root/ipi-attacker-data \
       -it radixark/miles:dev-202611181218 /bin/bash
   # inside: cd /root/miles && git fetch origin && git checkout b55c2a533f && pip install -e . --no-deps
   ```

## Smoke run (Qwen3-4B-Instruct-2507, one H100 80 GB)

Inside the container, from `/root/Gym/environments/ipi_attacker/miles`. `run.py` first drops blocked and
`audit_only` rows and checks every prompt fits `max_seq_len - rollout_max_response_len` with the model's chat
template; then it converts the checkpoint on first run (`--skip-prepare` to skip) and launches Miles. Like Miles'
example launcher, it first kills every process matching `sglang`, `train.py` or `MegatronTrain`: run it only inside
the container, never on a shared host.

1. One rollout, no training (8 episodes, paid):

   ```bash
   python run.py --mode debug_rollout_only --num-rollout 1
   ```

   Pass: 8 samples with prompt tokens in the thousands; rewards from `/run`; at most 1 masked, and each masked one
   counted in the `drop_masked` log line; no sample with status TRUNCATED.

2. Three GRPO steps (24 episodes, paid):

   ```bash
   python run.py
   ```

   Pass: 3 steps complete, weights pushed to SGLang each step, `rollout/raw_reward` logged, at least one unmasked
   sample per step. `rollout/zero_std` firing is expected when a small attacker never breaks the defender.

Out of memory: `--megatron-model-type qwen3-1.7B --hf-checkpoint Qwen/Qwen3-1.7B --model-name Qwen3-1.7B
--ref-load /root/Qwen3-1.7B_torch_dist`.

## Flags for another launcher

Put this folder on `PYTHONPATH` (next to Megatron and Miles), set `NEMO_GYM_URL` (default `http://localhost:12000`),
optionally `NEMO_GYM_RUN_TIMEOUT` (seconds per `/run`, default 3600) and the two breaker variables above, in the
environment of Miles' rollout processes (the Ray runtime env), and pass:

```bash
--input-key prompt --metadata-key metadata
--advantage-estimator grpo
--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path ipi_attacker_miles.run
--custom-rm-path ipi_attacker_miles.reward_func
--rollout-sample-filter-path ipi_attacker_miles.drop_masked
--custom-reward-post-process-path ipi_attacker_miles.normalize_unmasked   # replaces Miles' GRPO normalization
--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_aborted_filter
--use-session-server --session-server-ip 127.0.0.1
--tito-model qwen3                   # the policy model's TITO family
```

Gym and the session server bind to loopback, because both start paid episodes without authentication and
`allow_policy_base_url` lets a caller pick the URL the agent posts to. When Gym runs on another host than the
trainer, bind the session server wider (`--session-server-ip 0.0.0.0`, with `--session-server-external-host <IP>`
when the Gym host cannot resolve the trainer's hostname) and Gym (`default_host` in `env.yaml`), and firewall both
ports to the two hosts.

## Scaling to the main run

- Model and parallelism: `--megatron-model-type`, `--hf-checkpoint`, `--tensor-model-parallel-size`, GPUs per node.
- `--n-samples-per-prompt 8` gives GRPO a usable spread. Episodes in flight are
  `rollout_batch_size x n_samples_per_prompt`; keep that within the resources server's `max_concurrent_episodes`
  (32) and `--session-server-workers`, or raise both.
- Each episode is several defender calls plus judge calls, so API cost and rate limits scale with episodes.
- `--rollout-max-response-len 1024` covers the longest fill limit (2000 characters); a longer fill scores a real 0.

## Known limits

- `rollout/raw_reward` includes masked samples at their substituted reward.
- Miles regenerates an aborted group until the batch fills, so the paid episode count is not capped. A Gym failure
  before the policy call (no tokens recorded) aborts the sample, so it loops there rather than tripping the breaker.
- A `/run` past `NEMO_GYM_RUN_TIMEOUT` is masked, but its Gym episode keeps running and spending until it ends.
- simple_agent's client resends `/verify` after a dropped local connection, which runs that episode twice.
- Don't enable sample dumps that log `metadata`: it holds the environment's answer side.
