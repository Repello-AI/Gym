#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up and run the ipi_attacker Miles smoke on one GPU pod started from Miles' image, with the Gym servers in the
# same container. Safe to re-run: every step skips what is already done.
#
# The pod's image must be radixark/miles (the Megatron backend needs its patched SGLang and Megatron-LM):
#   radixark/miles:dev-202611181218   (the first build after the tested Miles commit)
#
# Upload to the pod first: ipi_envs_rl-0.2.1.zip, ipi-attacker-data.zip (or its folder), and a keys file with
# OPENAI_API_KEY and ANTHROPIC_API_KEY.
#
# Usage, as root or a user with passwordless sudo, inside the pod (in tmux: the smoke runs for a while):
#   bash setup_pod.sh --zip ipi_envs_rl-0.2.1.zip --data ipi-attacker-data.zip --keys keys.env
#   bash setup_pod.sh ... --smoke rollout    # paid: one rollout, no training (8 episodes)
#   bash setup_pod.sh ... --smoke train      # paid: three GRPO steps (24 episodes)
# Add --small on a GPU under 80 GB: Qwen3-1.7B with thinking off instead of Qwen3-4B-Instruct-2507.
set -euo pipefail

ZIP_SHA=a8f5e8ff51f56abfc0bd7420543c0491225cd637665ad492fe13be84a1792839
MILES_COMMIT=b55c2a533f
GYM_REPO=https://github.com/Repello-AI/Gym.git
GYM_REF=feat/ipi-attacker-own-envs
MODEL=Qwen3-4B-Instruct-2507
MODEL_ARGS=()

ZIP="" DATA="" KEYS="" SMOKE="" WORK="" SMALL_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --zip) ZIP=$(realpath "$2"); shift 2 ;;
        --data) DATA=$(realpath "$2"); shift 2 ;;
        --keys) KEYS=$(realpath "$2"); shift 2 ;;
        --smoke) SMOKE="$2"; shift 2 ;;
        --workdir) WORK="$2"; shift 2 ;;
        --gym-ref) GYM_REF="$2"; shift 2 ;;
        --small)
            MODEL=Qwen3-1.7B SMALL_FLAG=" --small"
            MODEL_ARGS=(--megatron-model-type qwen3-1.7B --hf-checkpoint Qwen/Qwen3-1.7B --model-name Qwen3-1.7B
                --apply-chat-template-kwargs '{"enable_thinking": false}')
            shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$ZIP" ] && [ -n "$DATA" ] && [ -n "$KEYS" ] || { sed -n '4,18p' "$0"; exit 2; }
case "$SMOKE" in ""|rollout|train) ;; *) echo "--smoke takes rollout or train" >&2; exit 2 ;; esac
# A pod's volume is usually /workspace or ~/workspace; else the home folder.
if [ -z "$WORK" ]; then
    for volume in /workspace "$HOME/workspace"; do
        if [ -d "$volume" ] && [ -w "$volume" ]; then WORK="$volume/ipi"; break; fi
    done
    WORK=${WORK:-$HOME/ipi}
fi
GYM="$WORK/Gym"
LOGS="$WORK/logs"
PY=$(command -v python3)  # the image's Python, with Miles; uv's Pythons are for Gym only
export PATH="$HOME/.local/bin:$PATH"

step() { echo; echo "=== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

step "Preflight"
if [ "$(id -u)" = 0 ]; then SUDO=""; elif sudo -n true 2>/dev/null; then SUDO=sudo; else die "needs root or passwordless sudo"; fi
echo "user: $(id -un), home: $HOME"
nvidia-smi -L || die "no GPU visible"
# Before Hopper, Transformer Engine's cuDNN kernel rejects packed sequences and FlashAttention 4 fails backward.
if [ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)" -lt 9 ]; then
    MODEL_ARGS+=(--attention-backend unfused)
fi
$SUDO test -d /root/miles && "$PY" -c "import miles, sglang, megatron.core" 2>/dev/null \
    || die "$PY cannot import miles, sglang and megatron: start the pod from radixark/miles:dev-202611181218, as root"
[ "$(sha256sum "$ZIP" | cut -d' ' -f1)" = "$ZIP_SHA" ] || die "$ZIP is not ipi_envs_rl 0.2.1 (sha mismatch)"
for name in OPENAI_API_KEY ANTHROPIC_API_KEY; do
    grep -q "^${name}=." "$KEYS" || die "$KEYS has no $name"
done
mkdir -p "$WORK" "$LOGS"
shm_gb=$(df -BG --output=size /dev/shm | tail -1 | tr -dc 0-9)
[ "$shm_gb" -ge 16 ] || echo "WARNING: /dev/shm is ${shm_gb}G; Miles' docs run with 32G. Raise it in the pod settings if Ray or NCCL fails."
free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 80 ] || echo "WARNING: ${free_gb}G free in $WORK; the model, its converted copy and the Gym venvs need about 80G."
echo "workdir: $WORK"

step "System packages and uv"
missing=""
for tool in git curl tmux unzip; do command -v "$tool" >/dev/null || missing="$missing $tool"; done
if [ -n "$missing" ]; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $missing; fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13

step "Gym checkout at $GYM_REF"
if [ ! -d "$GYM/.git" ]; then git clone -q -b "$GYM_REF" "$GYM_REPO" "$GYM"; else git -C "$GYM" pull -q --ff-only; fi
echo "Gym commit: $(git -C "$GYM" rev-parse --short HEAD)"
mkdir -p "$GYM/resources_servers/ipi_attacker/vendor"
cp "$ZIP" "$GYM/resources_servers/ipi_attacker/vendor/ipi_envs_rl-0.2.1.zip"
(cd "$GYM" && { [ -d .venv ] || uv venv -q --python 3.13; } && uv sync -q)
install -m 600 "$KEYS" "$WORK/keys.env"
sed "s#^ipi_attacker_api_key_file:.*#ipi_attacker_api_key_file: $WORK/keys.env#" \
    "$GYM/environments/ipi_attacker/miles/env.yaml.example" > "$GYM/env.yaml"

step "Data"
if [ -d "$DATA" ]; then
    [ "$DATA" = "$WORK/ipi-attacker-data" ] || cp -r "$DATA" "$WORK/ipi-attacker-data"
else
    unzip -qo "$DATA" -d "$WORK"
fi
[ -s "$WORK/ipi-attacker-data/train.jsonl" ] || die "no train.jsonl in $WORK/ipi-attacker-data"
wc -l "$WORK/ipi-attacker-data/train.jsonl" "$WORK/ipi-attacker-data/validation.jsonl"

step "Hugging Face cache on the work volume"
mkdir -p "$WORK/hf-cache" "$HOME/.cache"
if [ ! -L "$HOME/.cache/huggingface" ]; then
    [ -d "$HOME/.cache/huggingface" ] && cp -rn "$HOME/.cache/huggingface/." "$WORK/hf-cache/" && rm -rf "$HOME/.cache/huggingface"
    ln -s "$WORK/hf-cache" "$HOME/.cache/huggingface"
fi

step "Miles at $MILES_COMMIT"
if [ "$($SUDO git -C /root/miles rev-parse --short=10 HEAD)" != "$MILES_COMMIT" ]; then
    $SUDO git -C /root/miles fetch -q origin
    $SUDO git -C /root/miles checkout -q "$MILES_COMMIT"
    (cd /root/miles && $SUDO "$PY" -m pip install -q -e . --no-deps)
fi
"$PY" -c "import miles, typer, httpx, transformers; print('miles', miles.__file__)"

step "Gym servers"
gym_ready() {
    [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:12000/run \
        -H 'content-type: application/json' -d '{}')" = 422 ]
}
if ! gym_ready; then
    tmux has-session -t gym 2>/dev/null && tmux kill-session -t gym
    tmux new-session -d -s gym "cd '$GYM' && export PATH='$PATH' && .venv/bin/gym env start \
        --environment ipi_attacker --model-type vllm_model \
        '++ipi_attacker_agent.responses_api_agents.simple_agent.port=12000' \
        '++ipi_attacker_agent.responses_api_agents.simple_agent.allow_policy_base_url=true' \
        '++ipi_attacker_agent.responses_api_agents.simple_agent.max_model_retries=0' 2>&1 | tee '$LOGS/gym.log'"
    echo "starting (the first start builds each server's venv; up to 30 min)"
    for _ in $(seq 180); do
        gym_ready && break
        tmux has-session -t gym 2>/dev/null || { tail -30 "$LOGS/gym.log"; die "Gym exited; log above"; }
        sleep 10
    done
    gym_ready || { tail -30 "$LOGS/gym.log"; die "Gym not ready after 30 min"; }
fi
echo "Gym ready on 127.0.0.1:12000 (tmux attach -t gym; log $LOGS/gym.log)"

run_smoke() {
    local log="$LOGS/smoke-$1.log"
    shift
    cd "$GYM/environments/ipi_attacker/miles"
    "$PY" run.py --base-dir "$WORK/models" --ref-load "$WORK/models/${MODEL}_torch_dist" \
        --save-dir "$WORK/checkpoints/${MODEL}_ipi_attacker/" --prompt-data "$WORK/ipi-attacker-data/train.jsonl" \
        "${MODEL_ARGS[@]}" \
        "$@" 2>&1 | tee "$log"
    echo "log: $log"
}
case "$SMOKE" in
    rollout) step "Smoke: one rollout, no training (paid)"; run_smoke rollout --mode debug_rollout_only --num-rollout 1 ;;
    train) step "Smoke: three GRPO steps (paid)"; run_smoke train ;;
    *) step "Set up. Next, in tmux:"
       echo "  bash $0 --zip $ZIP --data $DATA --keys $KEYS$SMALL_FLAG --smoke rollout"
       echo "  bash $0 --zip $ZIP --data $DATA --keys $KEYS$SMALL_FLAG --smoke train" ;;
esac
