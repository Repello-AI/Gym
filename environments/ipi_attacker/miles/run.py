# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Miles launcher for the ipi_attacker environment, sized for a smoke run on one 80 GB GPU.

Runs inside Miles' image, with the Gym servers already started on the same machine (the host of a
``--network host`` container, or the same pod).
Defaults: Qwen3-4B-Instruct-2507, 3 GRPO steps of 2 prompts x 4 samples.

Usage:
    python run.py --mode debug_rollout_only --num-rollout 1
    python run.py
    python run.py --megatron-model-type qwen3-1.7B --hf-checkpoint Qwen/Qwen3-1.7B --model-name Qwen3-1.7B \
        --ref-load /root/Qwen3-1.7B_torch_dist
"""

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from ipi_attacker_miles import DEFAULT_GYM_URL, DEFAULT_RUN_TIMEOUT_S
from miles.utils.external_utils import command_utils


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    megatron_model_type: str = "qwen3-4B-Instruct-2507"
    num_gpus_per_node: int = 1
    tensor_parallel: int = 1
    megatron_path: str = "/root/Megatron-LM"

    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "Qwen3-4B-Instruct-2507"
    hf_checkpoint: str = "Qwen/Qwen3-4B-Instruct-2507"
    ref_load: str = "/root/Qwen3-4B-Instruct-2507_torch_dist"
    save_dir: str = "/root/Qwen3-4B-Instruct-2507_ipi_attacker/"
    prompt_data: str = "/root/ipi-attacker-data/train.jsonl"

    # Prompts are up to about 6k tokens; a fill is capped at 2000 characters.
    max_seq_len: int = 8192
    rollout_max_response_len: int = 1024
    num_rollout: int = 3
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    save_interval: int = 1000
    optimizer_cpu_offload: bool = True
    # Megatron's flash backend runs Hopper-only kernels; use unfused on an older GPU such as an A100.
    attention_backend: str = "flash"
    # JSON for the attacker's chat template, e.g. '{"enable_thinking": false}' for a Qwen3 hybrid-thinking model.
    apply_chat_template_kwargs: str = ""

    nemo_gym_url: str = os.environ.get("NEMO_GYM_URL", DEFAULT_GYM_URL)
    nemo_gym_run_timeout: str = os.environ.get("NEMO_GYM_RUN_TIMEOUT", DEFAULT_RUN_TIMEOUT_S)
    mask_breaker_rate: str = os.environ.get("IPI_MASK_BREAKER_RATE", "0.5")
    mask_breaker_rollouts: str = os.environ.get("IPI_MASK_BREAKER_ROLLOUTS", "3")
    # Empty binds Miles' default, the address Ray placed the server on; Miles checks readiness there, so not loopback.
    session_server_ip: str = ""
    session_server_external_host: str = ""


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true", shell=True)
    time.sleep(5)


def local_checkpoint(hf_checkpoint: str) -> str:
    """A local folder with the full checkpoint: the converter reads the weights from disk and never downloads them."""
    if Path(hf_checkpoint).is_dir():
        return hf_checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(hf_checkpoint)


def prepare(args: ScriptArgs):
    """Convert the HF checkpoint to torch_dist format if not already done."""
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def preflight(args: ScriptArgs) -> str:
    """Write the trainable rows to a new file and return its path; stop if a prompt leaves no room to reply.

    Miles skips its prompt-length filter for message-list prompts, so the check is done here.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
    rows = [json.loads(line) for line in Path(args.prompt_data).read_text().splitlines() if line.strip()]
    kept = [
        row
        for row in rows
        if not row["metadata"]["qualification"]["blocked"] and not row["metadata"].get("audit_only")
    ]
    budget = args.max_seq_len - args.rollout_max_response_len
    if not kept:
        raise SystemExit(f"no trainable rows in {args.prompt_data}")
    template_kwargs = json.loads(args.apply_chat_template_kwargs or "{}")
    # Render to text and count: newer transformers return an encoding dict from apply_chat_template.
    lengths = [
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    row["prompt"], add_generation_prompt=True, tokenize=False, **template_kwargs
                ),
                add_special_tokens=False,
            )
        )
        for row in kept
    ]
    too_long = [row["metadata"]["env_id"] for row, length in zip(kept, lengths) if length > budget]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed {budget} tokens (raise --max-seq-len): {too_long[:5]}")
    out = Path(args.save_dir) / "prompt_data.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in kept))
    print(f"preflight: {len(kept)}/{len(rows)} rows kept, longest prompt {max(lengths)} tokens, wrote {out}")
    return str(out)


def execute(args: ScriptArgs, prompt_data: str):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )

    template_args = (
        f"--apply-chat-template-kwargs {shlex.quote(args.apply_chat_template_kwargs)} "
        if args.apply_chat_template_kwargs
        else ""
    )
    session_ip_arg = f"--session-server-ip {args.session_server_ip} " if args.session_server_ip else ""
    rollout_args = (
        f"--prompt-data {prompt_data} "
        f"{template_args}"
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--custom-reward-post-process-path ipi_attacker_miles.normalize_unmasked "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel} "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_seq_len} "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    external_host_arg = (
        f"--session-server-external-host {args.session_server_external_host} "
        if args.session_server_external_host
        else ""
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path ipi_attacker_miles.run "
        "--custom-rm-path ipi_attacker_miles.reward_func "
        "--rollout-sample-filter-path ipi_attacker_miles.drop_masked "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_aborted_filter "
        "--use-session-server "
        f"{session_ip_arg}"
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        f"{external_host_arg}"
        "--tito-model qwen3 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--attention-backend {args.attention_backend} "
        "--colocate "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    args.create_backend().execute_train(
        train_args=(
            f"{ckpt_args}{rollout_args}{grpo_args}{optimizer_args}{perf_args}"
            f"{sglang_args}{agent_args}{misc_args}{debug_args}"
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        # These replace the launcher's own PYTHONPATH, so Megatron and Miles are listed too.
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
            "NEMO_GYM_URL": args.nemo_gym_url,
            "NEMO_GYM_RUN_TIMEOUT": args.nemo_gym_run_timeout,
            "IPI_MASK_BREAKER_RATE": args.mask_breaker_rate,
            "IPI_MASK_BREAKER_ROLLOUTS": args.mask_breaker_rollouts,
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    args.hf_checkpoint = local_checkpoint(args.hf_checkpoint)
    prompt_data = preflight(args)
    if not args.skip_prepare:
        prepare(args)
    execute(args, prompt_data)


if __name__ == "__main__":
    typer.run(main)
