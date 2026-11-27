# ipi_attacker

Single-turn indirect prompt injection attacker. The policy sees the attacker's goal and the conversation the defender
will have, and writes the text for one untrusted record. A frozen defender (`gpt-6-luna` by default) then does its
tool-use task over that record; the reward says whether it took the unwanted action and hid it.
See `resources_servers/ipi_attacker/README.md` for the reward, the keys it needs and where to put the delivered
`ipi_envs_rl-0.2.2.zip`.

- Integration profile: `custom-gym-verifier`
- Agent: `simple_agent` (one policy call, no tools)
- Scorer: `resources_servers/ipi_attacker`

## Data

`data/example.jsonl` holds 5 rows, one per world, from the evaluation split. The train and validation sets are
delivered separately; add them as datasets with their `jsonl_fpath` when you have them. Each row is the environment's
training row: `responses_create_params` (the attacker's prompt, built with the v2 attacker system prompt), `env_id`,
`split` and `environment` (read by `verify`, never shown to the policy).

## Training with Miles

Miles records each episode through its own per-session URL, which it sends on `/run` as `policy_base_url`. Set
`allow_policy_base_url: true` on `ipi_attacker_agent` (the `simple_agent` in `config.yaml`) for that deployment; the
agent refuses the field otherwise. Miles' example Gym adapter sends `/run` with an empty input and drops
`mask_sample`; use the hooks in `miles/` instead. `miles/README.md` has the setup, the launch steps and the flags.
