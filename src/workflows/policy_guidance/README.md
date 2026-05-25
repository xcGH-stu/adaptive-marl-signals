# Policy Guidance Package

This package contains the lightweight, context-only policy-guidance scaffold for `policy_method/`.

Scope of this scaffold:

- Do not control actions directly.
- Do not modify the learner.
- Do not add policy loss.
- Do not change the main reward workflow yet.
- Only collect or mock behavior evidence, let a policy-side Critic synthesize it, and pass the resulting Integrated Guidance Card into reward-side payload construction.

Design rules:

- Raw behavior evidence goes to the policy Critic first.
- The policy Critic produces an Integrated Guidance Card.
- The reward Generator reads only the integrated card, not the raw behavior stats.
- If policy evidence is missing or policy guidance fails, the workflow must fall back to reward-only.

Intervention points:

- `G1`: `stage1_after_sparse_baseline`
- `G2`: `stage3_c1_pre_checkpoint`
- `G3`: `stage3_c1_after_round1`
- `G4`: `stage3_c2_pre_checkpoint`
- `G5`: `stage3_c2_after_round1`

Sampling plan:

- Stage 1 sparse baseline: milestones `200k`, `500k`, `800k`, `final(2050000)`, with `10` eval episodes each.
- C1 pre-checkpoint: `10` eval episodes.
- C1 after round-1: `5` eval episodes per branch.
- C2 pre-checkpoint: `10` eval episodes.
- C2 after round-1: `10` eval episodes per branch.

First-pass support:

- The first implementation target is `lbforaging`.
- Unsupported envs must return `insufficient_evidence`.
- A first-pass LBF mock-state behavior collector is now implemented.
- The collector expects event-style inputs and can parse mock `env_state` dictionaries with field size, agent positions, food positions, actions, reward, and termination.
- Raw trajectories are not saved; the collector emits compact episode summaries and milestone summaries only.
- The current version is not wired into the real runner yet.

Current next integration target:

- A policy-guided small pilot after the tiny execute smoke artifact checks.

Completed so far:

- Stage 1 tiny behavior integration is implemented.
- Stage 1b prompt integration is implemented.
- Stage 1b production hook smoke is implemented.
- Stage 3 prompt integration mock is implemented for:
- `stage3_c1_pre_checkpoint`
- `stage3_c1_after_round1`
- `stage3_c2_pre_checkpoint`
- `stage3_c2_after_round1`
- These Stage 3 mocks remain context-only, do not launch training, and fall back to reward-only when evidence is missing.
- Stage 3 production hook smoke is implemented:
- The adaptive controller hook now runs at the real pre-checkpoint and after-round1 positions
- Stage 3 behavior summaries can be supplied by file or object context
- The real candidate-generation payload now carries a nested `policy_guidance` block with the Integrated Guidance Card and evidence keys
- Disabled or missing policy guidance remains reward-only compatible and does not interrupt the controller
- A policy-guided tiny end-to-end smoke is implemented:
- It chains Stage 1 behavior summary generation, Stage 1b production-hook guidance, a mock selected dense source checkpoint, and Stage 3 C1/C2 production-hook guidance inside one workflow-level manifest and memory example
- The tiny smoke remains deterministic, does not call a real LLM, does not launch formal 850k Stage 1b runs, and does not launch formal 300k Stage 3 branches
- A policy-guided real-LLM dry-run path is implemented:
- It reuses the same five intervention points, strict schema validation, repair retry, cache layout, payload construction, and candidate mock generation
- It is API-key gated and still does not launch any training
- A policy-guided tiny execute smoke is implemented:
- It reuses the successful real-LLM dry-run cards, payloads, candidate examples, and cache artifacts
- It emits an execute-or-mock manifest and memory summary without launching formal long training

Read this package before wiring any real behavior collector or production integration.
