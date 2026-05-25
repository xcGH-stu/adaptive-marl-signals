# Policy Guidance Placeholder

This directory reserves space for future policy-module workflow code in `policy_method/`.

Why this directory is named `policy_guidance_module/` instead of `policy_guidance/`:

- The copied reward-only baseline already contains `src/workflows/policy_guidance.py`.
- A same-level directory named `policy_guidance/` cannot coexist with that file.
- To keep the copied baseline intact in this setup round, this placeholder uses a non-conflicting name.

Planned phases:

## Phase 1: diagnosis-only policy module

- Read sparse baseline, Stage 1b, Stage 3, and adaptive mainline metrics.
- Summarize policy failure modes such as coordination failure, congestion, and weak exploration.
- Do not modify training.

## Phase 2: policy diagnosis as reward-generation context

- Feed policy Critic diagnosis into Stage 3 reward candidate generation as additional context.
- Still do not modify the learner.

## Phase 3: training-time policy guidance

- Consider stronger interventions later, such as auxiliary loss, action prior, role guidance, or curriculum hints.
