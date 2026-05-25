Return strict JSON only.

Role:
- You are the Generator for Stage 1b round-2 early dense candidate revision.

Goal:
- Propose exactly 3 revised PBRS candidates using sparse diagnosis, round-1 outcomes, and Critic advice.

Constraints:
- No markdown.
- No prose outside JSON.
- beta and wc must be within [0,1].
- wp must equal 1 - wc.
- Avoid duplicating round-1 candidates unless Critic explicitly asks to preserve a stable reference-like control.
- The objective is early dense initialization quality under a fixed 800000-step budget.

Required JSON keys:
- round_goal
- revision_rationale
- candidates

Each candidate must include:
- candidate_id
- beta
- wc
- wp
- candidate_type
- hypothesis
- expected_early_effect
- risk
