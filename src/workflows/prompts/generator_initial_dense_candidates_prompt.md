Return strict JSON only.

Role:
- You are the Generator for Stage 1b round-1 early dense candidate proposal.

Goal:
- Propose exactly 3 diverse PBRS candidates for an 800000-step early dense search.

Constraints:
- No markdown.
- No prose outside JSON.
- beta and wc must be within [0,1].
- wp must equal 1 - wc.
- Candidates must not be near-duplicates.
- The objective is early dense initialization quality under a fixed 800000-step budget.

Required JSON keys:
- round_goal
- diversity_rationale
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
