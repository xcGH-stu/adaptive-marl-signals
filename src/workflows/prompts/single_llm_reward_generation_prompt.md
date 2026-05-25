Return strict JSON only.

You are the only LLM used in a MARL reward-generation baseline.

Your job:
- Generate exactly 3 candidate PBRS-v2 structured reward configs for Level-Based Foraging.
- Use only the provided sparse-baseline context and the allowed PBRS-v2 schema.
- Do not output Python code.
- Do not act as a critic.
- Do not mention policy guidance, behavior summaries, critic diagnosis, Stage3, no_change, winner promotion, or adaptive replacement.
- `pbrs_mode` must be a clean runtime enum, not a baseline-mode alias string.
- `candidate_type` is required for every candidate.
- Every candidate object must include `candidate_type`; responses that omit it are invalid.

Candidate design goals:
- Keep candidates diverse across the allowed baseline modes.
- Use a clear `candidate_type` label chosen from:
  `balanced`, `exploration`, `collection_readiness`, `coverage_recovery`, `stability_recovery`, `allocation_rebalance`, `conservative_reference`
- Do not omit `candidate_type` for any candidate.
- Keep `alloc` and `stab` at `0.0`.
- Keep `beta` within `[0.2, 0.7]`.
- Use only valid LBF PBRS-v2 active terms and normalized weights.

Return one JSON object with:
{
  "candidates": [
    {
      "candidate_id": "...",
      "candidate_type": "balanced|exploration|collection_readiness|coverage_recovery|stability_recovery|allocation_rebalance|conservative_reference",
      "pbrs_version": "lbf_pbrs_v2",
      "baseline_mode": "balanced_progress|early_discovery|collection_readiness|coverage_recovery",
      "pbrs_mode": "balanced_collection_ready|coverage_ready_balance|approach_collection_push|allocation_stability_support",
      "beta": 0.5,
      "active_terms": ["col", "app", "cov", "ready"],
      "weights": {
        "col": 0.25,
        "app": 0.25,
        "cov": 0.25,
        "ready": 0.25,
        "alloc": 0.0,
        "stab": 0.0
      },
      "rationale": "...",
      "expected_effect": "...",
      "risk_notes": "..."
    }
  ],
  "notes": "short optional note"
}
