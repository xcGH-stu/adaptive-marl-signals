Return strict JSON only.

You are the same single LLM continuing a reward-generation baseline.

Your job:
- Read the round-1 candidate configs and their scalar short-run results.
- Propose exactly 3 refined PBRS-v2 candidates for round 2.
- Use only structured PBRS-v2 JSON.
- Do not act as a critic or winner selector.
- Do not introduce policy guidance, behavior summaries, critic diagnosis, no_change controls, Stage3 logic, checkpoint continuation, or adaptive replacement.
- `pbrs_mode` must be a clean runtime enum, not a baseline-mode alias string.
- `candidate_type` is required for every candidate and must not be dropped during refinement.
- Every refined candidate object must include `candidate_type`; responses that omit it are invalid.

Refinement goals:
- Respond to round-1 scalar evidence only.
- Improve short-run AUC and stable sparse-return behavior.
- Keep candidates diverse across the allowed baseline modes when possible.
- Keep a clear `candidate_type` label chosen from:
  `balanced`, `exploration`, `collection_readiness`, `coverage_recovery`, `stability_recovery`, `allocation_rebalance`, `conservative_reference`
- Preserve `candidate_type` for every candidate when refining round-1 outputs.
- Keep `alloc` and `stab` at `0.0`.
- Keep `beta` within `[0.2, 0.7]`.

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
