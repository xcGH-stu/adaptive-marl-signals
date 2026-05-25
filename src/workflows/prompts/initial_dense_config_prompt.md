You are selecting an initial dense/native-original PBRS configuration for an adaptive MARL workflow.

Your job:
- Choose one stable initial PBRS configuration using beta and wc, with wp derived as 1 - wc.
- Prefer a configuration that is likely to produce a complete learning progression suitable for later checkpoint-based adaptation, rather than selecting only for peak final score.
- Use the sparse trajectory summary as context for what kind of shaping may be needed early, mid, and late in training.

Rules:
- beta must be within [0, 1].
- wc must be within [0, 1].
- wp must equal 1 - wc.
- The selected beta must come from the provided candidate beta values.
- The selected wc must come from the provided candidate wc values.
- Return strict JSON only.

Return schema:
{
  "initial_config": {
    "beta": 0.3,
    "wc": 0.6,
    "wp": 0.4
  },
  "expected_role": "short phrase",
  "risk": "short phrase",
  "candidate_search_prior": {
    "beta_range": [0.1, 0.7],
    "wc_range": [0.3, 0.8]
  },
  "stage_hypotheses": {
    "early": "short phrase",
    "mid": "short phrase",
    "late": "short phrase"
  }
}

Do not include markdown. Do not include commentary outside JSON.
