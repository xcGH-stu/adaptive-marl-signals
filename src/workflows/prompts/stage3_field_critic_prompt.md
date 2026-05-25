# Stage 3 Field Round Critic Prompt

You are analyzing one Stage 3 field-validation round in a stage-conditioned reward workflow.

## Goal

For the given reward field, choose one candidate value for each training stage:

- `early_exploration`
- `mid_progress`
- `late_plateau`

You are not designing a whole reward function here. You are only selecting the best value
for the current field in each stage.

## Inputs

You will receive:

- `field_name`
- the shared candidate values evaluated in this round
- one entry per stage
- for each stage, a comparison table of short branch experiments

Each comparison-table row includes key metrics such as:

- `best_test_sparse_return_mean`
- `last_test_sparse_return_mean`
- `best_test_return_mean`
- `last_test_return_mean`
- `last_sparse_return_mean`
- `last_return_mean`

## Selection Principles

- Prioritize sparse-reward task success evidence first.
- Prefer `best_test_sparse_return_mean` as the main signal when it clearly separates candidates.
- Use `last_test_sparse_return_mean` as a stability signal.
- If sparse-test signals are tied or absent, fall back to the next most task-aligned metrics.
- Choose exactly one candidate value for each of:
  - `early_exploration`
  - `mid_progress`
  - `late_plateau`
- Base your choice only on the provided experimental evidence.

## Output Format

Return strict JSON with:

- `analysis_summary`
- `per_stage_reasoning`
- `recommended_values_by_stage`

Requirements:

- `per_stage_reasoning` must be an object keyed by the 3 stage labels
- `recommended_values_by_stage` must be an object keyed by the 3 stage labels
- each recommended value must be one of the provided shared candidate values

Return valid JSON and nothing else.
