# adaptive-marl-signals
Anonymous implementation for adaptive learning-signal optimization in MARL
# Anonymous Code Release

This repository provides the anonymized implementation for the submitted paper on adaptive learning-signal optimization in multi-agent reinforcement learning.

## Overview

The code implements a staged framework for optimizing learning signals in cooperative multi-agent reinforcement learning. The main workflow combines structured reward-shaping configurations, sparse-evaluation feedback, short-horizon branch validation, and LLM-assisted candidate generation and diagnosis.

The repository is anonymized for double-blind review. Author names, institutional information, personal paths, server names, and private credentials have been removed.

## Main Components

- `src/`: core implementation of the learning-signal optimization workflow.
- `configs/`: experiment and baseline configuration files, if included.
- `prompts/`: prompt templates for LLM-based candidate generation and diagnosis, if included.
- `scripts/`: example scripts or command templates for running the main workflows, if included.

## Implemented Methods

The implementation includes the following components:

- Sparse task-reward training baseline.
- Fixed reward-shaping baseline.
- Single-LLM reward-generation baseline.
- Structured PBRS-v2 reward-shaping configuration space.
- Stage 1b initial dense reward search.
- Stage 3 adaptive branch validation.
- Mandatory no-change control during adaptive replacement.
- Winner-branch promotion for continuing the selected training trajectory.
- Sparse-only evaluation for reported task performance.

## Repository Scope

This repository is intended to support anonymous review by exposing the main implementation and configuration structure used in the submitted paper.

Full raw training logs, model checkpoints, and large intermediate experiment outputs are omitted due to storage size. The released files focus on the implementation, workflow logic, configuration structure, and prompt templates needed to inspect the method.

## Reproducibility Notes

The full experiments require substantial multi-agent reinforcement learning training. The repository is organized to make the workflow and implementation auditable, while large-scale logs and checkpoints will be released after the review period.

## Anonymity Notice

This repository is prepared for double-blind review. Please do not attempt to identify the authors from external metadata.
