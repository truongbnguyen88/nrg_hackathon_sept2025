# NRG Hackathon VS Code Starter – FlexTrack 2025

This repo is pre-configured for the [FlexTrack Challenge 2025](https://www.aicrowd.com/challenges/flextrack-challenge-2025).

## Quick Start
1. Open in VS Code. If prompted, click **Reopen in Container**.
2. Wait for dependencies to install (`requirements.txt`).
3. Put your exploration notebooks in `notebooks/`.
4. Put reusable functions/models in `src/`.
5. Train and generate predictions with:
   ```bash
   make train
   make predict
   make submit
   ```
6. Upload the CSV in `submissions/` to AIcrowd.

## FlexTrack Submission Format
### Phase 1 (Classification)
CSV must contain:
- `site_id`
- `timestamp`
- `demand_response_event` (0/1)

### Phase 2 (Regression)
CSV must contain:
- `site_id`
- `timestamp`
- `demand_response_capacity` (float, signed)

## Rules Recap
- No external data allowed.
- Max 10 submissions per participant per day.
- Use fixed random seeds for reproducibility.
