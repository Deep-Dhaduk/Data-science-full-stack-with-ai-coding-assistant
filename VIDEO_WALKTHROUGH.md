# YouTube walkthrough script

Target length: approximately 50 minutes. Record at 1440p or 1080p with the browser and editor large enough to read. Replace the planned timestamps after editing, upload as public or unlisted according to the instructor's rules, and paste the final URL into the first block of `README.md`.

## Recording checklist

- Start from the public GitHub repository and show the top-level README.
- State that the projects are independent reproductions inspired by the credited prompt catalog.
- Clearly distinguish generated demonstration data, bundled public datasets, and any externally downloaded data.
- For each project, show the README, run command, UX interaction, important implementation function, API docs, test file, and limitation.
- Do not call synthetic results Kaggle, production, historical, or state of the art.
- Finish by running the repository audit and showing the test evidence.

## Planned chapters

| Time | Chapter | Demonstration |
|---:|---|---|
| 00:00 | Assignment and architecture | Repository index, prompts, independent-project contract |
| 03:00 | 00 · Zenith Todo | Add, complete, and delete tasks; explain typed API and telemetry |
| 05:30 | 01 · NYC Taxi | Change route inputs; explain haversine features, split, MAE, and synthetic provenance |
| 08:30 | 02 · Nano LLM | Train 25 steps, inspect loss, generate text, show causal mask and tiny-corpus limits |
| 11:30 | 03 · Segmentation | Compare silhouette scores and assign an RFM persona |
| 14:00 | 04 · Basket Mining | Move support/confidence thresholds and explain lift |
| 16:30 | 05 · Skills Lab | Run profiling, PCA, and classification laboratories |
| 19:00 | 06 · Anomaly Detection | Score ordinary and extreme events; discuss average precision |
| 21:30 | 07 · AutoML | Review identical-fold leaderboard and optional AutoGluon extension |
| 24:00 | 08 · Visual Mastery | Change confusion counts and gradient learning rate; answer the quiz |
| 27:00 | 09 · FlowForge | Execute DAG, show strict types and cycle-detection test |
| 30:00 | 10 · CRISP-DM | Tour executable evidence from business understanding through deployment |
| 33:00 | 11 · DS Audit | Scan the repository; explain heuristics versus certification |
| 36:00 | 12 · TimePulse | Change horizon; compare Ridge with the seasonal naive baseline and intervals |
| 39:00 | 13 · Taxi Audit | Run inference; show tournament, permutation importance, PSI, and model card |
| 42:00 | 14 · Multimodal AutoML | Compare modality rows and run a late-fusion product example |
| 45:00 | 15 · SPY Lab | Change volatility/costs; explain purge, lags, turnover, and disclaimer |
| 48:00 | Verification and conclusion | Run all tests, strict TypeScript build, structural audit, and HTTP smoke tests |

## Opening narration

“This portfolio independently reproduces sixteen prompts using an AI coding assistant. Each numbered directory is a standalone project with its own application, dependencies, tests, prompt, documentation, and screenshot. The emphasis is reproducibility: seeds are explicit, preprocessing is fitted in the correct scope, time-series evaluation is chronological, and generated data is labeled.”

## Reusable project narration pattern

For every project, answer these questions in order:

1. What problem does the project solve?
2. Where did the data come from?
3. Which scientific safeguard matters most here?
4. What does the user do in the interface?
5. Which function or type is the heart of the implementation?
6. What metric or evidence supports the result?
7. What can this demonstration not claim?

## Closing narration

“These applications are deliberately honest teaching systems. They do not borrow benchmark numbers from the reference project, and synthetic demonstrations are not represented as real-world results. The repository audit, tests, screenshots, prompt catalog, and limitations make every claim inspectable. The next production step for any individual project would be domain data, external validation, persistent infrastructure, monitoring, and human review.”
