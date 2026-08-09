# Initial research protocol

The first empirical MNEL study should ask whether structured experimental learning
outperforms search volume alone under equal budgets.

## Arms

- A0: random bounded candidate proposals;
- A1: aggregate scores only;
- A2: complete episodes without causal synthesis;
- A3: episodes plus competing hypotheses and probes;
- A4: attribution plus provisional principles;
- A5: transfer-gated strategies;
- A6: bounded policy recursion over investigator and probe selection.

## Equal resources

Each arm receives the same candidate count, operation budget, evaluator, hard gates,
development environments, selection material, and hidden transfer environment.

## Measurements

- improvement per candidate and operation;
- accepted update rate;
- rollback equality;
- duplicate proposal rate;
- prediction calibration;
- retention and transition preservation;
- transfer performance;
- evidence reuse depth;
- hypothesis diversity;
- verifier disagreement and `UNKNOWN` rate;
- time to stagnation; and
- authority-violation attempts.

The study must preserve every failed candidate and must not use transfer or final
observations for same-candidate repair.

## Machine-readable study contracts

MNEL now represents each arm as an identity-bound `mnel-study-arm/0.4` record with
allowed/forbidden information, retrieval and memory modes, attribution/strategy
availability, recursion allowance, and an equal operation/wall-time/candidate budget.
Control transformations are separate `mnel-ablation-spec/0.4` records. Shuffled
attribution uses a recorded seed and preserves the original attribution records; memory
ablations change only the eligible view. The reference study can therefore reproduce
the same control identities without deleting source evidence or reading hidden transfer
outcomes during development.

The current 0.4 reference study is deliberately synthetic and deterministic. Its
transition-frequency provider is a diagnostic observation source with explicit artifact,
training dataset, feature extractor, calibration, and reload identities. It is not a
verifier, evaluator, or promotion mechanism.
