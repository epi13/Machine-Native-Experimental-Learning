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
