# One-step distilled specialist reference

This checked-in study is the first executable MNEL one-step distillation slice for
`forge.evidence-relevance`.

```text
bounded recurrent teacher
        -> retained teacher observations
        -> independently checked fixture targets
        -> tiny affine/tanh student
        -> one learned forward pass or ABSTAIN
        -> explicit recurrent teacher fallback
```

The files are generated with:

```bash
mnel one-step-specialist-reference --workspace examples/one-step-specialists
```

- `forge-teacher-recurrent-g0.json` is the existing bounded recurrent teacher artifact.
- `forge-student-distilled-g0.json` is the one-step student trained from teacher-derived
  decision/confidence targets.
- `forge-student-no-distillation-control-g0.json` uses the same architecture and rows,
  but trains only from direct fixture labels.
- `distillation-records.json` retains every teacher observation, including rejected,
  unknown, and abstaining cases, plus the selected training targets.
- `reference-study.json` reports the iterative teacher, distilled student, deterministic
  nearest-centroid baseline, no-distillation control, calibration, and fallback costs.

The measurements are diagnostic evidence, not a correctness or promotion verdict. The
synthetic fixture does not currently support a claim that distillation wins end-to-end.

## MNCS-native vertical slice

The Python artifacts above remain the independent reference and control study. A
separate bounded native slice is checked in at:

- `../../mncs/source/mnel/one_step.mncs` — fixed-point dataset construction, eight-epoch
  model-state updates, calibration envelope, lineage fields, gated inference, and
  diagnostic-only output;
- `../../mncs/source/mnel/one_step-corpus.json` — stateful `train_artifact -> infer ->
  decision_code` cases for an in-domain query, OOD input, generation mismatch, and
  schema mismatch;
- `../../tools/run_mnel_native_one_step.py` — five-backend execution/evidence runner;
- `../../docs/mncs-reconstruction/evidence/mnel-native-one-step-study.json` — the latest
  bounded execution record.

Run it from the repository root after building the sibling `mncs-language` CLI:

```bash
python3 tools/run_mnel_native_one_step.py \
  --mncs-bin ../mncs-language/target/debug/mncs \
  --library-path ../mncs-language/library
```

The native source retains rejected-teacher and `UNKNOWN` observations in the dataset,
but excludes them from optimizer updates. Native decisions carry
`Authority::DIAGNOSTIC_ONLY`, explicit `fallback_required`, and zero learned passes on
gate failures. The bounded `u64` fields in this fixture are lineage fingerprints, not
cryptographic identities. Agreement across the five backends is finite execution
evidence; compiler `UNKNOWN` obligations and the lack of a measured wall-clock speed
claim remain explicit.
