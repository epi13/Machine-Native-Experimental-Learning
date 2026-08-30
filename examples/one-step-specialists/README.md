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
