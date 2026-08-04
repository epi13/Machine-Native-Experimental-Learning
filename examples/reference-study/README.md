# Reference study

Run the deterministic lifecycle from the repository root:

```bash
python -m mnel demo --workspace build/reference-study
python -m mnel ledger verify build/reference-study/evidence.jsonl
```

The example intentionally demonstrates process structure rather than meaningful RAVEL
performance. It produces one passing hard-gate evaluation and one provisional,
transfer-untested principle proposal.
