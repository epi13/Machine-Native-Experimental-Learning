# Contributing

MNEL is an evidence-governed research project. Contributions should preserve the
separation between proposal, execution, evaluation, attribution, and promotion.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

## Required properties

Changes must not:

- allow investigator output to become an evaluator verdict;
- permit mutation of evaluator identity, gates, partitions, or resource policy from an
  experiment proposal;
- remove rejected, failed, neutral, abstaining, or `UNKNOWN` records;
- expose future-final material to candidate generation;
- describe local replication as independent evaluation or protected custody; or
- convert repository-local success into formal MNCS/MNCDS status.

New record types should have canonical identity, schema documentation, negative tests,
and explicit source lineage. New integrations should remain provider-neutral and must
not grant external tools more authority than their declared role.
