# Security policy

## Current boundary

MNEL treats all investigator and model output as untrusted proposals. The current
standard-library adapters execute configured programs with argument arrays and
`shell=False`, but this repository is not a hardened sandbox.

Do not run unattended experiments against untrusted repositories or programs without a
separate constrained runner.

## Report a vulnerability

Open a private GitHub security advisory for vulnerabilities that could allow authority
escalation, evidence forgery, ledger bypass, path escape, command-policy bypass,
partition leakage, or silent mutation of evaluator and governor identities.

## High-priority threat classes

- evaluator, threshold, partition, or budget mutation;
- future-final or selection leakage into same-candidate repair;
- forged or reordered ledger records;
- source substitution and stale result replay;
- symlink or workspace escape in an integration;
- shell injection or uncontrolled command execution;
- automatic acceptance of a model-generated principle;
- deletion or concealment of failed and rejected experience.
