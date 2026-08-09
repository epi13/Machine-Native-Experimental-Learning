# Project-family integrations

## epi13-local-harness

The local harness is the intended model-routing and bounded-tool substrate. MNEL's
`LocalHarnessAdapter` sends the sibling JSON-line `chat/start` protocol with a role
contract, eligible-context identity and record identities, runtime identity envelope, task
prompt, allowed tools, and a detached proposal workspace. The command is an explicit
argument vector (`shell=False`), and the adapter enforces a timeout and output ceiling.

Responses must contain the expected protocol/method/request identity and bounded route,
attempt, and model-output fields. Verdict, conformance, promotion, evaluator, hidden
transfer, and future-final fields are rejected. A harness `successful` flag is retained
only as a diagnostic execution observation; MNEL emits no evaluator verdict. Malformed,
failed, or timed-out runs become quarantined or `UNKNOWN` observations and retain their
worktree until explicit cleanup.

`run_local_investigator` composes context packing, source commit identification, detached
Git worktree materialization, request execution, and append-only observation records. The
authoritative checkout is never used as the proposal mutation workspace. Worktree roots
must be configured outside the source checkout, and cleanup is an explicit operator call.

## MNCS Forge

Forge provides diagnostic and counterfactual probes. MNEL binds the exact question,
subject identities, expected witness type, resource budget, mutation prohibitions, and
provider identity. Large prose scans should be decomposed into bounded witnesses where
possible.

## MNCS Fabric

Fabric distributes identified experiment bundles, captures node capabilities, and
reconciles observations. MNEL owns experiment semantics; Fabric owns bounded execution
records. A Fabric `PASS` does not become an MNEL causal claim or formal MNCS result.

## RAVEL

MNEL may submit a principle, strategy, expert, routing-policy, replay-policy, or other
knowledge candidate to RAVEL. The proposal must bind parent identity, supporting record
lineage, scope, predicted effects, rollback target, and its own content identity. RAVEL
or an external candidate lifecycle evaluates it transactionally.

## MNCDS and MNCS

MNCDS records development evidence, feedback eligibility, lineage, and same-epoch repair
restrictions. MNCS evaluates a frozen implementation claim. MNEL should export eligible
records to those protocols rather than inventing a second conformance authority.
