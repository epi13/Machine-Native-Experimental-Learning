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

MNEL now includes a small local reference surface in `mnel.forge_lifecycle`. It is used
for deterministic tests and the `mnel forge-reference` command: it provides explicit
verifier declarations, bounded snapshot views, preconditions, witnesses, registered
mutations, independent comparison, health, and coverage. It is an adapter/test surface,
not a substitute Forge implementation and does not claim MNCS/MNCDS conformance.

MNEL also exposes a narrow external Forge Provider Protocol 0.1 adapter in
`mnel.forge_provider`. Its capabilities, bounded analysis requests, and one-line responses
are diagnostic-only; malformed, hidden-partition, and authority-expanding requests fail
closed. The project-scoped `mncs-forge.toml` declares the adapter without making Forge a
mandatory MNEL installation dependency. The current sibling Forge checkout can validate,
inspect, probe, and run the declared workflows when invoked with its source on `PYTHONPATH`.

The family integration reference path records a pinned compatibility snapshot and reports
whether live sibling checkouts are available. A compatibility snapshot is not a conformance
claim and does not silently accept protocol drift.

The 0.4 distillation study consumes the same identified diagnostic plane but remains an
MNEL-side research harness. Its groups, strategies, retrieval results, learned-provider
observations, and transfer records are append-only measurements. It does not turn a
retrieval hit, calibrated score, witness, or synthetic transfer result into Forge/MNCS
authority. A future external Forge adapter must supply its own identified evaluator and
keep hidden-transfer access outside development study code.

## MNCS Fabric

Fabric distributes identified experiment bundles, captures node capabilities, and
reconciles observations. MNEL owns experiment semantics; Fabric owns bounded execution
records. A Fabric `PASS` does not become an MNEL causal claim or formal MNCS result.

`mnel family-integration-reference` uses only Fabric's public `FabricService`: it binds a
provider-study identity and provider-artifact reference into a content-addressed local
manifest, executes a bounded local job, consumes the typed experimental receipt, and
normalizes the observation into the MNEL ledger. Repeated execution is labelled
`local-in-process-replication`; it is not multi-host independence, authenticated worker
enrollment, or protected custody.

`mnel fabric-reference` uses the current public `LocalController` and `LocalWorker`
interfaces for a two-logical-worker, network-free study. `NetworkFabricBackend` uses the
documented `NetworkController`/`TLSNetworkTransport` pair only after validating explicit
CA, client-certificate, key, trust-store, capability, and pre-staged bundle references.
Bulk artifact transfer remains outside MNEL until Fabric exposes a public verified
transfer profile. A worker is an execution location, not an expert identity; provider
artifact and model identities remain bound in every workload and observation.

The same path emits an inert Commons Observation-shaped record and a RAVEL 0.6 proposal
context fixture. Neither is published or granted trust-domain, evaluator, freeze, selection,
or promotion authority.

## RAVEL

MNEL may submit a principle, strategy, expert, routing-policy, replay-policy, or other
knowledge candidate to RAVEL. The proposal must bind parent identity, supporting record
lineage, scope, predicted effects, rollback target, and its own content identity. RAVEL
or an external candidate lifecycle evaluates it transactionally.

## MNCDS and MNCS

MNCDS records development evidence, feedback eligibility, lineage, and same-epoch repair
restrictions. MNCS evaluates a frozen implementation claim. MNEL should export eligible
records to those protocols rather than inventing a second conformance authority.
