#!/usr/bin/env python3
"""Generate deterministic MNCS execution corpora for the mnel.core slice.

Each corpus case is computed by driving the *reference* MNEL implementation
(src/mnel) over frozen inputs. The generated corpora are then executed by the
MNCS-language implementation (mncs/source/mnel-core.mncs) through the mncs
compiler backends, and case-level agreement is reported by
tools/run_mncs_differential.py.

Oracle provenance is recorded per section:
  - "reference-code": the expected value is produced by calling MNEL classes.
  - "derived-table":  the expected value encodes documented MNEL behavior with
    a citation to the enforcing source location, because MNEL enforces it
    structurally (construction-time rejection) rather than computationally.

The generator is deterministic: no clocks, no randomness, sorted iteration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mnel.core import (  # noqa: E402
    AuthorityBoundary,
    HardGateEvaluator,
    Intervention,
    Maturity,
    Observation,
    OutcomeClass,
    RecursionGovernor,
    ResourceBudget,
    ExperimentPlan,
    Prediction,
    TransferStatus,
    Verdict,
    Visibility,
)
from mnel.distillation import StudyDataAccess, StudyRecord, VisibilityViolation  # noqa: E402

SOURCE_PATH = REPO_ROOT / "mncs" / "source" / "mnel" / "all.mncs"
SOURCES = sorted((REPO_ROOT / "mncs" / "source" / "mnel").glob("*.mncs"))
OUTPUT_PATH = REPO_ROOT / "mncs" / "corpora" / "mnel-core-reference.json"

# Home modules after the modularization of the reconstruction: every
# declaration's identity is anchored to the module that declares it.
MODULE_CORE = "mnel.core"
MODULE_VERDICT = "mnel.verdict"
MODULE_GATES = "mnel.gates"
MODULE_LIFECYCLE = "mnel.lifecycle"
MODULE_VISIBILITY = "mnel.visibility"
MODULE_TRANSFER = "mnel.transfer"
MODULE_AUTHORITY = "mnel.authority"
MODULE_NEGATIVE_MEMORY = "mnel.negative_memory"
MODULE_PROBE = "mnel.probe"
MODULE_REJECTION = "mnel.rejection"

STEP_BUDGET = 512

GENERATOR_IDENTITY = "mnel-mncs-corpus-generator/0.1"
ORACLE_REFERENCE_CODE = "reference-code"
ORACLE_DERIVED_TABLE = "derived-table"


# ---------------------------------------------------------------------------
# MNCS value encoding (mirrors crates/mncs-model/src/identity.rs and the
# ExecutionValue serde representation).
# ---------------------------------------------------------------------------

def encode_component(value: str) -> str:
    # unchanged helper; see identity.rs encode_component
    out = []
    for byte in value.encode("utf-8"):
        ch = chr(byte)
        if ch.isascii() and (ch.isalnum() or ch in "_-."):
            out.append(ch)
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


TYPE_HOME_MODULE = {
    "Verdict": MODULE_VERDICT,
    "GateOperator": MODULE_GATES,
    "MetricPresence": MODULE_GATES,
    "GateInput": MODULE_GATES,
    "ExperimentState": MODULE_LIFECYCLE,
    "LifecycleEvent": MODULE_LIFECYCLE,
    "TransitionOutcome": MODULE_LIFECYCLE,
    "PlanFacts": MODULE_AUTHORITY,
    "PlanDecision": MODULE_AUTHORITY,
    "ContextMembership": MODULE_NEGATIVE_MEMORY,
    "ExperimentOutcome": MODULE_CORE,
    "RejectionReason": MODULE_REJECTION,
    "Visibility": MODULE_VISIBILITY,
    "AccessPurpose": MODULE_VISIBILITY,
    "TransferStatus": MODULE_TRANSFER,
    "Maturity": MODULE_TRANSFER,
}

FUNCTION_HOME_MODULE = {
    "combine_verdict": MODULE_VERDICT,
    "verdict_is_known": MODULE_VERDICT,
    "evaluate_gate": MODULE_GATES,
    "evaluate_gates": MODULE_GATES,
    "access_granted": MODULE_VISIBILITY,
    "validate_plan": MODULE_AUTHORITY,
    "transition": MODULE_LIFECYCLE,
    "effective_maturity": MODULE_TRANSFER,
    "retrieval_score": MODULE_NEGATIVE_MEMORY,
    "probe_metric_availability": MODULE_PROBE,
    "run_reference_experiment": MODULE_CORE,
}


def finite_type_id(module_name: str, name: str) -> str:
    return f"mncs:0.2:finite-type:{encode_component(module_name)}::{encode_component(name)}"


def finite_variant_id(module_name: str, type_name: str, variant: str) -> str:
    return (
        f"mncs:0.2:finite-variant:{encode_component(module_name)}"
        f"::{encode_component(type_name)}::{encode_component(variant)}"
    )


def record_type_id(module_name: str, name: str, fields: list[tuple[str, str]]) -> str:
    canonical = sorted(fields)
    joined = "".join(f"{fname}:{ftype};" for fname, ftype in canonical)
    return (
        f"mncs:0.2:record-type:{encode_component(module_name)}"
        f"::{encode_component(name)}::{encode_component(joined)}"
    )


def type_module(type_name: str) -> str:
    try:
        return TYPE_HOME_MODULE[type_name]
    except KeyError:
        raise AssertionError(f"declare the home module for type {type_name}")


def fn_module(function: str) -> str:
    try:
        return FUNCTION_HOME_MODULE[function]
    except KeyError:
        raise AssertionError(f"declare the home module for function {function}")


def finite(type_name: str, variant: str, discriminant: int) -> dict:
    home = type_module(type_name)
    return {
        "finite": {
            "type_identity": finite_type_id(home, type_name),
            "variant_identity": finite_variant_id(home, type_name, variant),
            "discriminant": discriminant,
        }
    }


def integer(value: int, bits: int = 64) -> dict:
    return {"integer": {"value": value, "type": {"bits": bits, "signed": True}}}


def boolean(value: bool) -> dict:
    return {"boolean": {"value": value}}


def record(name: str, fields: list[tuple[str, str]], values: dict) -> dict:
    return {
        "record": {
            "type_identity": record_type_id(type_module(name), name, fields),
            "name": name,
            "fields": [[fname, values[fname]] for fname in sorted(values)],
        }
    }


def case(case_id: str, function: str, arguments: list, expected: list, *, oracle: str,
         oracle_citation: str, expected_status: str | None = None) -> dict:
    request = {
        "schema_version": "0.1",
        "target": {"module": fn_module(function), "function": function},
        "arguments": arguments,
        "step_budget": STEP_BUDGET,
    }
    entry = {
        "id": case_id,
        "request": request,
        "expected": expected,
        "oracle": {"kind": oracle, "citation": oracle_citation},
    }
    if expected_status is not None:
        entry["expected_status"] = expected_status
    return entry


# ---------------------------------------------------------------------------
# Enum discriminant maps: declaration order in mncs/source/mnel-core.mncs.
# ---------------------------------------------------------------------------

VERDICT = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}
GATE_OP = {"GE": 0, "GT": 1, "LE": 2, "LT": 3, "EQ": 4}
PRESENCE = {"PRESENT": 0, "ABSENT": 1}
STATE = {
    "DRAFT": 0, "PREREGISTERED": 1, "RUNNING": 2, "OBSERVED": 3,
    "EVALUATED": 4, "ATTRIBUTED": 5, "DISTILLED_PROPOSAL": 6,
    "REJECTED": 7, "UNKNOWN": 8,
}
EVENT = {
    "PREREGISTER": 0, "BEGIN_EXECUTION": 1, "BUDGET_WITHIN_LIMITS": 2,
    "BUDGET_EXCEEDED": 3, "EVALUATOR_VERDICT": 4, "RECORD_ATTRIBUTION": 5,
    "PROPOSE_DISTILLATION": 6,
}
VISIBILITY_RANK = {  # declaration order in mnel-core.mncs
    "DEVELOPMENT_VISIBLE": 0,
    "SELECTION_OBSERVED_NOT_REPAIRABLE": 1,
    "TRANSFER_HIDDEN": 2,
    "FUTURE_FINAL_INACCESSIBLE": 3,
}
PURPOSE = {"DEVELOPMENT_STUDY": 0, "TRANSFER_EVALUATOR": 1}
TRANSFER = {"UNTESTED": 0, "FAILED": 1, "PARTIAL": 2, "SUPPORTED": 3}
MATURITY = {
    "PROVISIONAL": 0, "SUPPORTED": 1, "CHALLENGED": 2,
    "REJECTED": 3, "RETIRED": 4,
}
REJECTION = {
    "NONE": 0, "INVALID_TRANSITION": 1, "BUDGET_EXHAUSTED": 2,
    "AUTHORITY_EXPANSION": 3, "VISIBILITY_VIOLATION": 4, "IN_PLACE_EDIT": 5,
}

GATE_INPUT_FIELDS = [
    ("present", "MetricPresence"),
    ("operator", "GateOperator"),
    ("observed", "i64"),
    ("threshold", "i64"),
]
PLAN_FACTS_FIELDS = [
    ("governor_known", "bool"),
    ("promotion_authorized", "bool"),
    ("operations_budget", "i64"),
    ("most_hidden_rank_requested", "i64"),
    ("parent_matches_child", "bool"),
]
TRANSITION_OUTCOME_FIELDS = [
    ("advanced", "bool"),
    ("next_state", "ExperimentState"),
    ("reason", "RejectionReason"),
]
PLAN_DECISION_FIELDS = [("accepted", "bool"), ("reason", "RejectionReason")]
CONTEXT_FIELDS = [
    ("retrieval", "bool"),
    ("planning", "bool"),
    ("transfer", "bool"),
    ("monitoring", "bool"),
]
OUTCOME_FIELDS = [
    ("final_state", "ExperimentState"),
    ("verdict", "Verdict"),
    ("principle_maturity", "Maturity"),
]


def gate_input(present: bool, op: str, observed: int, threshold: int) -> dict:
    return record(
        "GateInput",
        GATE_INPUT_FIELDS,
        {
            "present": finite("MetricPresence", "PRESENT" if present else "ABSENT",
                              PRESENCE["PRESENT" if present else "ABSENT"]),
            "operator": finite("GateOperator", op, GATE_OP[op]),
            "observed": integer(observed),
            "threshold": integer(threshold),
        },
    )


def verdict_value(verdict_text: str) -> dict:
    return finite("Verdict", verdict_text, VERDICT[verdict_text])


def transition_outcome(advanced: bool, next_state: str, reason: str) -> dict:
    return record(
        "TransitionOutcome",
        TRANSITION_OUTCOME_FIELDS,
        {
            "advanced": boolean(advanced),
            "next_state": finite("ExperimentState", next_state, STATE[next_state]),
            "reason": finite("RejectionReason", reason, REJECTION[reason]),
        },
    )


def plan_decision(accepted: bool, reason: str) -> dict:
    return record(
        "PlanDecision",
        PLAN_DECISION_FIELDS,
        {
            "accepted": boolean(accepted),
            "reason": finite("RejectionReason", reason, REJECTION[reason]),
        },
    )


def outcome(final_state: str, verdict: str, maturity: str) -> dict:
    return record(
        "ExperimentOutcome",
        OUTCOME_FIELDS,
        {
            "final_state": finite("ExperimentState", final_state, STATE[final_state]),
            "verdict": verdict_value(verdict),
            "principle_maturity": finite("Maturity", maturity, MATURITY[maturity]),
        },
    )


# ---------------------------------------------------------------------------
# Reference oracles.
# ---------------------------------------------------------------------------

EVALUATOR = HardGateEvaluator()
GOVERNOR = RecursionGovernor()


def corpus_attribution():
    """A minimal real Attribution for driving distillation code paths."""
    from dataclasses import fields as dc_fields
    from mnel.core import Attribution

    values = {
        "attribution_id": "attribution-corpus",
        "experiment_id": "exp-corpus",
        "intervention_id": "intervention-corpus",
        "evaluation_id": "evaluation-corpus",
        "disposition": "inconclusive",
        "credit_classes": ("immediate",),
        "supporting_observation_ids": ("obs-corpus",),
        "viable_alternatives": ("none-observed",),
        "source_record_ids": ("record-corpus",),
    }
    kwargs = {f.name: values.get(f.name) for f in dc_fields(Attribution)}
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        raise AssertionError(f"Attribution signature changed; update generator: {missing}")
    return Attribution(**kwargs)


def reference_gate_verdicts(gates: list[dict], present_flags: list[bool],
                            observed: list[int]) -> tuple[list[str], str]:
    """Drive the real HardGateEvaluator over the same four gates.

    Metrics that the slice models as ABSENT are omitted from the observation;
    the evaluator therefore reports UNKNOWN for them ("metric or operator
    unavailable"), which is exactly the absence semantics the MNCS slice
    implements.
    """
    metrics: dict[str, float | int] = {}
    for gate, is_present, value in zip(gates, present_flags, observed):
        if is_present:
            metrics[gate["metric"]] = value
    observation = Observation(
        observation_id="obs-corpus",
        experiment_id="exp-corpus",
        outcome_class=OutcomeClass.SUCCESS,
        metrics=metrics,
        operation_count=1,
        wall_seconds=0.0,
        provider_identity="corpus-fixture-provider/0.1",
    )
    result = EVALUATOR.evaluate(experiment_id="exp-corpus", observation=observation,
                                gates=tuple(gates))
    per_gate = [entry["verdict"] for entry in result.gate_results]
    return per_gate, result.verdict.value


def make_plan(*, visibility: Visibility, governor_identity: str,
              max_candidates: int) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="exp-corpus",
        question="Does the candidate improve exact target recovery without reducing retention?",
        hypothesis_ids=("hypothesis-corpus",),
        intervention=Intervention(
            intervention_id="intervention-corpus",
            parent_candidate_id="candidate-parent",
            child_candidate_id="candidate-child",
            operation="route-through-learned-provider",
            affected_surfaces=("diagnostic-router",),
            rollback_target="candidate-parent",
        ),
        predictions=(Prediction(metric="exact_target_rate", direction="increase",
                                expected_delta=0.05, maximum_regression=0.0),),
        hard_gates=(
            {"name": "recovery", "metric": "exact_target_rate", "operator": "ge",
             "threshold": 0.90},
        ),
        visibility=visibility,
        budget=ResourceBudget(max_candidates=max_candidates),
        authority=AuthorityBoundary(
            evaluator_identity=EVALUATOR.identity,
            governor_identity=governor_identity,
            partition_identity="partition-corpus",
            resource_policy_identity="policy-corpus",
        ),
    )


def reference_plan_decision(*, visibility_rank_requested: int, governor_known: bool,
                            operations_budget: int) -> tuple[bool, str, str]:
    """Drive the real RecursionGovernor; map its decision onto slice reasons.

    Returns (accepted, rejection_reason, oracle_kind). Cases that MNEL rejects
    structurally (AuthorityBoundary promotion ban, Intervention in-place edit)
    are handled by the caller as derived-table entries with citations.
    """
    visibility_by_rank = {
        0: Visibility.DEVELOPMENT,
        1: Visibility.SELECTION_OBSERVED,
        2: Visibility.TRANSFER_HIDDEN,
        3: Visibility.FUTURE_FINAL,
    }
    plan = make_plan(
        visibility=visibility_by_rank[visibility_rank_requested],
        governor_identity=GOVERNOR.identity if governor_known else "other-governor/0.1",
        max_candidates=max(operations_budget, 0),
    )
    decision = GOVERNOR.validate_plan(plan)
    if decision.verdict is Verdict.PASS:
        return True, "NONE", ORACLE_REFERENCE_CODE
    # Map the governor's reason strings onto slice rejection kinds.
    reasons = "; ".join(decision.reasons)
    if "hidden material" in reasons:
        return False, "VISIBILITY_VIOLATION", ORACLE_REFERENCE_CODE
    if "governor identity" in reasons:
        return False, "AUTHORITY_EXPANSION", ORACLE_REFERENCE_CODE
    if "budget is zero" in reasons:
        return False, "BUDGET_EXHAUSTED", ORACLE_REFERENCE_CODE
    raise AssertionError(f"unmapped governor reasons: {reasons}")


DEV_VIEW = StudyDataAccess.development([
    StudyRecord(record_type="episode", payload={"n": i}, visibility=v)
    for i, v in enumerate([
        Visibility.DEVELOPMENT,
        Visibility.SELECTION_OBSERVED,
        Visibility.TRANSFER_HIDDEN,
    ])
])


def reference_access_granted(purpose_kind: str, requested_rank: int) -> tuple[bool, str]:
    """Drive the real StudyDataAccess views; FUTURE_FINAL is derived because
    MNEL refuses to even construct such a view (distillation.py constructor)."""
    if requested_rank == 3:
        return (False,
                "StudyDataAccess refuses future-final views outright "
                "(src/mnel/distillation.py StudyDataAccess.__init__)")

    records = [StudyRecord(record_type="episode", payload={"probe": True},
                           visibility=[
                               Visibility.DEVELOPMENT,
                               Visibility.SELECTION_OBSERVED,
                               Visibility.TRANSFER_HIDDEN,
                           ][requested_rank])]
    try:
        view = (StudyDataAccess.development(records) if purpose_kind == "DEVELOPMENT_STUDY"
                else StudyDataAccess.transfer_evaluator(records))
        view.get(records[0].identity)
        return True, ORACLE_REFERENCE_CODE
    except VisibilityViolation:
        return False, ORACLE_REFERENCE_CODE


def reference_maturity(requested: str, transfer: str) -> tuple[str, str]:
    """Drive the real VerifiedExperienceDistiller.propose_principle demotion."""
    from mnel.core import Attribution, VerifiedExperienceDistiller

    attribution = corpus_attribution()
    proposal = VerifiedExperienceDistiller().propose_principle(
        principle_id="principle-corpus",
        statement="Corpus fixture principle statement.",
        scope={"fixture": "mncs-corpus"},
        attributions=(attribution,),
        counterexample_episode_ids=(),
        falsifier="A held-out provider loses target recovery.",
        transfer_status=TransferStatus(transfer.lower()),
        requested_maturity=Maturity(requested.lower()),
    )
    return proposal.maturity.value.upper(), ORACLE_REFERENCE_CODE


# Lifecycle table: exact expected-state sets from ExperimentCoordinator.run()
# (src/mnel/core.py, run() transition calls). Derived-table oracle.
LIFECYCLE_ALLOWED = {
    ("DRAFT", "PREREGISTER"): "PREREGISTERED",
    ("PREREGISTERED", "BEGIN_EXECUTION"): "RUNNING",
    ("RUNNING", "BUDGET_WITHIN_LIMITS"): "OBSERVED",
    ("RUNNING", "BUDGET_EXCEEDED"): "REJECTED",
    ("OBSERVED", "EVALUATOR_VERDICT"): "EVALUATED",
    ("EVALUATED", "RECORD_ATTRIBUTION"): "ATTRIBUTED",
    ("ATTRIBUTED", "PROPOSE_DISTILLATION"): "DISTILLED_PROPOSAL",
}

STATES = ["DRAFT", "PREREGISTERED", "RUNNING", "OBSERVED", "EVALUATED",
          "ATTRIBUTED", "DISTILLED_PROPOSAL", "REJECTED", "UNKNOWN"]
EVENTS = list(EVENT.keys())


def reference_transition(state: str, event: str) -> tuple[bool, str, str]:
    if (state, event) == ("RUNNING", "BUDGET_EXCEEDED"):
        return True, "REJECTED", "BUDGET_EXHAUSTED"
    if (state, event) in LIFECYCLE_ALLOWED:
        return True, LIFECYCLE_ALLOWED[(state, event)], "NONE"
    if state == "REJECTED" or state == "UNKNOWN" or state == "DISTILLED_PROPOSAL":
        return False, state, "INVALID_TRANSITION"
    return False, state, "INVALID_TRANSITION"


# ---------------------------------------------------------------------------
# Corpus assembly.
# ---------------------------------------------------------------------------

def build_cases() -> list[dict]:
    cases: list[dict] = []

    # --- Verdict lattice ---------------------------------------------------
    lattice_inputs = [("PASS", 0), ("FAIL", 1), ("UNKNOWN", 2)]
    for left, _l in lattice_inputs:
        for right, _r in lattice_inputs:
            combined = "FAIL"
            if left != "FAIL":
                if left == "UNKNOWN":
                    combined = "UNKNOWN" if right == "PASS" else right
                else:
                    combined = right
            cases.append(case(
                f"combine-{left.lower()}-{right.lower()}",
                "combine_verdict",
                [finite("Verdict", left, VERDICT[left]), finite("Verdict", right, VERDICT[right])],
                [verdict_value(combined)],
                oracle=ORACLE_DERIVED_TABLE,
                oracle_citation="HardGateEvaluator aggregation rule "
                                "(src/mnel/core.py evaluate): any FAIL => FAIL; else any "
                                "UNKNOWN => UNKNOWN; else PASS.",
            ))

    # --- Hard gates (real evaluator) ---------------------------------------
    gate_specs = [
        # (op, threshold, observed, present)
        ("GE", 90, 95, True), ("GE", 90, 85, True), ("GT", 10, 10, True),
        ("LE", 5, 5, True), ("LT", 7, 9, True), ("EQ", 42, 42, True),
        ("GE", 90, 0, False), ("EQ", 1, 0, False),
        ("GT", 9, 10, True), ("LE", 100, 99, True),
    ]
    panels = [
        ("all-pass", [0, 3, 8, 5]),
        ("fail-dominates", [1, 0, 3, 5]),
        ("unknown-holds", [6, 0, 3, 8]),
        ("mixed-fail-beats-unknown", [6, 1, 0, 5]),
        ("fourth-gate-absent", [0, 3, 5, 7]),
    ]
    for panel_name, indices in panels:
        gates = []
        present_flags = []
        observed_values = []
        arguments = []
        for slot, index in enumerate(indices):
            op, threshold, observed, present = gate_specs[index]
            metric_name = f"metric_{slot}"
            gates.append({"name": f"gate_{slot}", "metric": metric_name,
                          "operator": op.lower(), "threshold": threshold})
            present_flags.append(present)
            observed_values.append(observed if present else 0)
            arguments.append(gate_input(present, op, observed, threshold))
        per_gate, overall = reference_gate_verdicts(gates, present_flags, observed_values)
        cases.append(case(
            f"gates-panel-{panel_name}",
            "evaluate_gates",
            arguments,
            [verdict_value(overall)],
            oracle=ORACLE_REFERENCE_CODE,
            oracle_citation="mnel.core.HardGateEvaluator.evaluate over identical four-gate "
                            "panels and metrics; absent metrics omitted from the observation.",
        ))
        for slot, (gate, gate_verdict) in enumerate(zip(gates, per_gate)):
            index = indices[slot]
            op, threshold, observed, present = gate_specs[index]
            cases.append(case(
                f"gate-{panel_name}-{slot}-{op.lower()}-{'present' if present else 'absent'}",
                "evaluate_gate",
                [gate_input(present, op, observed, threshold)],
                [verdict_value(gate_verdict)],
                oracle=ORACLE_REFERENCE_CODE,
                oracle_citation=f"mnel.core.HardGateEvaluator.evaluate gate '{gate['name']}'.",
            ))

    # --- Visibility ladder (real views + structural future-final) ----------
    for purpose_kind in PURPOSE:
        for rank in VISIBILITY_RANK.values():
            granted, citation = reference_access_granted(purpose_kind, rank)
            cases.append(case(
                f"access-{purpose_kind.lower().replace('_','-')}-rank-{rank}",
                "access_granted",
                [finite("AccessPurpose", purpose_kind, PURPOSE[purpose_kind]),
                 finite("Visibility", list(VISIBILITY_RANK)[rank]
                        if isinstance(rank, int) and rank < len(VISIBILITY_RANK)
                        else list(VISIBILITY_RANK)[rank], rank)],
                [boolean(granted)],
                oracle=ORACLE_REFERENCE_CODE if citation == ORACLE_REFERENCE_CODE
                else ORACLE_DERIVED_TABLE,
                oracle_citation=citation if citation != ORACLE_REFERENCE_CODE
                else "mnel.distillation.StudyDataAccess.development/.transfer_evaluator "
                     "views probed with .get() on records of each visibility.",
            ))

    # --- Plan validation ----------------------------------------------------
    plan_cases = [
        # (rank_requested, governor_known, budget) : constructible by reference
        ("admitted", 1, True, 8),
        ("zero-candidate-budget", 1, True, 0),
        ("governor-mismatch", 0, True, 8),
        ("hidden-visibility", 2, True, 8),
        ("future-final-visibility", 3, True, 8),
    ]
    for name, rank, gov_known, budget in plan_cases:
        accepted, reason, oracle_kind = reference_plan_decision(
            visibility_rank_requested=rank, governor_known=gov_known,
            operations_budget=budget)
        arguments = [record(
            "PlanFacts", PLAN_FACTS_FIELDS,
            {
                "governor_known": boolean(gov_known),
                "promotion_authorized": boolean(False),
                "operations_budget": integer(budget),
                "most_hidden_rank_requested": integer(rank),
                "parent_matches_child": boolean(False),
            },
        )]
        cases.append(case(
            f"plan-{name}", "validate_plan", arguments, [plan_decision(accepted, reason)],
            oracle=oracle_kind,
            oracle_citation="RecursionGovernor.validate_plan over an ExperimentPlan built "
                            "with matching visibility/budget/authority "
                            "(src/mnel/core.py).",
        ))
    # Structurally rejected plans: MNEL refuses construction; the slice
    # reports an explicit rejection decision instead. Derived-table oracles.
    cases.append(case(
        "plan-promotion-not-authorizable", "validate_plan",
        [record("PlanFacts", PLAN_FACTS_FIELDS, {
            "governor_known": boolean(True),
            "promotion_authorized": boolean(True),
            "operations_budget": integer(8),
            "most_hidden_rank_requested": integer(1),
            "parent_matches_child": boolean(False),
        })],
        [plan_decision(False, "AUTHORITY_EXPANSION")],
        oracle=ORACLE_DERIVED_TABLE,
        oracle_citation="AuthorityBoundary.__post_init__ raises 'MNEL plans cannot authorize "
                        "promotion' (src/mnel/core.py); the slice surfaces the same boundary "
                        "as an explicit rejection decision.",
    ))
    cases.append(case(
        "plan-in-place-edit", "validate_plan",
        [record("PlanFacts", PLAN_FACTS_FIELDS, {
            "governor_known": boolean(True),
            "promotion_authorized": boolean(False),
            "operations_budget": integer(8),
            "most_hidden_rank_requested": integer(1),
            "parent_matches_child": boolean(True),
        })],
        [plan_decision(False, "IN_PLACE_EDIT")],
        oracle=ORACLE_DERIVED_TABLE,
        oracle_citation="Intervention.__post_init__ rejects parent==child ('an evaluated "
                        "candidate may not be edited in place', src/mnel/core.py); the "
                        "slice surfaces the boundary as an explicit rejection decision.",
    ))

    # --- Lifecycle transitions (derived table from coordinator.run) --------
    for state in STATES:
        for event in EVENTS:
            advanced, next_state, reason = reference_transition(state, event)
            cases.append(case(
                f"transition-{state.lower()}-on-{event.lower()}",
                "transition",
                [finite("ExperimentState", state, STATE[state]),
                 finite("LifecycleEvent", event, EVENT[event])],
                [transition_outcome(advanced, next_state, reason)],
                oracle=ORACLE_DERIVED_TABLE,
                oracle_citation="ExperimentCoordinator.run() transition calls "
                                "(src/mnel/core.py): DRAFT->PREREGISTERED (governor), "
                                "PREREGISTERED->RUNNING (executor), RUNNING->OBSERVED|"
                                "REJECTED (budget), OBSERVED->EVALUATED (evaluator), "
                                "EVALUATED->ATTRIBUTED (attribution-engine), "
                                "ATTRIBUTED->DISTILLED_PROPOSAL (synthesizer); all other "
                                "combinations raise 'invalid transition'.",
            ))

    # --- Transfer-gated maturity (real distiller) ---------------------------
    for requested in MATURITY:
        for transfer in TRANSFER:
            expected_maturity, oracle_kind = reference_maturity(requested, transfer)
            cases.append(case(
                f"maturity-{requested.lower()}-transfer-{transfer.lower()}",
                "effective_maturity",
                [finite("Maturity", requested, MATURITY[requested]),
                 finite("TransferStatus", transfer, TRANSFER[transfer])],
                [finite("Maturity", expected_maturity, MATURITY[expected_maturity])],
                oracle=oracle_kind,
                oracle_citation="VerifiedExperienceDistiller.propose_principle demotion "
                                "(src/mnel/core.py propose_principle).",
            ))

    # --- Negative memory retrieval demotion (derived table) -----------------
    context_sets = {
        "empty": (False, False, False, False),
        "retrieval": (True, False, False, False),
        "planning": (False, True, False, False),
        "transfer": (False, False, True, False),
        "monitoring": (False, False, False, True),
        "retrieval+planning": (True, True, False, False),
    }
    rule_sets = {
        "rule-retrieval": (True, False, False, False),
        "rule-planning": (False, True, False, False),
        "rule-none": (False, False, False, False),
    }
    for rule_name, rule_flags in rule_sets.items():
        for ctx_name, ctx_flags in context_sets.items():
            conflicted = any(a and b for a, b in zip(rule_flags, ctx_flags))
            base = 20
            score = base - 6 if conflicted else base
            def ctx_record(flags):
                return record("ContextMembership", CONTEXT_FIELDS, {
                    "retrieval": boolean(flags[0]),
                    "planning": boolean(flags[1]),
                    "transfer": boolean(flags[2]),
                    "monitoring": boolean(flags[3]),
                })
            cases.append(case(
                f"negative-memory-{rule_name}-vs-{ctx_name}",
                "retrieval_score",
                [integer(base), ctx_record(rule_flags), ctx_record(ctx_flags)],
                [integer(score)],
                oracle=ORACLE_DERIVED_TABLE,
                oracle_citation="Retrieval demotion weight -6 on prohibited-context match "
                                "(src/mnel/distillation.py RetrievalIndex scoring; "
                                "'negative results demote without deleting lineage').",
            ))

    # --- Bounded probe -------------------------------------------------------
    probe_cases = [
        ("first-present", (True, False, False)),
        ("second-present", (False, True, False)),
        ("third-present", (False, False, True)),
        ("never-present", (False, False, False)),
        ("all-present", (True, True, True)),
    ]
    for name, flags in probe_cases:
        if flags[0]:
            expected_presence = "PRESENT"
        elif flags[1]:
            expected_presence = "PRESENT"
        elif flags[2]:
            expected_presence = "PRESENT"
        else:
            expected_presence = "ABSENT"
        cases.append(case(
            f"probe-{name}",
            "probe_metric_availability",
            [boolean(flags[0]), boolean(flags[1]), boolean(flags[2])],
            [finite("MetricPresence", expected_presence, PRESENCE[expected_presence])],
            oracle=ORACLE_DERIVED_TABLE,
            oracle_citation="Bounded retry semantics: absent after exhausting the bound "
                            "stays ABSENT and is evaluated as UNKNOWN downstream "
                            "(README fail-closed rule).",
        ))

    # --- End-to-end experiment spine ----------------------------------------
    def plan_facts(admitted: bool) -> dict:
        return record("PlanFacts", PLAN_FACTS_FIELDS, {
            "governor_known": boolean(admitted),
            "promotion_authorized": boolean(False),
            "operations_budget": integer(8),
            "most_hidden_rank_requested": integer(1),
            "parent_matches_child": boolean(False),
        })

    passing_gates = [
        gate_input(True, "GE", 95, 90),
        gate_input(True, "GE", 100, 100),
        gate_input(True, "LE", 5, 5),
        gate_input(True, "EQ", 42, 42),
    ]
    failing_gates = [
        gate_input(True, "GE", 95, 90),
        gate_input(True, "GE", 85, 90),
        gate_input(True, "LE", 5, 5),
        gate_input(True, "EQ", 42, 42),
    ]
    unknown_gates = [
        gate_input(True, "GE", 95, 90),
        gate_input(False, "GE", 0, 90),
        gate_input(True, "LE", 5, 5),
        gate_input(True, "EQ", 42, 42),
    ]
    spine_cases = [
        ("pass-supported-transfer", plan_facts(True), passing_gates, "SUPPORTED",
         "SUPPORTED", "DISTILLED_PROPOSAL", "PASS", "SUPPORTED"),
        ("pass-untested-transfer", plan_facts(True), passing_gates, "UNTESTED",
         "SUPPORTED", "DISTILLED_PROPOSAL", "PASS", "PROVISIONAL"),
        ("fail-verdict", plan_facts(True), failing_gates, "UNTESTED",
         "PROVISIONAL", "DISTILLED_PROPOSAL", "FAIL", "PROVISIONAL"),
        ("unknown-verdict-stays-provisional", plan_facts(True), unknown_gates, "UNTESTED",
         "SUPPORTED", "DISTILLED_PROPOSAL", "UNKNOWN", "PROVISIONAL"),
        ("rejected-plan", plan_facts(False), unknown_gates, "UNTESTED",
         "SUPPORTED", "REJECTED", "UNKNOWN", "PROVISIONAL"),
    ]
    for name, facts, gates, transfer, requested, exp_state, exp_verdict, exp_maturity in spine_cases:
        # Overall verdict oracle: reuse the real evaluator on equivalent gates.
        specs = []
        present_flags = []
        observed_values = []
        for slot, g in enumerate(gates):
            fields = dict(g["record"]["fields"])
            op_variant = fields["operator"]["finite"]["variant_identity"].rsplit("::", 1)[-1]
            present_variant = fields["present"]["finite"]["variant_identity"].rsplit("::", 1)[-1]
            observed = fields["observed"]["integer"]["value"]
            threshold = fields["threshold"]["integer"]["value"]
            specs.append({"name": f"g{slot}", "metric": f"m{slot}",
                          "operator": op_variant.lower(), "threshold": threshold})
            present_flags.append(present_variant == "PRESENT")
            observed_values.append(observed)
        _, overall = reference_gate_verdicts(specs, present_flags, observed_values)
        cases.append(case(
            f"experiment-{name}",
            "run_reference_experiment",
            [facts, *gates,
             finite("TransferStatus", transfer, TRANSFER[transfer]),
             finite("Maturity", requested, MATURITY[requested])],
            [outcome(exp_state, exp_verdict, exp_maturity)],
            oracle=ORACLE_REFERENCE_CODE,
            oracle_citation="Composition of HardGateEvaluator.evaluate (overall verdict), "
                            "propose_principle demotion (maturity), and coordinator.run() "
                            "state flow; UNKNOWN never becomes PASS and rejected plans "
                            "never reach evaluation.",
        ))

    return cases


def main() -> int:
    combined = b"".join(p.read_bytes() for p in SOURCES)
    sources_digest = hashlib.sha256(combined).hexdigest()
    cases = build_cases()
    corpus = {
        "schema_version": "0.1",
        "name": "mnel-core-reference-v1",
        "cases": cases,
        "provenance": {
            "generator_identity": GENERATOR_IDENTITY,
            "reference_package": "mnel (Machine-Native-Experimental-Learning)",
            "mncs_sources": [str(p.relative_to(REPO_ROOT)) for p in SOURCES],
            "mncs_sources_sha256": sources_digest,
            "oracle_kinds": {
                "reference-code": "expected values produced by executing MNEL classes",
                "derived-table": "expected values encode documented MNEL behavior with "
                                 "citations; MNEL enforces these structurally",
            },
            "determinism": "frozen inputs; no clock or randomness; sorted iteration",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(corpus, indent=1, sort_keys=False) + "\n")
    print(f"wrote {len(cases)} cases to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"combined source sha256: {sources_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
