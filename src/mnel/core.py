"""Core records, ledger, governor, evaluator, distillation, and reference workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Visibility(StrEnum):
    DEVELOPMENT = "development-visible"
    SELECTION_OBSERVED = "selection-observed-not-repairable"
    TRANSFER_HIDDEN = "transfer-hidden"
    FUTURE_FINAL = "future-final-inaccessible"


class OutcomeClass(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    NEUTRAL = "neutral"
    ABSTENTION = "abstention"


class ExperimentState(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    RUNNING = "running"
    OBSERVED = "observed"
    EVALUATED = "evaluated"
    ATTRIBUTED = "attributed"
    DISTILLED = "distilled-proposal"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class TransferStatus(StrEnum):
    UNTESTED = "untested"
    FAILED = "failed"
    PARTIAL = "partial"
    SUPPORTED = "supported"


class Maturity(StrEnum):
    PROVISIONAL = "provisional"
    SUPPORTED = "supported"
    CHALLENGED = "challenged"
    REJECTED = "rejected"
    RETIRED = "retired"


@dataclass(frozen=True)
class ResourceBudget:
    max_operations: int = 100
    max_wall_seconds: int = 3600
    max_candidates: int = 8
    max_failures: int = 32

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in asdict(self).values()):
            raise ValueError("resource budgets must be non-negative integers")


@dataclass(frozen=True)
class AuthorityBoundary:
    evaluator_identity: str
    governor_identity: str
    partition_identity: str
    resource_policy_identity: str
    promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.evaluator_identity,
                self.governor_identity,
                self.partition_identity,
                self.resource_policy_identity,
            )
        ):
            raise ValueError("all authority identities are required")
        if self.promotion_authorized:
            raise ValueError("MNEL plans cannot authorize promotion")


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    supporting_episode_ids: tuple[str, ...]
    competing_hypothesis_ids: tuple[str, ...]
    falsifier: str
    created_before_intervention: bool = True

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.statement or not self.falsifier:
            raise ValueError("hypothesis id, statement, and falsifier are required")
        if not self.created_before_intervention:
            raise ValueError("hypotheses must predate intervention results")


@dataclass(frozen=True)
class Prediction:
    metric: str
    direction: str
    expected_delta: float | None = None
    maximum_regression: float | None = None


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    parent_candidate_id: str
    child_candidate_id: str
    operation: str
    affected_surfaces: tuple[str, ...]
    rollback_target: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.intervention_id,
                self.parent_candidate_id,
                self.child_candidate_id,
                self.operation,
                self.rollback_target,
            )
        ):
            raise ValueError("intervention identities, operation, and rollback are required")
        if self.parent_candidate_id == self.child_candidate_id:
            raise ValueError("an evaluated candidate may not be edited in place")


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    question: str
    hypothesis_ids: tuple[str, ...]
    intervention: Intervention
    predictions: tuple[Prediction, ...]
    hard_gates: tuple[dict[str, Any], ...]
    visibility: Visibility
    budget: ResourceBudget
    authority: AuthorityBoundary
    source_identities: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.question:
            raise ValueError("experiment id and question are required")
        if not self.hypothesis_ids or not self.predictions or not self.hard_gates:
            raise ValueError("hypotheses, predictions, and hard gates are required")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["visibility"] = self.visibility.value
        return value


@dataclass(frozen=True)
class Observation:
    observation_id: str
    experiment_id: str
    outcome_class: OutcomeClass
    metrics: dict[str, float | int | bool]
    operation_count: int
    wall_seconds: float
    provider_identity: str
    artifact_identities: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id or not self.experiment_id or not self.provider_identity:
            raise ValueError("observation, experiment, and provider identities are required")
        if self.operation_count < 0 or self.wall_seconds < 0:
            raise ValueError("resource observations cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outcome_class"] = self.outcome_class.value
        return value


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    experiment_id: str
    evaluator_identity: str
    verdict: Verdict
    gate_results: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["verdict"] = self.verdict.value
        return value


@dataclass(frozen=True)
class Attribution:
    attribution_id: str
    experiment_id: str
    intervention_id: str
    evaluation_id: str
    disposition: str
    credit_classes: tuple[str, ...]
    supporting_observation_ids: tuple[str, ...]
    viable_alternatives: tuple[str, ...]
    source_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.credit_classes or not self.source_record_ids:
            raise ValueError("attribution requires credit classes and source lineage")


@dataclass(frozen=True)
class PrincipleProposal:
    principle_id: str
    statement: str
    scope: dict[str, Any]
    supporting_attribution_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    counterexample_episode_ids: tuple[str, ...]
    falsifier: str
    transfer_status: TransferStatus
    maturity: Maturity

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transfer_status"] = self.transfer_status.value
        value["maturity"] = self.maturity.value
        return value


@dataclass(frozen=True)
class StrategyProposal:
    strategy_id: str
    trigger_conditions: tuple[str, ...]
    intervention_class: str
    preconditions: tuple[str, ...]
    known_failure_modes: tuple[str, ...]
    applicability_scope: dict[str, Any]
    supporting_principle_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    transfer_status: TransferStatus


@dataclass(frozen=True)
class NegativeMemoryRecord:
    memory_id: str
    memory_type: str
    statement: str
    source_record_ids: tuple[str, ...]
    prohibited_contexts: tuple[str, ...]
    reconsideration_condition: str

    def __post_init__(self) -> None:
        if not self.memory_id or not self.memory_type or not self.statement:
            raise ValueError("negative-memory identity, type, and statement are required")
        if not self.source_record_ids or not self.reconsideration_condition:
            raise ValueError("negative memory requires source lineage and reconsideration condition")


class LedgerIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    record_count: int
    head_digest: str | None
    errors: tuple[str, ...] = ()


class EvidenceLedger:
    GENESIS = "GENESIS"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def records(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return ()
        values = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise LedgerIntegrityError(f"blank ledger line at {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerIntegrityError(f"invalid JSON at line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise LedgerIntegrityError(f"ledger line {line_number} is not an object")
                values.append(value)
        return tuple(values)

    def append(self, record_type: str, payload: dict[str, Any], *, actor: str) -> dict[str, Any]:
        verification = self.verify()
        if not verification.valid:
            raise LedgerIntegrityError("refusing to append to an invalid ledger")
        body = {
            "schema": "mnel-ledger-record/0.1",
            "sequence": verification.record_count + 1,
            "timestamp": utc_now(),
            "record_type": record_type,
            "actor": actor,
            "previous_digest": verification.head_digest or self.GENESIS,
            "payload": payload,
        }
        envelope = {**body, "record_digest": canonical_digest(body)}
        self.initialize()
        with self.path.open("ab", buffering=0) as handle:
            handle.write(canonical_json(envelope) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return envelope

    def verify(self) -> LedgerVerification:
        if not self.path.exists():
            return LedgerVerification(True, 0, None)
        previous = self.GENESIS
        errors = []
        count = 0
        head = None
        try:
            records = self.records()
        except LedgerIntegrityError as exc:
            return LedgerVerification(False, 0, None, (str(exc),))
        for expected_sequence, envelope in enumerate(records, 1):
            count += 1
            if envelope.get("sequence") != expected_sequence:
                errors.append(f"record {expected_sequence}: sequence mismatch")
            if envelope.get("previous_digest") != previous:
                errors.append(f"record {expected_sequence}: previous digest mismatch")
            claimed = envelope.get("record_digest")
            actual = canonical_digest(
                {key: value for key, value in envelope.items() if key != "record_digest"}
            )
            if claimed != actual:
                errors.append(f"record {expected_sequence}: record digest mismatch")
            previous = claimed if isinstance(claimed, str) else actual
            head = previous
        return LedgerVerification(not errors, count, head, tuple(errors))

    def summarize(self) -> dict[str, Any]:
        verification = self.verify()
        counts: dict[str, int] = {}
        experiments: set[str] = set()
        if verification.valid:
            for record in self.records():
                kind = str(record.get("record_type", "unknown"))
                counts[kind] = counts.get(kind, 0) + 1
                payload = record.get("payload")
                if isinstance(payload, dict) and isinstance(payload.get("experiment_id"), str):
                    experiments.add(payload["experiment_id"])
        return {
            "valid": verification.valid,
            "record_count": verification.record_count,
            "head_digest": verification.head_digest,
            "record_types": dict(sorted(counts.items())),
            "experiment_ids": sorted(experiments),
            "errors": list(verification.errors),
        }


@dataclass(frozen=True)
class GovernorDecision:
    verdict: Verdict
    reasons: tuple[str, ...]


class RecursionGovernor:
    FORBIDDEN_MUTATION_KEYS = frozenset(
        {
            "evaluator_identity",
            "governor_identity",
            "partition_identity",
            "resource_policy_identity",
            "hard_gates",
            "thresholds",
            "promotion_authorized",
            "future_final",
        }
    )

    def __init__(self, identity: str = "mnel-recursion-governor/0.1") -> None:
        self.identity = identity

    def validate_plan(self, plan: ExperimentPlan) -> GovernorDecision:
        reasons = []
        if plan.visibility in {Visibility.TRANSFER_HIDDEN, Visibility.FUTURE_FINAL}:
            reasons.append("candidate generation may not consume hidden material")
        if plan.authority.governor_identity != self.identity:
            reasons.append("governor identity mismatch")
        if plan.budget.max_candidates == 0:
            reasons.append("candidate budget is zero")
        return GovernorDecision(Verdict.FAIL if reasons else Verdict.PASS, tuple(reasons))

    def validate_proposed_mutation(self, mutation: dict[str, Any]) -> GovernorDecision:
        touched = sorted(self.FORBIDDEN_MUTATION_KEYS.intersection(mutation))
        if touched:
            return GovernorDecision(
                Verdict.FAIL,
                ("forbidden authority mutation: " + ", ".join(touched),),
            )
        return GovernorDecision(Verdict.PASS, ())

    def check_budget(self, plan: ExperimentPlan, observation: Observation) -> GovernorDecision:
        reasons = []
        if observation.operation_count > plan.budget.max_operations:
            reasons.append("operation budget exceeded")
        if observation.wall_seconds > plan.budget.max_wall_seconds:
            reasons.append("wall-time budget exceeded")
        return GovernorDecision(Verdict.FAIL if reasons else Verdict.PASS, tuple(reasons))


class HardGateEvaluator:
    OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
        "ge": lambda observed, threshold: observed >= threshold,
        "gt": lambda observed, threshold: observed > threshold,
        "le": lambda observed, threshold: observed <= threshold,
        "lt": lambda observed, threshold: observed < threshold,
        "eq": lambda observed, threshold: observed == threshold,
        "true": lambda observed, threshold: observed is True,
        "false": lambda observed, threshold: observed is False,
    }

    def __init__(self, identity: str = "mnel-hard-gate-evaluator/0.1") -> None:
        self.identity = identity

    def evaluate(
        self,
        *,
        experiment_id: str,
        observation: Observation,
        gates: tuple[dict[str, Any], ...],
    ) -> EvaluationResult:
        results = []
        for index, gate in enumerate(gates, 1):
            name = str(gate.get("name", f"gate-{index}"))
            metric = str(gate.get("metric", ""))
            operator = str(gate.get("operator", ""))
            threshold = gate.get("threshold")
            observed = observation.metrics.get(metric)
            if metric not in observation.metrics or operator not in self.OPERATORS:
                verdict, reason = Verdict.UNKNOWN, "metric or operator unavailable"
            else:
                try:
                    passed = self.OPERATORS[operator](observed, threshold)
                    verdict = Verdict.PASS if passed else Verdict.FAIL
                    reason = "hard gate satisfied" if passed else "hard gate failed"
                except (TypeError, ValueError):
                    verdict, reason = Verdict.UNKNOWN, "metric and threshold are not comparable"
            results.append(
                {
                    "name": name,
                    "metric": metric,
                    "verdict": verdict.value,
                    "observed": observed,
                    "operator": operator,
                    "threshold": threshold,
                    "reason": reason,
                }
            )
        verdicts = {result["verdict"] for result in results}
        if Verdict.FAIL.value in verdicts:
            verdict = Verdict.FAIL
        elif Verdict.UNKNOWN.value in verdicts or not results:
            verdict = Verdict.UNKNOWN
        else:
            verdict = Verdict.PASS
        payload = {
            "experiment_id": experiment_id,
            "observation_id": observation.observation_id,
            "evaluator_identity": self.identity,
            "gate_results": results,
        }
        return EvaluationResult(
            "evaluation-" + canonical_digest(payload).split(":", 1)[1][:20],
            experiment_id,
            self.identity,
            verdict,
            tuple(results),
            tuple(r["reason"] for r in results if r["verdict"] != Verdict.PASS.value),
        )


class DistillationError(ValueError):
    pass


class VerifiedExperienceDistiller:
    """Create compact proposals without accepting or promoting them."""

    def propose_principle(
        self,
        *,
        principle_id: str,
        statement: str,
        scope: dict[str, object],
        attributions: tuple[Attribution, ...],
        counterexample_episode_ids: tuple[str, ...],
        falsifier: str,
        transfer_status: TransferStatus = TransferStatus.UNTESTED,
        requested_maturity: Maturity = Maturity.PROVISIONAL,
    ) -> PrincipleProposal:
        if not attributions or not statement or not scope or not falsifier:
            raise DistillationError("principles require attributions, statement, scope, and falsifier")
        source_ids = tuple(
            dict.fromkeys(source for item in attributions for source in item.source_record_ids)
        )
        if not source_ids:
            raise DistillationError("distillation may not sever source lineage")
        maturity = requested_maturity
        if transfer_status is not TransferStatus.SUPPORTED and maturity is Maturity.SUPPORTED:
            maturity = Maturity.PROVISIONAL
        return PrincipleProposal(
            principle_id,
            statement,
            scope,
            tuple(item.attribution_id for item in attributions),
            source_ids,
            counterexample_episode_ids,
            falsifier,
            transfer_status,
            maturity,
        )

    def propose_strategy(
        self,
        *,
        strategy_id: str,
        trigger_conditions: tuple[str, ...],
        intervention_class: str,
        preconditions: tuple[str, ...],
        known_failure_modes: tuple[str, ...],
        applicability_scope: dict[str, object],
        principles: tuple[PrincipleProposal, ...],
        transfer_status: TransferStatus,
    ) -> StrategyProposal:
        if not principles or not known_failure_modes:
            raise DistillationError("strategies require principles and known failure modes")
        if transfer_status is TransferStatus.SUPPORTED and any(
            item.transfer_status is not TransferStatus.SUPPORTED for item in principles
        ):
            raise DistillationError("strategy transfer cannot outrank supporting principles")
        return StrategyProposal(
            strategy_id,
            trigger_conditions,
            intervention_class,
            preconditions,
            known_failure_modes,
            applicability_scope,
            tuple(item.principle_id for item in principles),
            tuple(dict.fromkeys(source for item in principles for source in item.source_record_ids)),
            transfer_status,
        )


class ExperimentCoordinator:
    def __init__(
        self,
        ledger: EvidenceLedger,
        governor: RecursionGovernor,
        evaluator: HardGateEvaluator,
    ) -> None:
        self.ledger = ledger
        self.governor = governor
        self.evaluator = evaluator
        self.distiller = VerifiedExperienceDistiller()
        self.states: dict[str, ExperimentState] = {}

    def transition(
        self,
        experiment_id: str,
        expected: tuple[ExperimentState, ...],
        new_state: ExperimentState,
        *,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.states.get(experiment_id, ExperimentState.DRAFT)
        if current not in expected:
            raise ValueError(f"invalid transition from {current.value} to {new_state.value}")
        self.states[experiment_id] = new_state
        return self.ledger.append(
            "experiment-state",
            {
                "experiment_id": experiment_id,
                "previous_state": current.value,
                "state": new_state.value,
                "details": details or {},
            },
            actor=actor,
        )

    def run(
        self,
        plan: ExperimentPlan,
        hypothesis: Hypothesis,
        observation: Observation,
    ) -> dict[str, object]:
        decision = self.governor.validate_plan(plan)
        if decision.verdict is not Verdict.PASS:
            raise ValueError("plan rejected: " + "; ".join(decision.reasons))
        self.ledger.append("causal-hypothesis", asdict(hypothesis), actor="investigator")
        self.ledger.append("experiment-plan", plan.to_dict(), actor="investigator")
        self.transition(
            plan.experiment_id,
            (ExperimentState.DRAFT,),
            ExperimentState.PREREGISTERED,
            actor=self.governor.identity,
        )
        self.transition(
            plan.experiment_id,
            (ExperimentState.PREREGISTERED,),
            ExperimentState.RUNNING,
            actor="executor",
        )
        budget = self.governor.check_budget(plan, observation)
        episode = self.ledger.append("experience-episode", observation.to_dict(), actor="executor")
        if budget.verdict is not Verdict.PASS:
            self.transition(
                plan.experiment_id,
                (ExperimentState.RUNNING,),
                ExperimentState.REJECTED,
                actor=self.governor.identity,
                details={"reasons": list(budget.reasons)},
            )
            raise ValueError("observation exceeded its preregistered budget")
        self.transition(
            plan.experiment_id,
            (ExperimentState.RUNNING,),
            ExperimentState.OBSERVED,
            actor=self.governor.identity,
        )
        if plan.authority.evaluator_identity != self.evaluator.identity:
            raise ValueError("evaluator identity mismatch")
        evaluation = self.evaluator.evaluate(
            experiment_id=plan.experiment_id,
            observation=observation,
            gates=plan.hard_gates,
        )
        evaluation_record = self.ledger.append(
            "evaluation-result",
            evaluation.to_dict(),
            actor=self.evaluator.identity,
        )
        self.transition(
            plan.experiment_id,
            (ExperimentState.OBSERVED,),
            ExperimentState.EVALUATED,
            actor=self.evaluator.identity,
            details={"verdict": evaluation.verdict.value},
        )
        attribution = Attribution(
            f"attribution-{plan.experiment_id}",
            plan.experiment_id,
            plan.intervention.intervention_id,
            evaluation.evaluation_id,
            "supported-with-alternatives" if evaluation.verdict is Verdict.PASS else "inconclusive",
            ("immediate", "retention"),
            (observation.observation_id,),
            ("fixture-specific route advantage",),
            (episode["record_digest"], evaluation_record["record_digest"]),
        )
        self.ledger.append("causal-attribution", asdict(attribution), actor="attribution-engine")
        self.transition(
            plan.experiment_id,
            (ExperimentState.EVALUATED,),
            ExperimentState.ATTRIBUTED,
            actor="attribution-engine",
        )
        principle = self.distiller.propose_principle(
            principle_id=f"principle-{plan.experiment_id}",
            statement=(
                "In the declared reference fixture, retaining supported transition candidates "
                "can improve exact target recovery without reducing measured retention."
            ),
            scope={"fixture": "reference-study/0.1", "mechanism": "transition-aware-routing"},
            attributions=(attribution,),
            counterexample_episode_ids=(),
            falsifier="A held-out provider loses target recovery or retention.",
        )
        self.ledger.append(
            "learned-principle-proposal",
            principle.to_dict(),
            actor="synthesizer",
        )
        self.transition(
            plan.experiment_id,
            (ExperimentState.ATTRIBUTED,),
            ExperimentState.DISTILLED,
            actor="synthesizer",
        )
        return {
            "experiment_id": plan.experiment_id,
            "evaluation_verdict": evaluation.verdict.value,
            "principle_id": principle.principle_id,
            "principle_maturity": principle.maturity.value,
            "transfer_status": principle.transfer_status.value,
            "ledger": self.ledger.summarize(),
        }


def run_reference_study(workspace: str | Path) -> dict[str, object]:
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    ledger = EvidenceLedger(root / "evidence.jsonl")
    if ledger.path.exists():
        ledger.path.unlink()
    ledger.initialize()
    governor = RecursionGovernor()
    evaluator = HardGateEvaluator()
    plan = ExperimentPlan(
        "reference-routing-probe-001",
        "Does transition-aware routing improve exact target recovery without regression?",
        ("hypothesis-transition-routing-001",),
        Intervention(
            "intervention-transition-routing-001",
            "ravel-reference-parent",
            "ravel-reference-child-001",
            "enable bounded transition-aware route candidate",
            ("routing", "transition-memory"),
            "ravel-reference-parent",
        ),
        (
            Prediction("exact_target_rate", "increase", expected_delta=0.10),
            Prediction("retention_rate", "invariant", maximum_regression=0.0),
        ),
        (
            {"name": "exact-target", "metric": "exact_target_rate", "operator": "ge", "threshold": 0.90},
            {"name": "retention", "metric": "retention_rate", "operator": "ge", "threshold": 1.0},
            {"name": "identity", "metric": "identity_match", "operator": "true", "threshold": True},
        ),
        Visibility.DEVELOPMENT,
        ResourceBudget(max_operations=100, max_wall_seconds=60, max_candidates=1),
        AuthorityBoundary(
            evaluator.identity,
            governor.identity,
            "reference-development-partition/0.1",
            "reference-budget/0.1",
        ),
        {"fixture": "reference-study/0.1"},
    )
    hypothesis = Hypothesis(
        "hypothesis-transition-routing-001",
        (
            "Retaining a second supported transition candidate improves exact target "
            "recovery without reducing retention."
        ),
        (),
        ("hypothesis-fixture-artifact-001",),
        "The effect fails a gate or disappears under a held-out provider.",
    )
    observation = Observation(
        "observation-transition-routing-001",
        plan.experiment_id,
        OutcomeClass.SUCCESS,
        {"exact_target_rate": 0.95, "retention_rate": 1.0, "identity_match": True},
        24,
        0.1,
        "deterministic-reference-provider/0.1",
        {"raw": "sha256:reference-observation"},
    )
    return ExperimentCoordinator(ledger, governor, evaluator).run(plan, hypothesis, observation)
