"""Investigator role contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class InvestigatorRole(StrEnum):
    INVESTIGATOR = "investigator"
    SKEPTIC = "skeptic"
    REPLICATOR = "replicator"
    SYNTHESIZER = "synthesizer"
    AUDITOR = "auditor"


@dataclass(frozen=True)
class RoleContract:
    role: InvestigatorRole
    purpose: str
    may: tuple[str, ...]
    may_not: tuple[str, ...]
    required_outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


DEFAULT_ROLE_CONTRACTS = (
    RoleContract(
        InvestigatorRole.INVESTIGATOR,
        "Propose falsifiable hypotheses and bounded interventions.",
        ("read eligible experience", "propose hypotheses", "request probes", "propose candidates"),
        ("derive evaluator verdicts", "change gates", "open future-final material", "promote"),
        ("hypothesis", "falsifier", "predicted effects", "bounded intervention"),
    ),
    RoleContract(
        InvestigatorRole.SKEPTIC,
        "Search for alternative explanations, omitted assumptions, and verifier gaps.",
        ("propose counterexamples", "request adversarial probes", "challenge attribution"),
        ("delete favorable evidence", "rewrite verdicts", "weaken resource limits"),
        ("alternative explanations", "counterexample plan", "disconfirming probe"),
    ),
    RoleContract(
        InvestigatorRole.REPLICATOR,
        "Repeat frozen experiments across declared seeds, nodes, and providers.",
        ("execute frozen plans", "compare identified results", "report divergence"),
        ("change the candidate", "change the plan", "repair from selection results"),
        ("execution identities", "raw observations", "reconciliation record"),
    ),
    RoleContract(
        InvestigatorRole.SYNTHESIZER,
        "Propose compact principles and strategies from eligible causal attributions.",
        ("cluster eligible evidence", "propose principles", "propose strategies"),
        ("erase raw evidence", "claim global transfer", "accept its own proposal"),
        ("source lineage", "scope", "falsifier", "counterexamples", "transfer status"),
    ),
    RoleContract(
        InvestigatorRole.AUDITOR,
        "Check identity, lineage, contamination, budgets, and authority boundaries.",
        ("reject malformed records", "flag contamination", "halt on authority expansion"),
        ("optimize task metrics", "silently discard failures", "issue formal conformance"),
        ("reason codes", "identity checks", "budget checks", "visibility checks"),
    ),
)
