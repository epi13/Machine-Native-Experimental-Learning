//! Safe Rust-facing provider contract layered over the versioned C ABI.

use mnel_provider_api::{
    Digest32, OutputKindV1, ProviderQueryV1, ResourceBudgetV1, SnapshotViewV1, ABI_VERSION_V1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvocationIdentity {
    pub declaration: Digest32,
    pub model: Digest32,
    pub calibration: Digest32,
    pub query: Digest32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    pub wall_time_ns: u64,
    pub operation_limit: u64,
    pub memory_bytes: u64,
}

impl ResourceBudget {
    pub fn validate(self) -> Result<Self, ProviderError> {
        if self.wall_time_ns == 0 || self.operation_limit == 0 || self.memory_bytes == 0 {
            return Err(ProviderError::InvalidBudget);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SnapshotRef<'a> {
    pub schema_version: u32,
    pub identity: Digest32,
    pub feature_extractor_identity: Digest32,
    pub payload: &'a [u8],
}

#[derive(Debug)]
pub struct Invocation<'a> {
    identities: InvocationIdentity,
    budget: ResourceBudget,
    raw_snapshots: Vec<SnapshotViewV1>,
    snapshot_lifetimes: Vec<SnapshotRef<'a>>,
}

impl<'a> Invocation<'a> {
    pub fn new(
        identities: InvocationIdentity,
        budget: ResourceBudget,
        snapshots: Vec<SnapshotRef<'a>>,
    ) -> Result<Self, ProviderError> {
        let budget = budget.validate()?;
        if snapshots.is_empty() {
            return Err(ProviderError::MissingSnapshots);
        }
        if snapshots.iter().any(|snapshot| snapshot.payload.is_empty()) {
            return Err(ProviderError::EmptySnapshot);
        }
        let raw_snapshots = snapshots
            .iter()
            .map(|snapshot| SnapshotViewV1 {
                schema_version: snapshot.schema_version,
                reserved: 0,
                snapshot_identity: snapshot.identity,
                feature_extractor_identity: snapshot.feature_extractor_identity,
                payload: mnel_provider_api::ByteView {
                    data: snapshot.payload.as_ptr(),
                    len: snapshot.payload.len(),
                },
            })
            .collect();
        Ok(Self {
            identities,
            budget,
            raw_snapshots,
            snapshot_lifetimes: snapshots,
        })
    }

    pub fn as_raw(&self) -> ProviderQueryV1 {
        ProviderQueryV1 {
            abi_version: ABI_VERSION_V1,
            reserved: 0,
            declaration_identity: self.identities.declaration,
            model_identity: self.identities.model,
            calibration_identity: self.identities.calibration,
            query_identity: self.identities.query,
            snapshots: self.raw_snapshots.as_ptr(),
            snapshot_count: self.raw_snapshots.len(),
            budget: ResourceBudgetV1 {
                wall_time_ns: self.budget.wall_time_ns,
                operation_limit: self.budget.operation_limit,
                memory_bytes: self.budget.memory_bytes,
            },
        }
    }

    pub fn snapshots(&self) -> &[SnapshotRef<'a>] {
        &self.snapshot_lifetimes
    }

    pub fn identities(&self) -> InvocationIdentity {
        self.identities
    }

    pub fn budget(&self) -> ResourceBudget {
        self.budget
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticResult {
    pub output_kind: OutputKindV1,
    pub value: f64,
    pub calibration_band: u32,
    pub out_of_distribution: bool,
    pub payload: Vec<u8>,
}

impl DiagnosticResult {
    pub fn validate(self) -> Result<Self, ProviderError> {
        if !self.value.is_finite() {
            return Err(ProviderError::NonFiniteResult);
        }
        Ok(self)
    }
}

pub trait LearnedProvider: Send + Sync {
    fn infer(&self, invocation: &Invocation<'_>) -> Result<DiagnosticResult, ProviderError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderError {
    InvalidBudget,
    MissingSnapshots,
    EmptySnapshot,
    NonFiniteResult,
    Abstained,
    OutOfDistribution,
    BudgetExceeded,
    RuntimeFailure,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> Digest32 {
        Digest32 { bytes: [value; 32] }
    }

    #[test]
    fn builds_identity_bound_query_without_copying_payload() {
        let payload = [1_u8, 2, 3, 4];
        let invocation = Invocation::new(
            InvocationIdentity {
                declaration: digest(1),
                model: digest(2),
                calibration: digest(3),
                query: digest(4),
            },
            ResourceBudget {
                wall_time_ns: 1_000_000,
                operation_limit: 10_000,
                memory_bytes: 1_048_576,
            },
            vec![SnapshotRef {
                schema_version: 1,
                identity: digest(5),
                feature_extractor_identity: digest(6),
                payload: &payload,
            }],
        );
        let invocation = match invocation {
            Ok(value) => value,
            Err(error) => panic!("expected valid invocation, got {error:?}"),
        };
        let raw = invocation.as_raw();
        assert_eq!(raw.abi_version, ABI_VERSION_V1);
        assert_eq!(raw.snapshot_count, 1);
        assert_eq!(invocation.snapshots()[0].payload.as_ptr(), payload.as_ptr());
    }

    #[test]
    fn rejects_unbounded_or_empty_invocations() {
        let result = Invocation::new(
            InvocationIdentity {
                declaration: digest(1),
                model: digest(2),
                calibration: digest(3),
                query: digest(4),
            },
            ResourceBudget {
                wall_time_ns: 0,
                operation_limit: 1,
                memory_bytes: 1,
            },
            Vec::new(),
        );
        assert_eq!(result.err(), Some(ProviderError::InvalidBudget));
    }
}
