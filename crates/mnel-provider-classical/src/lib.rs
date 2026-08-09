//! A small executable HMM-style diagnostic provider.
//!
//! The provider is intentionally boring: it consumes a compact state-sequence snapshot,
//! computes a bounded negative log likelihood against a fixed transition model, and
//! returns an anomaly observation. It has no evaluator or promotion authority.

use mnel_provider_api::OUTPUT_ANOMALY_SCORE;
use mnel_provider_sdk::{DiagnosticResult, Invocation, LearnedProvider, ProviderError};

const STATE_COUNT: usize = 4;
const TRANSITION_PROBABILITIES: [[f64; STATE_COUNT]; STATE_COUNT] = [
    [0.70, 0.20, 0.08, 0.02],
    [0.05, 0.75, 0.15, 0.05],
    [0.10, 0.10, 0.70, 0.10],
    [0.20, 0.10, 0.10, 0.60],
];

#[derive(Clone, Copy, Debug, Default)]
pub struct HiddenMarkovProvider;

impl HiddenMarkovProvider {
    pub const fn new() -> Self {
        Self
    }
}

impl LearnedProvider for HiddenMarkovProvider {
    fn infer(&self, invocation: &Invocation<'_>) -> Result<DiagnosticResult, ProviderError> {
        let snapshot = invocation
            .snapshots()
            .first()
            .ok_or(ProviderError::MissingSnapshots)?;
        let sequence = snapshot.payload;
        let operation_limit = invocation.budget().operation_limit as usize;
        if sequence.len() > operation_limit {
            return Err(ProviderError::BudgetExceeded);
        }
        if sequence
            .iter()
            .any(|state| usize::from(*state) >= STATE_COUNT)
        {
            return Err(ProviderError::OutOfDistribution);
        }

        let mut negative_log_likelihood = 0.0_f64;
        let mut transitions = 0_usize;
        for pair in sequence.windows(2) {
            let probability = TRANSITION_PROBABILITIES[usize::from(pair[0])][usize::from(pair[1])];
            negative_log_likelihood -= probability.ln();
            transitions += 1;
        }
        let score = if transitions == 0 {
            0.0
        } else {
            negative_log_likelihood / transitions as f64
        };
        let calibration_band = if score < 0.6 {
            0
        } else if score < 1.5 {
            1
        } else {
            2
        };
        let payload = format!("transitions={transitions};mean_negative_log_likelihood={score:.6}")
            .into_bytes();
        DiagnosticResult {
            output_kind: OUTPUT_ANOMALY_SCORE,
            value: score,
            calibration_band,
            out_of_distribution: false,
            payload,
        }
        .validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnel_provider_api::Digest32;
    use mnel_provider_sdk::{InvocationIdentity, ResourceBudget, SnapshotRef};

    fn digest(value: u8) -> Digest32 {
        Digest32 { bytes: [value; 32] }
    }

    fn invocation(payload: &[u8], limit: u64) -> Invocation<'_> {
        let result = Invocation::new(
            InvocationIdentity {
                declaration: digest(1),
                model: digest(2),
                calibration: digest(3),
                query: digest(4),
            },
            ResourceBudget {
                wall_time_ns: 1_000_000,
                operation_limit: limit,
                memory_bytes: 1024,
            },
            vec![SnapshotRef {
                schema_version: 1,
                identity: digest(5),
                feature_extractor_identity: digest(6),
                payload,
            }],
        );
        match result {
            Ok(invocation) => invocation,
            Err(_) => panic!("test invocation should be bounded and non-empty"),
        }
    }

    #[test]
    fn produces_a_bounded_diagnostic_score() {
        let result = match HiddenMarkovProvider::new().infer(&invocation(&[0, 0, 1, 1, 2], 10)) {
            Ok(result) => result,
            Err(_) => panic!("provider should infer"),
        };
        assert_eq!(result.output_kind, OUTPUT_ANOMALY_SCORE);
        assert!(result.value.is_finite());
        assert!(!result.payload.is_empty());
    }

    #[test]
    fn respects_operation_budget_and_ood_states() {
        let budget_error = match HiddenMarkovProvider::new().infer(&invocation(&[0, 1, 2], 2)) {
            Ok(_) => panic!("operation budget should be enforced"),
            Err(error) => error,
        };
        assert_eq!(budget_error, ProviderError::BudgetExceeded);
        let ood_error = match HiddenMarkovProvider::new().infer(&invocation(&[0, 9], 10)) {
            Ok(_) => panic!("unknown state should abstain as OOD"),
            Err(error) => error,
        };
        assert_eq!(ood_error, ProviderError::OutOfDistribution);
    }
}
