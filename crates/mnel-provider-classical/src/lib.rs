//! A small executable HMM-style diagnostic provider.
//!
//! The provider is intentionally boring: it consumes a compact state-sequence snapshot,
//! computes a bounded negative log likelihood against a fixed transition model, and
//! returns an anomaly observation. It has no evaluator or promotion authority.

use mnel_provider_api::OUTPUT_ANOMALY_SCORE;
use mnel_provider_sdk::{DiagnosticResult, Invocation, LearnedProvider, ProviderError};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

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

/// The portable artifact projection emitted by MNEL's Python transition-frequency
/// reference provider. This parser is intentionally narrower than a general model
/// runtime and has no evaluator or promotion fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionFrequencyArtifact {
    pub provider_id: String,
    pub training_dataset_identity: String,
    pub training_record_ids: Vec<String>,
    pub feature_extractor_identity: String,
    pub training_code_identity: String,
    pub calibration_identity: String,
    pub calibration_dataset_identity: Option<String>,
    pub transition_counts: BTreeMap<String, u64>,
    pub total_count: u64,
    pub model_identity: String,
    pub artifact_identity: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionDiagnostic {
    pub score_millionths: u64,
    pub abstained: bool,
    pub out_of_distribution: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactError(pub String);

impl core::fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactError {}

#[derive(Deserialize)]
struct RawTransitionArtifact {
    schema: String,
    provider_id: String,
    training_dataset_identity: String,
    training_record_ids: Vec<String>,
    feature_extractor_identity: String,
    training_code_identity: String,
    calibration_identity: String,
    #[serde(default)]
    calibration_dataset_identity: Option<String>,
    transition_counts: BTreeMap<String, u64>,
    total_count: u64,
    authority: String,
    semantics: String,
    model_identity: String,
    artifact_identity: String,
}

const ARTIFACT_SCHEMA: &str = "mnel-learned-provider-artifact/0.4";
const PROVIDER_ID: &str = "mnel-reference-transition-frequency/0.4";
const AUTHORITY: &str = "diagnostic-only";
const SEMANTICS: &str = "learned-provider-artifact; diagnostic-only; not-a-verdict";

impl TransitionFrequencyArtifact {
    pub fn from_json(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() > 256 * 1024 || bytes.is_empty() {
            return Err(ArtifactError(
                "artifact is empty or exceeds 256 KiB".to_owned(),
            ));
        }
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|error| ArtifactError(format!("artifact JSON is malformed: {error}")))?;
        let raw: RawTransitionArtifact = serde_json::from_value(value.clone())
            .map_err(|error| ArtifactError(format!("artifact fields are malformed: {error}")))?;
        if raw.schema != ARTIFACT_SCHEMA || raw.provider_id != PROVIDER_ID {
            return Err(ArtifactError(
                "unsupported transition artifact family".to_owned(),
            ));
        }
        if raw.authority != AUTHORITY || raw.semantics != SEMANTICS {
            return Err(ArtifactError(
                "artifact authority or semantics are invalid".to_owned(),
            ));
        }
        if raw.training_record_ids.is_empty() || raw.total_count == 0 || raw.total_count > 1_000_000
        {
            return Err(ArtifactError(
                "artifact training counts are outside bounds".to_owned(),
            ));
        }
        if raw.transition_counts.is_empty()
            || raw.transition_counts.values().any(|count| *count == 0)
            || raw
                .transition_counts
                .values()
                .try_fold(0_u64, |total, count| total.checked_add(*count))
                != Some(raw.total_count)
            || raw
                .transition_counts
                .keys()
                .any(|key| key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(ArtifactError("transition counts are malformed".to_owned()));
        }
        let supplied_artifact = raw.artifact_identity.clone();
        let mut artifact_value = value.clone();
        let object = artifact_value
            .as_object_mut()
            .ok_or_else(|| ArtifactError("artifact must be a JSON object".to_owned()))?;
        object.remove("artifact_identity");
        if sha256_identity(&canonical_json(&artifact_value)) != supplied_artifact {
            return Err(ArtifactError(
                "artifact identity does not verify".to_owned(),
            ));
        }
        let mut model_value = value;
        let model_object = model_value
            .as_object_mut()
            .ok_or_else(|| ArtifactError("artifact must be a JSON object".to_owned()))?;
        model_object.remove("artifact_identity");
        model_object.remove("model_identity");
        if sha256_identity(&canonical_json(&model_value)) != raw.model_identity {
            return Err(ArtifactError("model identity does not verify".to_owned()));
        }
        Ok(Self {
            provider_id: raw.provider_id,
            training_dataset_identity: raw.training_dataset_identity,
            training_record_ids: raw.training_record_ids,
            feature_extractor_identity: raw.feature_extractor_identity,
            training_code_identity: raw.training_code_identity,
            calibration_identity: raw.calibration_identity,
            calibration_dataset_identity: raw.calibration_dataset_identity,
            transition_counts: raw.transition_counts,
            total_count: raw.total_count,
            model_identity: raw.model_identity,
            artifact_identity: supplied_artifact,
        })
    }

    /// Decode the compact MNEL-T1 transition view and reproduce the Python score.
    pub fn infer_transition(&self, payload: &[u8]) -> Result<TransitionDiagnostic, ArtifactError> {
        if payload.len() < 11 || &payload[..7] != b"MNEL-T1" {
            return Err(ArtifactError(
                "transition snapshot header is invalid".to_owned(),
            ));
        }
        let left_len = u16::from_be_bytes([payload[7], payload[8]]) as usize;
        let right_len = u16::from_be_bytes([payload[9], payload[10]]) as usize;
        if left_len == 0 || right_len == 0 || payload.len() != 11 + left_len + right_len {
            return Err(ArtifactError(
                "transition snapshot length is invalid".to_owned(),
            ));
        }
        let left = &payload[11..11 + left_len];
        let right = &payload[11 + left_len..];
        let mut key_material = Vec::with_capacity(left.len() + right.len() + 1);
        key_material.extend_from_slice(left);
        key_material.push(0);
        key_material.extend_from_slice(right);
        let key = hex(&Sha256::digest(key_material));
        let count = self.transition_counts.get(&key).copied().unwrap_or(0);
        if count == 0 {
            return Ok(TransitionDiagnostic {
                score_millionths: 0,
                abstained: true,
                out_of_distribution: true,
            });
        }
        Ok(TransitionDiagnostic {
            score_millionths: count.saturating_mul(1_000_000) / self.total_count,
            abstained: false,
            out_of_distribution: false,
        })
    }
}

fn sha256_identity(value: &[u8]) -> String {
    format!("sha256:{}", hex(&Sha256::digest(value)))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[usize::from(byte >> 4)] as char);
        result.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    result
}

fn canonical_json(value: &Value) -> Vec<u8> {
    match value {
        Value::Null => b"null".to_vec(),
        Value::Bool(value) => value.to_string().into_bytes(),
        Value::Number(value) => value.to_string().into_bytes(),
        Value::String(value) => serde_json::to_vec(value).unwrap_or_default(),
        Value::Array(values) => {
            let mut output = Vec::from(b"[".as_slice());
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(canonical_json(item));
            }
            output.push(b']');
            output
        }
        Value::Object(values) => {
            let mut entries: Vec<(&String, &Value)> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            let mut output = Vec::from(b"{".as_slice());
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(serde_json::to_vec(key).unwrap_or_default());
                output.push(b':');
                output.extend(canonical_json(item));
            }
            output.push(b'}');
            output
        }
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

    #[test]
    fn parses_python_artifact_and_reproduces_transition_inference() {
        let mut base = serde_json::json!({
            "authority": AUTHORITY,
            "calibration_identity": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
            "feature_extractor_identity": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
            "provider_id": PROVIDER_ID,
            "schema": ARTIFACT_SCHEMA,
            "semantics": SEMANTICS,
            "total_count": 2,
            "training_code_identity": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
            "training_dataset_identity": "sha256:6666666666666666666666666666666666666666666666666666666666666666",
            "training_record_ids": ["sha256:7777777777777777777777777777777777777777777777777777777777777777"],
            "transition_counts": {},
        });
        let mut key_material = b"cold".to_vec();
        key_material.push(0);
        key_material.extend_from_slice(b"warm");
        let key = hex(&Sha256::digest(key_material));
        match base.as_object_mut() {
            Some(object) => {
                object.insert("transition_counts".to_owned(), serde_json::json!({key: 2}));
            }
            None => panic!("artifact fixture must be an object"),
        }
        let model_identity = sha256_identity(&canonical_json(&base));
        match base.as_object_mut() {
            Some(object) => {
                object.insert("model_identity".to_owned(), Value::String(model_identity));
            }
            None => panic!("artifact fixture must be an object"),
        }
        let artifact_identity = sha256_identity(&canonical_json(&base));
        match base.as_object_mut() {
            Some(object) => {
                object.insert(
                    "artifact_identity".to_owned(),
                    Value::String(artifact_identity),
                );
            }
            None => panic!("artifact fixture must be an object"),
        }
        let artifact = match TransitionFrequencyArtifact::from_json(&canonical_json(&base)) {
            Ok(artifact) => artifact,
            Err(error) => panic!("artifact fixture should parse: {error}"),
        };
        let mut payload = b"MNEL-T1".to_vec();
        payload.extend_from_slice(&(4_u16).to_be_bytes());
        payload.extend_from_slice(&(4_u16).to_be_bytes());
        payload.extend_from_slice(b"coldwarm");
        let result = match artifact.infer_transition(&payload) {
            Ok(result) => result,
            Err(error) => panic!("transition fixture should infer: {error}"),
        };
        assert_eq!(result.score_millionths, 1_000_000);
        assert!(!result.abstained);
        assert!(!result.out_of_distribution);
        let unknown = match artifact.infer_transition(b"MNEL-T1\0\x04\0\x04coldcool") {
            Ok(result) => result,
            Err(error) => panic!("unknown transition should be bounded: {error}"),
        };
        assert!(unknown.abstained);
        assert!(unknown.out_of_distribution);
    }

    #[test]
    fn rejects_artifact_identity_and_snapshot_length_drift() {
        let error = match TransitionFrequencyArtifact::from_json(
            b"{\"schema\":\"mnel-learned-provider-artifact/0.4\"}",
        ) {
            Ok(_) => panic!("incomplete artifact must be rejected"),
            Err(error) => error,
        };
        assert!(error.0.contains("malformed"));
    }

    #[test]
    fn parses_the_checked_in_python_artifact_fixture() {
        let artifact = match TransitionFrequencyArtifact::from_json(include_bytes!(
            "../tests/fixtures/transition-frequency-artifact.json"
        )) {
            Ok(artifact) => artifact,
            Err(error) => panic!("checked-in artifact should parse: {error}"),
        };
        let mut payload = b"MNEL-T1".to_vec();
        payload.extend_from_slice(&(4_u16).to_be_bytes());
        payload.extend_from_slice(&(4_u16).to_be_bytes());
        payload.extend_from_slice(b"coldwarm");
        let result = match artifact.infer_transition(&payload) {
            Ok(result) => result,
            Err(error) => panic!("checked-in artifact should infer: {error}"),
        };
        assert_eq!(result.score_millionths, 1_000_000);
        assert!(!result.abstained);
    }
}
