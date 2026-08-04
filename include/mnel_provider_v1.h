#ifndef MNEL_PROVIDER_V1_H
#define MNEL_PROVIDER_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MNEL_PROVIDER_ABI_VERSION_V1 1u
#define MNEL_PROVIDER_ENTRY_SYMBOL_V1 "mnel_provider_entry_v1"
#define MNEL_AUTHORITY_DIAGNOSTIC_ONLY 1u
#define MNEL_VERDICT_SEMANTICS_NOT_A_VERDICT 1u

typedef struct {
    uint8_t bytes[32];
} mnel_digest32;

typedef struct {
    const uint8_t *data;
    size_t len;
} mnel_byte_view;

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t len;
} mnel_mutable_byte_buffer;

typedef struct {
    uint64_t wall_time_ns;
    uint64_t operation_limit;
    uint64_t memory_bytes;
} mnel_resource_budget_v1;

typedef struct {
    uint32_t schema_version;
    uint32_t reserved;
    mnel_digest32 snapshot_identity;
    mnel_digest32 feature_extractor_identity;
    mnel_byte_view payload;
} mnel_snapshot_view_v1;

typedef struct {
    uint32_t abi_version;
    uint32_t reserved;
    mnel_digest32 declaration_identity;
    mnel_digest32 model_identity;
    mnel_digest32 calibration_identity;
    mnel_digest32 query_identity;
    const mnel_snapshot_view_v1 *snapshots;
    size_t snapshot_count;
    mnel_resource_budget_v1 budget;
} mnel_provider_query_v1;

typedef uint32_t mnel_provider_status_v1;
#define MNEL_PROVIDER_COMPLETED 0u
#define MNEL_PROVIDER_ABSTAINED 1u
#define MNEL_PROVIDER_INVALID_INPUT 2u
#define MNEL_PROVIDER_BUDGET_EXCEEDED 3u
#define MNEL_PROVIDER_OUT_OF_DISTRIBUTION 4u
#define MNEL_PROVIDER_RUNTIME_ERROR 5u

typedef uint32_t mnel_output_kind_v1;
#define MNEL_OUTPUT_LATENT_DISCREPANCY 1u
#define MNEL_OUTPUT_STRUCTURAL_DISCREPANCY 2u
#define MNEL_OUTPUT_ANOMALY_SCORE 3u
#define MNEL_OUTPUT_PAIR_SIMILARITY 4u
#define MNEL_OUTPUT_NEXT_STATE_DISTRIBUTION 5u
#define MNEL_OUTPUT_FEATURE_CONTRIBUTIONS 6u
#define MNEL_OUTPUT_CANDIDATE_RANKING 7u

#define MNEL_RESULT_OUT_OF_DISTRIBUTION (1ull << 0)
#define MNEL_RESULT_CALIBRATION_REQUIRED (1ull << 1)
#define MNEL_RESULT_TRUNCATED_PAYLOAD (1ull << 2)

typedef struct {
    uint32_t abi_version;
    mnel_provider_status_v1 status;
    mnel_output_kind_v1 output_kind;
    uint32_t calibration_band;
    double scalar_value;
    uint64_t flags;
    mnel_mutable_byte_buffer observation_payload;
    uint32_t authority;
    uint32_t verdict_semantics;
} mnel_provider_result_v1;

struct mnel_provider_descriptor_v1;

typedef int32_t (*mnel_provider_infer_v1)(
    void *context,
    const mnel_provider_query_v1 *query,
    mnel_provider_result_v1 *result
);

typedef struct mnel_provider_descriptor_v1 {
    uint32_t abi_version;
    uint32_t reserved;
    mnel_byte_view provider_id;
    mnel_byte_view provider_version;
    mnel_digest32 declaration_identity;
    void *implementation_context;
    mnel_provider_infer_v1 infer;
} mnel_provider_descriptor_v1;

typedef const mnel_provider_descriptor_v1 *(*mnel_provider_entry_v1)(void);

/*
 * Every shared provider library exposes:
 *
 *   const mnel_provider_descriptor_v1 *mnel_provider_entry_v1(void);
 *
 * The descriptor and referenced identifier bytes must remain valid for the lifetime of
 * the loaded library. The host owns query memory and the result payload buffer.
 * Providers may return diagnostic output only; this ABI intentionally has no evaluator
 * verdict, conformance, acceptance, or promotion field.
 */

#ifdef __cplusplus
}
#endif

#endif
