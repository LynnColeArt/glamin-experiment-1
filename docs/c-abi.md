# Glamin C ABI

## Status

The first four ABI slices are implemented in Glamin and consumed by this
experiment. ABI version 4 provides version discovery, worker-runtime lifecycle,
opaque runtime, flat-index, generation, and pin handles, synchronous float32
add/search, immutable generation publication, and caller-owned diagnostics and
results. It also loads contract-validated persistent flat artifacts.

Asynchronous request/traversal and trace functions later in this document
remain proposals. The authoritative implemented header is
`third_party/glamin/include/glamin_runtime.h`.

## Design Rules

- Export only C-compatible functions and data.
- Use `ISO_C_BINDING` and explicit `bind(C, name=...)` on the Fortran side.
- Never expose a Fortran derived-type layout.
- Represent runtime objects with opaque 64-bit handles.
- Use fixed-width integers and IEEE 32-bit vectors.
- Pass arrays as pointer, count, dimension, and stride.
- Keep allocation and deallocation within the same runtime.
- Return status codes; no callback or exception crosses the boundary unchecked.
- Make ABI, capability, contract, and generation versions queryable.

## Handle Types

```c
typedef uint64_t glamin_runtime_t;
typedef uint64_t glamin_index_t;
typedef uint64_t glamin_generation_t;
typedef uint64_t glamin_generation_pin_t;
```

The following handles are proposed but not implemented:

```c
typedef uint64_t glamin_request_t;
typedef uint64_t glamin_trace_t;
```

Zero is invalid for every handle. Handles are process-local and must not be
serialized as durable object identities.

## Status Model

```c
typedef enum glamin_status {
    GLAMIN_STATUS_OK = 0,
    GLAMIN_STATUS_UNKNOWN = 1,
    GLAMIN_STATUS_INVALID_ARGUMENT = 2,
    GLAMIN_STATUS_OUT_OF_MEMORY = 3,
    GLAMIN_STATUS_NOT_READY = 4,
    GLAMIN_STATUS_CANCELLED = 5,
    GLAMIN_STATUS_BUFFER_TOO_SMALL = 6
} glamin_status;
```

Lifecycle failures record a global or runtime-scoped diagnostic retrievable
through a caller-provided buffer. Diagnostics are not a stable
machine-readable API. The current registry supports 64 simultaneous runtimes,
256 simultaneous indexes, 256 simultaneous generations, and 1024 simultaneous
pins. Creation fails closed when a registry is full.

## Runtime Lifecycle

```c
uint32_t glamin_abi_version(void);

glamin_status glamin_runtime_create(
    uint32_t worker_count,
    glamin_runtime_t *out_runtime);

glamin_status glamin_runtime_destroy(glamin_runtime_t runtime);

glamin_status glamin_last_error(
    glamin_runtime_t runtime,
    char *buffer,
    uint64_t capacity,
    uint64_t *out_required);
```

Destroying a runtime with live indexes returns `NOT_READY`. Every generation is
bound to an index, so this also prevents destruction while a generation or pin
remains live. Runtime creation and destruction are externally serialized in
the current ABI.

## Implemented Flat Index

```c
typedef enum glamin_metric {
    GLAMIN_METRIC_L2 = 0,
    GLAMIN_METRIC_INNER_PRODUCT = 1
} glamin_metric;

glamin_status glamin_flat_index_create(
    glamin_runtime_t runtime,
    uint32_t dimension,
    glamin_metric metric,
    glamin_index_t *out_index);

glamin_status glamin_flat_index_load_artifact(
    glamin_runtime_t runtime,
    const char *artifact_directory,
    uint64_t artifact_directory_length,
    const char *space_id,
    uint64_t space_id_length,
    glamin_metric metric,
    glamin_index_t *out_index,
    uint32_t *out_dimension,
    uint64_t *out_vector_count);

glamin_status glamin_index_add_f32(
    glamin_runtime_t runtime,
    glamin_index_t index,
    const float *vectors,
    uint64_t vector_count,
    uint32_t vector_stride);

glamin_status glamin_index_search_f32(
    glamin_runtime_t runtime,
    glamin_index_t index,
    const float *queries,
    uint64_t query_count,
    uint32_t query_stride,
    uint32_t k,
    float *out_distances,
    uint64_t *out_labels);

glamin_status glamin_index_destroy(
    glamin_runtime_t runtime,
    glamin_index_t index);
```

Input strides are measured in float elements. Search writes exactly
`query_count * k` dense results in query-major order. Labels are zero-based row
numbers assigned in add order. The current implementation requires `k` not to
exceed the number of indexed rows.

Persistent loading expects `vector_layout.json`, `vectors.bin`, and
`contracts.json` in one artifact directory. It validates the requested space,
layout dimension, requested metric, declared normalization, canonical
space-contract SHA-256 hash, embedder structure, and registered contract hooks
before returning an index. Failed loads clear every output and publish no index.
ABI version 4 does not yet bind checksums for `vectors.bin` and
`vector_layout.json` into a generation manifest; complete artifact provenance
remains a later hardening step.

## Implemented Generation Lifecycle

```c
glamin_status glamin_generation_create(
    glamin_runtime_t runtime,
    glamin_index_t index,
    const char *label,
    uint64_t label_length,
    glamin_generation_t *out_generation);

glamin_status glamin_generation_activate(
    glamin_runtime_t runtime,
    glamin_generation_t generation);

glamin_status glamin_generation_deactivate(
    glamin_runtime_t runtime);

glamin_status glamin_generation_pin_active(
    glamin_runtime_t runtime,
    glamin_generation_pin_t *out_pin,
    glamin_generation_t *out_generation);

glamin_status glamin_generation_unpin(
    glamin_runtime_t runtime,
    glamin_generation_pin_t pin);

glamin_status glamin_generation_retire(
    glamin_runtime_t runtime,
    glamin_generation_t generation);

glamin_status glamin_generation_label(
    glamin_runtime_t runtime,
    glamin_generation_t generation,
    char *buffer,
    uint64_t capacity,
    uint64_t *out_required);

glamin_status glamin_generation_search_f32(
    glamin_runtime_t runtime,
    glamin_generation_pin_t pin,
    const float *queries,
    uint64_t query_count,
    uint32_t query_stride,
    uint32_t k,
    float *out_distances,
    uint64_t *out_labels);
```

Creating a generation binds one existing flat index to an immutable generation.
The index rejects add and destroy operations until the generation is reclaimed.
Activation changes which generation future pins observe; existing pins continue
to identify and search their original generation. An active generation must be
superseded or explicitly deactivated before retirement.

Retirement prevents future activation but does not invalidate existing pins or
their synchronous searches. A retired generation is reclaimed after its last
pin is released, at which point its index may be destroyed. Labels contain 1 to
128 non-null bytes and are copied to caller-owned buffers.

## Proposed Traversal Lifecycle

```c
typedef struct glamin_vector_view_f32 {
    const float *data;
    uint64_t count;
    uint32_t dimension;
    uint64_t row_stride_bytes;
    uint32_t memory_location;
} glamin_vector_view_f32;

glamin_status glamin_traverse_submit(
    glamin_runtime_t runtime,
    glamin_generation_pin_t generation_pin,
    glamin_trace_t trace,
    const glamin_vector_view_f32 *queries,
    glamin_request_t *out_request);

glamin_status glamin_request_poll(
    glamin_runtime_t runtime,
    glamin_request_t request,
    uint32_t *out_state);

glamin_status glamin_request_wait(
    glamin_runtime_t runtime,
    glamin_request_t request,
    int64_t timeout_milliseconds);

glamin_status glamin_request_cancel(
    glamin_runtime_t runtime,
    glamin_request_t request);

glamin_status glamin_request_result_shape(
    glamin_runtime_t runtime,
    glamin_request_t request,
    uint64_t *out_count,
    uint32_t *out_dimension);

glamin_status glamin_request_copy_result_f32(
    glamin_runtime_t runtime,
    glamin_request_t request,
    float *output,
    uint64_t output_float_capacity);

glamin_status glamin_request_release(
    glamin_runtime_t runtime,
    glamin_request_t request);
```

The initial ABI copies results into caller-owned host memory. Device-resident
views can be added only after their lifetime and synchronization contracts are
specified.

## Proposed Trace Lifecycle

```c
glamin_status glamin_trace_create(
    glamin_runtime_t runtime,
    glamin_trace_t *out_trace);

glamin_status glamin_trace_snapshot(
    glamin_runtime_t runtime,
    glamin_trace_t trace,
    const char *output_path,
    uint64_t output_path_length);

glamin_status glamin_trace_destroy(
    glamin_runtime_t runtime,
    glamin_trace_t trace);
```

Trace persistence is append-only. Destroying an in-memory trace handle does not
delete a committed trace artifact.

## Intended Threading and Reentrancy

- Current ABI lifecycle, add, search, generation, and pin calls are externally
  serialized.
- Proposed submit, poll, wait, and result-copy operations must be thread-safe.
- A request may be waited on by one host thread at a time.
- Callback execution is excluded from the first ABI revision.
- Fortran worker threads never call into C++ application objects.
- Activation and pinning have a defined total order within one runtime.

## Ownership Summary

| Resource | Created by | Released by |
| --- | --- | --- |
| Runtime | `glamin_runtime_create` | `glamin_runtime_destroy` |
| Flat index | `glamin_flat_index_create` | `glamin_index_destroy` |
| Persistent flat index | `glamin_flat_index_load_artifact` | `glamin_index_destroy` |
| Generation | `glamin_generation_create` | Retirement plus last pin release |
| Pin | `glamin_generation_pin_active` | `glamin_generation_unpin` |
| Request (proposed) | `glamin_traverse_submit` | `glamin_request_release` |
| Trace (proposed) | `glamin_trace_create` | `glamin_trace_destroy` |
| Synchronous input vector memory | Caller | Caller after the ABI call returns |
| Output vector memory | Caller | Caller |

The implemented flat add copies inputs before returning. Flat and pinned
generation searches consume queries and write results before returning. Future
asynchronous submission should copy on submit unless a separate retained-buffer
contract is introduced explicitly.

Releasing an index while a mounted generation still refers to it returns
`NOT_READY`.
