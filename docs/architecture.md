# Architecture

## Purpose

Glamin Experiment 1 tests whether executable geometry can participate directly
in language-model inference. The design keeps the model, executable geometry,
source records, and authority system distinct enough to govern while allowing
their computation to become tightly integrated.

## System Context

```text
clients
   |
   v
cognitive runtime --------------------------------------+
   |                                                     |
   +--> inference engine                                 |
   |       |                                             |
   |       +--> selected hidden-state hook               |
   |                    |                                |
   |                    v                                |
   |              Glamin adapter --> Fortran runtime     |
   |                    |              |                 |
   |                    |              +--> generations  |
   |                    |              +--> traces       |
   |                    |              +--> indexes      |
   |                    |                                |
   |                    +--> residual/result vector -----+
   |
   +--> source ledger
   +--> capability broker
   +--> immutable audit trace
```

## Component Responsibilities

### C++ cognitive runtime

The host owns process lifecycle and coordinates inference. It selects the
active model and geometry generation, maintains request-local state, mediates
actions, and converts native errors into typed failures.

It must not interpret private Fortran memory layouts or mutate a generation
that is visible to an active request.

### Inference engine

The implemented engine is pinned `llama.cpp` revision `ecd99d6`, initially
supporting Qwen3 on CPU. Its evaluation callback pauses after one named
`l_out-N` tensor. The hook can project the last-token state, one explicit token
row, or every token row. In scan mode it searches a pinned Glamin generation for
each candidate and chooses the globally nearest result. It resolves the label
through a generation-qualified residual ledger, writes the gated residual into
the live last-token row, and lets inference continue.

### Glamin adapter

The adapter is a C++ RAII wrapper over the versioned Glamin C ABI. It translates
host vectors and request state into ABI calls, pins a geometry generation,
waits for or polls traversal, and retrieves a result vector and trace metadata.

The implemented slices own real Glamin worker-runtime creation, diagnostic
retrieval, move-safe shared lifetimes, synchronous flat-index recall, immutable
generation publication, activation, stable pins, retirement, and rollback.
ABI version 4 also loads compiler-emitted persistent artifacts after validating
their selected space contract. Asynchronous traversal and trace calls are later
slices.

The adapter also implements the fixed hidden-state projection and residual
boundary. `PersistentHookGenerationStore` validates and mounts one hash-bound
artifact containing the active-model contract, projection, Glamin compiler
artifact, residual labels, and residual payloads. Its pin API prevents those
resources from being selected from different generations. The adapter owns no
geometry semantics; it enforces lifetimes and makes invalid states difficult
to express.

### Fortran Glamin runtime

Glamin owns indexes, executable geometry, request processing, snapshots,
contracts, and trace traversal. It exports opaque numeric handles and plain C
data through `ISO_C_BINDING`.

### Source ledger

Vectors are not exact memories. A separate append-only ledger retains original
events, documents, provenance, supersession relationships, and payloads.
Glamin stores addresses and executable representations; the ledger provides
recoverable specifics.

The ledger is outside Experiment 1, but its boundary is preserved in the
architecture so document vectors are never mistaken for source records.

### Capability broker

The broker is the only component authorized to perform external effects. A
Glamin mini-app may select or propose an action, but it cannot acquire authority
merely by being near the current state.

### Audit trace

Every traversal records the model contract, geometry generation, input identity,
selected behavior, result identity, and any requested or completed effect. The
trace is append-only and sufficient to explain which deployed artifacts
participated in a result.

## Inference Hook

For address state `h_address` and action-position state `h_action` at layer `l`,
the implemented synchronous hook is:

```text
q = project_in(h_address)
address = glamin_search(q, pinned_generation)
r = residual_ledger[pinned_generation, address]
h_action' = h_action + fixed_gate * r
```

The current slice uses a fixed gate and a persisted, generation-qualified
residual ledger loaded into the host. The intended trained form remains
`gate(h, r) * project_out(r)`. A production design could also overlap
asynchronous traversal with independent model work where the graph permits it.

The implemented factorized probe adds a bounded join path:

```text
entity = gated_search(entity_space, token_rows)
relation = gated_search(relation_space, token_rows)
variants = reviewed_tuple_ledger[entity, relation]
action = gated_nearest(h_action, variants)
h_action' = h_action + fixed_gate * action.residual
```

The entity, relation, exact-tuple, and action-context checks all fail closed.
The probe currently keeps tuple variants in host memory; a deployable
factorized generation must persist and hash-bind the complete join contract.

The two-stage probe separates those checks from the tensor that receives the
action:

```text
l_out-34 -> factor and tuple authorization -> no early tensor mutation
                                                |
                                                v
                         accepted tuple target state
                                                |
                                                v
l_out-35 --------------------------------> gated final-row interpolation
```

The action tensor is never written when authorization fails. The current probe
uses a gate of `1.0`, replacing the final row with a reviewed tuple target state.
This ledger is host-resident and is not yet part of the persisted artifact
contract.

## Request Lifecycle

```text
1. Accept observation and request identity.
2. Resolve model, adapter, and active geometry generation.
3. Pin the generation for the request or trace segment.
4. Run inference until the configured hook.
5. Project the configured address candidate or candidates into the contracted
   Glamin space.
6. Submit and complete traversal.
7. Validate the returned contract and dimensions.
8. Apply the authorized residual or later target state to the action-position
   tensor.
9. Continue inference or suspend for an authorized action.
10. Commit trace metadata and release the generation pin.
```

## Artifact Contract

A deployable configuration binds these artifacts:

```text
model hash
model architecture and layer hook
projection adapter hash
address-selection policy
residual label and payload hashes
Glamin ABI version
space and embedder contracts
geometry generation
capability manifest
probe and evaluation set versions
```

Changing any representation-defining artifact creates a new deployment
contract. Geometry compiled for one model or layer cannot silently load into
another.

The implemented `gx1-hook-v1` manifest realizes the model, projection,
address-selection, Glamin, and residual subset of this contract. Capability
manifests, signatures, probes, and deployment metadata remain control-plane
work.

## Concurrency Model

- Generations are immutable after publication.
- Activation is atomic from the perspective of new pins.
- A request operates against exactly one pinned generation per trace segment.
- Retirement removes a generation from future selection but not from active
  leases.
- Destruction occurs only after the last lease is released.
- External actions are never executed on Glamin worker threads.

## Initial Performance Position

Experiment 1 runs on CPU and invokes Glamin once per request at one fixed model
layer. This avoids hiding semantic failures behind GPU synchronization and
device-transfer costs.

Only after correctness is established should the project explore token-level
hooks, batching, asynchronous overlap, or shared device memory.
