# Glamin Experiment 1: Project Overview

## Summary

Glamin Experiment 1 explores a model architecture in which a language model and
an executable memory system participate in the same inference process.

Most agent memory systems sit outside the model. A client retrieves records,
places them into a prompt, invokes a stateless model, and stores the response.
Continuity belongs to the client application; memory remains passive context.

This project asks a different question:

> What happens when persistent, executable geometry becomes part of inference
> itself?

The experiment integrates a model with
[Glamin](https://github.com/LynnColeArt/glamin), the Geometric Logical
Application Meta-Instruction Network. Glamin treats vector-space traversal as
control flow. Its indexed objects can describe not only what is similar, but
which behavior should become active and what may happen next.

The intended result is neither an LLM with a memory plugin nor a vector database
attached to an agent. It is a persistent neural-geometric runtime whose model
weights, memories, executable manifolds, and active traces jointly determine
behavior.

## Core Idea

A conventional retrieval system follows this pattern:

```text
query -> retrieve documents -> add documents to prompt -> run model
```

The proposed system places Glamin within the inference loop:

```text
token or observation
        |
        v
transformer layers
        |
        v
project hidden state into Glamin space
        |
        v
traverse executable geometry
   |           |             |
   |           |             +--> continue an active trace
   |           +----------------> retrieve a precise memory
   +----------------------------> select or execute a behavior
        |
        v
project the result into model state
        |
        v
remaining transformer layers
        |
        v
token, internal transition, or authorized action
```

At selected model layers, the initial operation can be expressed as:

```text
query    = project_in(hidden_state)
result   = glamin(query, active_trace, geometry_generation)
hidden'  = hidden_state + gate * project_out(result)
```

Glamin is not called as an optional user-facing tool. It behaves more like a
persistent memory-and-control layer whose result participates directly in the
model's computation.

## Why Glamin

Approximate nearest-neighbor search alone would provide associative recall, but
not procedural continuity. Glamin adds several properties required by this
experiment:

- **Executable geometry:** neighborhoods can contain meta-instructions,
  behaviors, corridors, and trajectories rather than only documents.
- **Asynchronous operation:** search, updates, and snapshots can occur without
  making every caller manage a blocking storage transaction.
- **Space contracts:** vectors are bound to a declared dimension, metric,
  normalization strategy, embedder, model version, and transformation chain.
- **Traceability:** execution can be represented as a path through a manifold,
  providing a persistent account of how behavior unfolded.
- **Snapshot semantics:** geometry can evolve while active work remains pinned
  to a consistent generation.
- **Document/geometry separation:** retrieved content and executable behavior
  can remain distinct, connected only through explicit, versioned transforms.

Glamin's geometric logic layer is still experimental. This project is intended
to test and extend that direction, not to claim that the complete integrated
system already exists.

## Memory Becomes Active

The architecture supports several complementary forms of memory:

| Memory form | Representation | Purpose |
| --- | --- | --- |
| Episodic | Immutable source records and document vectors | Preserve what specifically happened |
| Semantic | Consolidated concepts and relationships | Represent what repeated experiences mean |
| Procedural | Glamin mini-apps, mints, and corridors | Encode what to do in situations like this |
| Working state | Current position, bindings, and active trace | Preserve continuity through an ongoing activity |

The important change is procedural memory. A recurring pattern of successful
behavior can eventually become a bounded mini-application embedded in Glamin.
Similarity does not merely retrieve an explanation of the procedure; it can
activate the procedure itself.

This suggests a possible consolidation path:

```text
episodes -> concepts -> validated procedure -> executable geometry
```

Experiences become interpretations, interpretations become skills, and skills
remain addressable objects rather than disappearing into an undifferentiated
weight update.

## The Model Is More Than Its Weights

In this architecture, the deployable cognitive system is:

```text
model = neural weights
      + projection adapters
      + Glamin spaces
      + geometry generation
      + active traces
      + capability manifest
```

The model supplies broad linguistic and conceptual generalization. Document
memory supplies exact records and provenance. Executable geometry supplies
procedural structure. The active trace supplies continuity.

The conversation is therefore not the system's state. It is one input and
output channel attached to a persistent process. Other clients--a command-line
interface, robot, service, or game character--could participate in the same
trajectory without carrying a transcript as the source of identity.

## Live Cognitive Geometry

Glamin geometry can be mounted, inspected, replaced, or removed while the host
process remains alive. This makes learned behavior potentially more editable
than knowledge distributed through model weights.

A safe live update uses immutable generations:

```text
active request                         control plane
     |                                      |
     +-- pinned to generation A             +-- derive generation B
                                            +-- validate contracts
                                            +-- run probes and geometry diff
                                            +-- sign and publish B

new requests use B; requests on A finish, migrate explicitly, or are cancelled
```

This could allow a system to:

- Install a new procedure without retraining or reloading the model.
- Quarantine or disable a harmful behavior immediately.
- Extract a mini-app for inspection, simulation, editing, or transfer.
- Roll back behavioral changes independently of episodic memory.
- Mount specialized capabilities temporarily.
- Fork a geometry and allow two systems to develop along different paths.

Every portable mini-app should retain its human-readable source specification,
compiled vectors, contracts, representative traces, provenance, signature, and
capability requirements. Vectors are compiled artifacts, not sufficient source
code by themselves.

## Native Runtime Shape

The proposed first implementation uses three language layers:

```text
C++17 host                  C ABI                    Fortran Glamin
------------------          ------------------       ---------------------
model inference       <-->  opaque handles      <--> async runtime
tensor coordination         fixed-width types        indexes and geometry
layer hooks                 vectors and results       traces and snapshots
capability control          status codes              contract enforcement
```

C++ is responsible for inference integration and process-level coordination.
Glamin remains implemented in Fortran and exposes a versioned C ABI through
`ISO_C_BINDING`. C++ consumes that interface with `extern "C"` and wraps its
opaque handles in RAII types.

No Fortran derived-type layout crosses the ABI. Runtime contexts, indexes,
requests, traces, and generations are represented as numeric handles. Vectors
cross as typed pointers with explicit dimensions, counts, strides, ownership,
and device location.

The initial model runtime is expected to be `llama.cpp`, pinned to a known
revision and limited to one supported model architecture. Python may be used
later for adapter training and analysis, but it is not part of the live
inference loop.

## Experiment 1

The first experiment deliberately excludes the model. It proves the native
boundary and live geometry semantics before introducing model-specific
complexity.

### Hypothesis

A C++ process can traverse Glamin geometry, replace the mounted geometry while
remaining live, and observe a deterministic behavioral change without changing
the input vector or restarting.

### Required demonstration

1. Create a Glamin runtime through the C ABI.
2. Load and pin geometry generation A.
3. Traverse it using a fixed input vector.
4. Mount geometry generation B while the process remains active.
5. Confirm existing requests remain consistent with A.
6. Confirm new requests observe B.
7. Restore A and reproduce the original result.

### Success criteria

- The C++ host does not access private Fortran data layouts.
- Generation changes are atomic from the perspective of a traversal.
- Results are deterministic for a pinned generation.
- Rollback does not require restarting the process.
- Resource ownership is explicit and leak-free.
- The trace records which generation produced each result.

### Current checkpoint

Glamin ABI version 4 and the C++ adapter now demonstrate the first five
properties with real Fortran-owned flat indexes, including sanitizer-clean
retirement and final reclamation. Persistent generations can also load from
compiler-emitted artifacts after validating the selected space contract. The
runnable demo prints generation identity but does not yet create a durable
Glamin trace, so the trace criterion remains open.

Once this works, the second experiment replaces the fixed input with a hidden
state from one layer of a small local model and uses Glamin's returned address
to select a gated residual.

That second boundary is now executable. A pinned llama.cpp revision exposes the
Qwen3 `l_out-18` tensor during graph evaluation. The adapter projects the last
token, searches a pinned Glamin generation, resolves its generation-qualified
payload, and mutates the live hidden state before later layers run. A -> B
activation changed both logits and the top next token; rollback to A reproduced
the baseline exactly without reloading model weights.

The original payload and projection remain deterministic mechanism probes. The
scaled associative experiment derives 24 real keys from three views of eight
associations at `l_out-34`. It searches every prompt-token state, automatically
selects the entity-bearing Glamin address, and applies the retrieved action to
the final answer position. Sixteen held-out positive prompts reach target rank
one, while 16 controls—including four held-out unknown-name questions—abstain
with unchanged logits. This establishes constrained natural-language
associative generalization and automatic inference-time address selection for
one synthetic relation, not broad semantic generalization. The model-agnostic
writer derives key positions without entity annotations, permits separate
surface-form neighborhoods, and binds the all-token scan policy into the
artifact. Multi-relation evaluation, trace execution, artifact authentication,
and learned gating remain subsequent work.

## Safety Boundary

Blurring the boundary between inference and memory must not blur the boundary
between information and authority.

The design therefore assumes:

- Documents are passive and cannot compile themselves into executable geometry.
- Cross-space transforms are explicit, versioned, and auditable.
- Executable mini-apps are signed and carry capability manifests.
- Consequential actions can require approval even when traversal selects them.
- Every action and geometry transition is recorded in an immutable trace.
- New geometry is probed, diffed, and mounted as a new generation.
- A known-good generation can be restored immediately.

The cognitive boundary may become difficult to locate. The security boundary
must remain obvious.

## Questions This Project Is Testing

- Can model hidden states address executable geometry reliably?
- At which layers and token boundaries is memory access most useful?
- Can a returned geometric vector alter behavior predictably without damaging
  the model's general capabilities?
- Can useful traces be consolidated into portable procedural mini-apps?
- What state must persist when cross-turn KV caches are discarded?
- How should active traces migrate when their geometry changes?
- Can geometry diffs predict behavioral changes before deployment?
- What latency is introduced by traversal and device synchronization?
- Can the system learn when to recall, when to act, and when to abstain?

## Non-Goals of the First Experiment

- Eliminating the within-response KV cache.
- Building a complete autonomous agent.
- Performing online weight updates.
- Allowing unreviewed geometry to execute external actions.
- Supporting multiple model families or hardware backends.
- Demonstrating human-like consciousness or identity.

The first objective is smaller and falsifiable: prove that persistent,
hot-swappable executable geometry can participate in a native inference-style
computation through a stable boundary.

## Long-Term Direction

If the experiment succeeds, later work can move the Glamin operation deeper
into inference, train the projection and gating components jointly, add
episodic and semantic consolidation, and allow approved mini-apps to mediate
real actions.

The long-term object would be neither a conventional model nor a conventional
memory service. It would be an editable cognitive runtime:

- Neural weights provide generalization.
- Exact records provide reality and provenance.
- Executable geometry provides accumulated procedure.
- Persistent traces provide continuity.
- Live generations provide plasticity, inspection, and rollback.

The central proposition is simple:

> Memory does not merely inform the program. Memory is where programs can live.
