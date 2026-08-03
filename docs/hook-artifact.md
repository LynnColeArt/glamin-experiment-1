# Persistent Hook Artifact

## Purpose

A model hook is only meaningful when its model identity, tensor location,
projection, Glamin address space, and returned residuals agree. Loading those
parts independently creates a dangerous mixed-generation state: a valid search
can select a payload compiled for another model or layer.

The `gx1-hook-v1` artifact binds those parts into one checked deployment unit.
`PersistentHookGenerationStore` validates the complete unit before it publishes
a generation and returns inference a single pinned hook assembled from that
same unit.

## Directory Layout

```text
hook-generation/
├── hook_manifest.txt
├── projection.f32
├── residual_labels.u64
├── residuals.f32
└── glamin/
    ├── contracts.json
    ├── vector_layout.json
    └── vectors.bin
```

All numeric payloads are little-endian. Projection values and residuals are
float32. Labels are unsigned 64-bit integers. Projection storage is row-major
`[query_dimension, hidden_dimension]`; residual storage is row-major
`[residual_count, hidden_dimension]` and corresponds positionally to the label
file.

The Glamin directory is a compiler artifact consumed through ABI version 4.
Its selected space dimension must equal the hook query dimension, and its row
count must equal the residual count. The residual labels must cover every
zero-based Glamin row address exactly once.

## Bound Contract

The canonical manifest records and hash-binds:

- model file SHA-256, architecture, llama.cpp revision, hidden width, and target
  tensor;
- query width, metric, normalization, gate, maximum retrieval distance,
  address-selection policy, and projection checksum;
- residual count plus label and residual checksums;
- Glamin space identifier plus checksums of its contracts, layout, and vectors;
- a self checksum over the sorted canonical manifest fields.

The loader requires an expected `ModelHookContract` derived from the model that
is actually active. Every model field must match exactly. It then verifies the
manifest checksum, every component checksum, exact tensor sizes, finite float
values, unique labels, and safe local file names before asking Glamin to mount
anything.

Version 1 binds both the tensor that contains the address and action rows and an
address-selection policy. The implemented policies are `last_token` and
`all_token_rows`. A pinned hook exposes that policy to the llama.cpp adapter, so
the natural-question experiment's scanner is selected by the mounted generation
rather than an unrecorded caller convention. Future learned selectors will need
their own versioned, hash-bound parameters.

For the current squared-L2 memory space, a nearest result beyond
`maximum_distance` is observable but does not resolve or inject its payload.

The checksum is an integrity and identity mechanism, not authentication. The
format has no signature or trust-chain field yet, so callers must only accept
artifacts from a trusted control plane.

## Publication and Pinning

Mounting proceeds in two stages:

1. Load and validate the entire hook artifact.
2. Mount the Glamin index, construct the generation-qualified payload ledger,
   and publish the matching C++ resource.

If either stage fails, no hook generation is published and the currently
active generation is unchanged. Activation is a separate explicit operation.
`pin_active()` returns the Glamin lease, projection, address-selection policy,
payload ledger, target tensor, and artifact checksum together. Activating or
retiring generations does not alter an existing inference pin.

The store follows Glamin ABI version 4's external-serialization requirement; a
caller must serialize mount, activation, pin acquisition, and retirement.

## Construction Boundary

`write_hook_artifact()` writes tensor files and then writes the manifest. It is
intended for building a new, private staging directory. It does not make an
in-place rewrite of an existing directory filesystem-atomic. A deployment
control plane should build into a fresh directory, synchronize it as needed,
then rename or otherwise publish that complete directory before mounting it.

## Demonstrated Behavior

The real Qwen3 demo constructs two complete temporary artifacts and mounts both
through `PersistentHookGenerationStore`. Their artifact checksums differ. An
A -> B -> A activation changes the top token from ` the` to ` for`, then
reproduces A's logits exactly while the model weights remain loaded.

The artifacts still contain deterministic experimental projection and residual
fixtures. Persistence has been established; learning a useful adapter is the
next research boundary.
