# Development Guide

## Prerequisites

- CMake 3.24 or newer
- A C++17 compiler
- Ninja or another CMake-supported build tool
- GNU Fortran 2018 compiler for the Glamin adapter
- Git submodule support
- A compatible Qwen3 GGUF file for the real-model demonstration

Initialize the pinned Glamin and llama.cpp dependencies after cloning:

```bash
git submodule update --init
```

## Configure and Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## Run the Reference Demo

```bash
./build/gx1_generation_swap_demo
```

The demo pins generation A, activates generation B, proves the old lease still
observes A, then rolls back to A. Its numeric transforms are stand-ins for
Glamin traversal.

Run the real Fortran runtime and flat-recall smoke test:

```bash
./build/gx1_glamin_runtime_smoke
```

Run the same generation swap against real Glamin-owned indexes:

```bash
./build/gx1_glamin_generation_swap_demo
```

This demo activates generation A, swaps to B, proves an old A pin remains
stable, and rolls back to A without restarting the process.

Compile Glamin's sample persistent artifact and load it through the C++ adapter:

```bash
make -C third_party/glamin spec-compile spec-embed
./build/gx1_glamin_persistent_generation_demo \
  third_party/glamin/build/specs geometry.auth
```

The load validates the selected space contract before publishing the index as
an immutable generation.

Run a real Qwen3 hidden-state intervention:

```bash
GX1_MODEL_PATH=/home/lynn/.qwench/models/Qwen_Qwen3-4B-Q4_K_M.gguf
./build/gx1_llama_hidden_state_demo "$GX1_MODEL_PATH"
```

An optional second argument selects the zero-based transformer layer. The
default is the model midpoint. The program loads weights once, runs A -> B -> A
request contexts, and requires a measurable B logit change plus exact rollback.
Each generation is a complete temporary `gx1-hook-v1` artifact mounted through
the production loader rather than an in-memory test shortcut. The verified
local model contract is recorded in [the model-hook document](model-hook.md),
and the bundle layout is described in the [artifact document](hook-artifact.md).

Run the activation-derived two-memory experiment:

```bash
./build/gx1_llama_activation_memory_experiment "$GX1_MODEL_PATH"
```

It captures query and teacher states, constructs a persistent two-row memory,
requires both target-token logits to improve to rank one, and requires two
unrelated controls to abstain with zero logit change. See the
[experiment report](activation-memory.md) for results and limitations.

## Repository Layout

```text
include/gx1/       public C++ interfaces
src/               C++ implementation
examples/          runnable demonstrations
tests/             dependency-free native tests
docs/              architecture and contracts
third_party/glamin/ pinned Glamin source and public C ABI
third_party/llama.cpp/ pinned model runtime
```

## Build Policy

- C++17 is the baseline to align with the intended `llama.cpp` integration.
- Warnings are enabled and treated as errors for project-owned targets.
- Tests avoid external frameworks during the initial spike.
- Build artifacts stay under `build/` and are ignored by Git.
- Third-party dependencies are pinned to exact revisions.

## Honesty Rule

Reference and mock implementations must identify themselves in names, output,
and documentation. A passing reference test establishes desired semantics; it
does not establish that Glamin, the C ABI, or in-model inference satisfies them.

## Current Model Boundary

The real Glamin lifecycle, synchronous flat-index wrappers, immutable generation
boundary, persistent loading, and fixed llama.cpp hidden-state hook are
implemented. Complete hook generations now persist and hash-bind the projection,
Glamin compiler artifact, active-model contract, and generation-qualified
residual payloads. The activation-derived adapter now scans every token state,
automatically selects the nearest memory address, and applies its action at the
final answer position. The reusable model-agnostic builder now derives 24 keys
from three views of eight associations without entity annotations. It passes 16
validation prompts, 16 held-out positive prompts, and 16 negative controls with
exact abstention. The artifact persists the selection policy. A separate
factorized experiment now retrieves entity and relation evidence from
independently calibrated Glamin spaces, performs an exact tuple join, and gates
a contextual residual. Its latest topology-aware probe passes 36 registered
action contexts, routes 12/12 frozen paraphrases, and exactly abstains on two
missing known-factor tuples plus 12 tuple-matched wrong intents. The selected
residual produces rank-one targets on 9/12 frozen prompts. A two-stage callback
now authorizes at `l_out-34` without mutation and applies a reviewed target state
at `l_out-35`. On its new frozen split it produced rank-one targets for all
10 authorized prompts, versus 6/10 for residuals. An entity-address follow-up
then compared variance and association-signal projections on 18 new prompts.
Association signal identified 18/18 entities and rejected 6/6 unknown-entity
controls exactly, but unchanged relation and action gates limited end-to-end
routing and rank-one recall to 9/18. A gate-local follow-up then established
form-stable centroid relation addressing at 12/12 positives and 4/4 unknown
relations. Its authorization signal reached 11/12 positives but rejected only
5/12 new wrong intents; composition reached 9/12 and failed. The next boundary
is a more explicit retrieval-intent decision, followed by persistence of the
complete factorized contract, batched scanning, and a substantially larger
negative set.

Run the factorized frozen-model probe with:

```bash
./build/gx1_llama_factorized_memory_experiment \
  /home/lynn/.qwench/models/Qwen_Qwen3-4B-Q4_K_M.gguf
```
