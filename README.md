# Glamin Experiment 1

## Executable associative memory inside transformer inference

Glamin Experiment 1 is a research prototype for studying whether persistent,
executable vector geometry can participate directly in language-model
inference. It integrates a C++17 host, a pinned `llama.cpp` inference graph, and
the Fortran-based [Glamin](https://github.com/LynnColeArt/glamin) runtime through
a versioned C ABI.

The project is deliberately narrower than a general agent-memory framework. It
does not retrieve documents into a prompt and it does not modify model weights.
Instead, a callback intercepts a named transformer tensor, derives an address
from live hidden states, searches an immutable Glamin generation, applies a
generation-qualified residual to the active tensor, and resumes the same model
evaluation.

> **Research status:** experimental. The repository establishes several small,
> reproducible mechanisms and records their failures as well as their successes.
> It is not a production memory system, a safety-certified action runtime, or
> evidence of broad factual recall.

## Abstract

Contemporary language-model memory systems usually operate outside inference:
an application retrieves text, inserts it into a context window, and invokes a
model whose persistent state remains unchanged. This project investigates a
different boundary. We ask whether an approximate-nearest-neighbor runtime can
act as a persistent, dynamically replaceable component of the model's forward
computation while retaining explicit generation, provenance, and abstention
semantics.

The implemented system projects transformer activations into Glamin-owned
spaces, retrieves generation-pinned addresses, resolves reviewed host-side
actions, and injects bounded residuals before inference continues. Successive
experiments establish: (1) live geometry activation and deterministic rollback;
(2) activation-derived associative recall with distance-gated abstention; and
(3) a factorized entity–relation–action join that separates broad semantic
evidence from authority to apply a specific memory action.

On the latest frozen-model probe, the factorized system routed all 12 unseen
retrieval prompts to the intended entity, relation, tuple, and action
neighborhood. It rejected all 12 tuple-matched wrong-intent controls with
exactly unchanged logits and also abstained on both deliberately absent tuples.
The selected residual produced the stored target as the rank-one next token on
9 of 12 prompts. This isolates the current boundary: authorization generalized
on this small split, while view-conditioned action transfer did not generalize
completely.

## Research questions

The repository is organized around four falsifiable questions:

1. Can a C++ process mount, activate, pin, retire, and restore Glamin geometry
   without restarting or invalidating in-flight readers?
2. Can a live transformer activation address an external associative memory and
   receive a result inside the same forward pass?
3. Can distance boundaries make nearest-neighbor memory fail closed on
   unrelated or unauthorized contexts?
4. Can broad entity and relation evidence be composed while specific action
   authority remains restricted to reviewed tuples and compatible intents?

The intended inference operation is:

```text
q             = project_in(h_address)
selection     = glamin_search(q, pinned_generation)
authorized_r  = reviewed_action(selection, request_state)
h_action'     = h_action + gate * authorized_r
```

The factorized probe refines `reviewed_action` into a bounded join:

```text
all token states ──> entity Glamin space ──> entity evidence + distance gate
                 └─> relation Glamin space -> relation evidence + distance gate

(entity, relation) ──> exact reviewed tuple ledger
                    └─> tuple-local action search + distance gate
                        └─> residual injection at the final token row
```

Failure at any gate leaves the hooked tensor unchanged.

## Contributions of the prototype

This repository currently provides:

- a C++ RAII wrapper over Glamin's Fortran/C runtime boundary;
- immutable geometry generations with atomic activation, stable request pins,
  retirement, and rollback;
- compiler-artifact loading with selected-space contract validation;
- a pinned `llama.cpp` tensor callback for reading and modifying live hidden
  states;
- a hash-bound `gx1-hook-v1` artifact joining the model/layer contract,
  projection, Glamin index, address-selection policy, and residual payloads;
- a model-agnostic activation-memory builder that derives keys and
  teacher-minus-query actions from grouped forward passes;
- variance and association-signal projection strategies;
- global and runtime-topology-aware gate calibration;
- factorized entity and relation search, an exact tuple ledger, and a
  tuple-local action gate; and
- executable probes that distinguish construction, development, frozen
  evaluation, negative controls, and deliberately missing tuples.

No gradient-based training is performed. “Construction” and “calibration” in
this repository refer to deriving geometry and thresholds from frozen-model
activations, not updating neural weights.

## System architecture

```text
                         control plane
                    mount / activate / retire
                               |
                               v
prompt -> pinned llama.cpp -> tensor callback --------------------+
                               |                                  |
                               v                                  |
                    activation projection                         |
                               |                                  |
                               v                                  |
                     pinned Glamin generation                     |
                               |                                  |
                               v                                  |
                 address + distance + generation                  |
                               |                                  |
                               v                                  |
                 reviewed residual/action ledger                  |
                               |                                  |
                               +---- gated tensor update ----------+
                                                                  |
                                    remaining transformer layers <-+
                                                   |
                                                   v
                                             next-token logits
```

The current real-model probes target Qwen3's penultimate transformer output,
`l_out-34`, where all prompt-token rows remain available for address discovery.
The selected action is applied only to the final row. The model, tensor name,
hidden width, projection, selection policy, Glamin space, residual labels, and
distance boundary are treated as one compatibility contract.

Glamin indexes are used for geometric search. Exact tuple membership,
generation-qualified payload resolution, and external authority remain explicit
host-side responsibilities. Proximity alone never creates a new tuple or grants
permission to perform an external effect.

## Experimental progression and results

The experiments use frozen Qwen3 4B Q4_K_M weights on CPU. Synthetic entities,
relations, and one-token targets make routing and abstention directly
measurable. Model files are not included in this repository.

| Probe | Positive result | Abstention result | Observed boundary |
| --- | --- | --- | --- |
| Live generation swap | A → B changed next-token behavior; B → A reproduced A logits exactly | Existing pins continued to observe their original generation | Establishes lifecycle semantics, not model memory |
| Activation-derived memory | 16/16 held-out prompts produced the target at rank one | 16/16 controls abstained with exactly unchanged logits | One synthetic relation; view-conditioned actions |
| Initial factorized join | 18/18 registered contexts produced rank-one targets | Two missing known-factor tuples and 12 unfamiliar contexts were exact no-ops | Unseen action phrasing remained outside the raw action neighborhood |
| Projected action probe | 4/12 frozen paraphrases produced rank-one targets | Missing tuples remained exact no-ops | Compact syntax often failed relation retrieval |
| Association-signal relation probe | Intended relation accepted on 12/12 frozen prompts | Six tuple-matched wrong intents abstained exactly | Legacy negatives constrained the action radius; 0/12 end-to-end recall |
| Topology-aware action probe | 12/12 frozen prompts routed; 9/12 produced rank-one targets | 12/12 tuple-matched wrong intents and 2/2 missing tuples were exact no-ops | Authorization generalized; selected residual transfer remained form-sensitive |

### Latest factorized probe

Six of eight possible entity–relation tuples were registered. The other two
combine known factors but have no reviewed payload, explicitly testing whether
factor composition invents a fact.

The independently calibrated boundaries were:

| Space | Maximum accepted distance | Nearest calibration negative |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Association-signal relation | `1.09326` | `1.28097` |
| Tuple-scoped projected action | `0.0823871` | `0.0974977` |

Action calibration includes two wrong-intent families—metaphor writing and
spelling comparison—for every registered tuple. These controls contain the
correct entity and relation, pass both factor gates, and reach the exact tuple
join. They are therefore genuine negatives for the action gate rather than
controls rejected earlier in the runtime topology.

Before its first run, the frozen evaluation set was fixed to two new surface
forms per tuple. All 12 selected the intended factors and action. All six
`consult` forms and three of six `associate` forms reached rank one. The three
misses ranked the target at 7, 70, and 7. No prompt, mechanism, residual, or
threshold was changed after observing the frozen result.

These measurements support a limited conclusion: the tested topology can
separate “retrieve this reviewed tuple” from same-factor non-retrieval intent.
They do not show that nearest view-conditioned residuals constitute a
form-invariant memory action.

## Calibration semantics

Nearest-neighbor search always returns a candidate, so every searchable space
requires an explicit acceptance boundary. The builder calibrates a maximum
squared-L2 distance between the hardest accepted development view and the
nearest calibration negative, retaining a configurable safety margin.

The default is global calibration. The action probe additionally supports
association-scoped calibration because runtime has already used factor evidence
and an exact tuple join to restrict the reachable action variants. Scoped
validation and negatives are compared only with keys runtime can actually
search at that stage.

This scope is a safety assertion, not a threshold optimization. Applying scoped
calibration to a globally searched space would conceal reachable competitors
and invalidate the gate.

## Reproducing the native runtime

### Prerequisites

- CMake 3.24 or newer
- a C++17 compiler
- a C compiler
- GNU Fortran with Fortran 2018 and OpenMP support
- GNU Make, Ninja, POSIX threads, and Git submodule support

Clone the complete repository:

```bash
git clone --recurse-submodules \
  https://github.com/LynnColeArt/glamin-experiment-1.git
cd glamin-experiment-1
```

The pinned dependencies are:

- Glamin commit `1ed72f9716b8401325f75c88fd63da98af946448`
- llama.cpp commit `ecd99d6a9acbc436bad085783bcd5d0b9ae9e9e9`

Configure, build, and test:

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Project-owned targets compile with `-Wall -Wextra -Wpedantic -Wconversion
-Wsign-conversion`, with warnings treated as errors by default.

Run the lifecycle demonstrations:

```bash
./build/gx1_generation_swap_demo
./build/gx1_glamin_runtime_smoke
./build/gx1_glamin_generation_swap_demo
```

Exercise persistent Glamin artifact loading:

```bash
make -C third_party/glamin spec-compile spec-embed
./build/gx1_glamin_persistent_generation_demo \
  third_party/glamin/build/specs geometry.auth
```

The reference runtime can be built without the native backends when only its
lifecycle semantics are needed:

```bash
cmake -S . -B build-reference -G Ninja \
  -DGX1_BUILD_GLAMIN_BACKEND=OFF \
  -DGX1_BUILD_LLAMA_HOOK=OFF
cmake --build build-reference
ctest --test-dir build-reference --output-on-failure
```

## Reproducing the model probes

Provide a compatible Qwen3 4B Q4_K_M GGUF file locally. The model is neither
downloaded automatically nor committed to the repository.

```bash
export GX1_MODEL_PATH=/absolute/path/to/Qwen_Qwen3-4B-Q4_K_M.gguf

./build/gx1_llama_hidden_state_demo "$GX1_MODEL_PATH"
./build/gx1_llama_activation_memory_experiment "$GX1_MODEL_PATH"
./build/gx1_llama_factorized_memory_experiment "$GX1_MODEL_PATH"
```

The factorized executable is the current end-to-end regression probe. A
successful run reports:

```text
evaluation_routes=12/12 evaluation_recall=9/12
factorized memory experiment passed
```

Quantized kernels, compiler versions, processor behavior, and future dependency
changes may affect exact floating-point distances. Reproduction should record
the model checksum, submodule revisions, compiler toolchain, CPU architecture,
and complete executable output. The checked-in values are experimental
observations, not universal thresholds.

## Repository structure

```text
include/gx1/         public C++ interfaces
src/                 runtime, builders, hooks, and artifact implementation
examples/            executable demonstrations and frozen-model probes
tests/               native unit and integration tests
docs/                architecture, safety contracts, and experiment reports
third_party/glamin/  pinned Glamin submodule
third_party/llama.cpp/ pinned llama.cpp submodule
```

The most relevant documents are:

- [Project overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Activation-derived memory report](docs/activation-memory.md)
- [Factorized memory report](docs/factorized-memory.md)
- [Hook artifact contract](docs/hook-artifact.md)
- [Generation semantics](docs/generation-semantics.md)
- [Safety model](docs/safety-model.md)
- [Development guide](docs/development.md)
- [Roadmap](docs/roadmap.md)

## Safety and governance position

Executable memory introduces risks that passive retrieval does not. The design
therefore separates geometric selection from external authority:

- generations are immutable after publication;
- each inference request pins a specific generation;
- activation affects new pins atomically;
- retirement cannot invalidate an active lease;
- payloads are resolved through generation-qualified ledgers;
- unrecognized factors, missing tuples, incompatible intents, and out-of-radius
  states fail closed;
- model and geometry compatibility is hash-bound in persisted hook artifacts;
  and
- external effects remain the responsibility of a separate capability broker,
  which is outside the present experiment.

The current implementation does not provide artifact signatures, adversarially
robust gating, untrusted multi-tenant isolation, or a complete audit service.

## Limitations and threats to validity

- The semantic tasks are synthetic and intentionally small.
- Reported model experiments use one quantized Qwen3 checkpoint and one primary
  layer on CPU.
- Targets are single tokens; multi-token generation quality is not evaluated.
- Projection is selected from frozen activations rather than learned with an
  independent training corpus.
- Calibration and evaluation corpora are small and authored by the experiment
  designer.
- The latest 12/12 routing result does not imply open-ended paraphrase coverage.
- Residual actions remain view-conditioned and fail on three frozen prompts.
- The factorized tuple/action ledger is not yet persisted as one atomic Glamin
  generation.
- Scaling behavior, index collisions, online writes, durable traces, and
  general-capability retention remain unmeasured.
- No external action execution is implemented.

## Near-term research agenda

1. Replace nearest view-conditioned residual transfer with a reviewed
   neighborhood transform or later-layer target-state action.
2. Evaluate it against a new development/frozen split without modifying the
   existing frozen prompts.
3. Persist entity space, relation space, tuple membership, action variants,
   projections, gates, and payloads as one atomically swappable generation.
4. Add generation-A/generation-B behavioral baselines and geometry diffs.
5. Measure larger memories, broader negatives, multi-token recall, and general
   capability retention.
6. Add durable trace segments and an immutable source ledger before considering
   procedural mini-apps or external capabilities.

## Citation

There is no archival release or DOI yet. If referencing the current software,
please cite the repository and the exact commit used:

```bibtex
@software{lynncoleart_glamin_experiment_1_2026,
  author  = {LynnColeArt},
  title   = {Glamin Experiment 1: Executable Associative Memory Inside
             Transformer Inference},
  year    = {2026},
  url     = {https://github.com/LynnColeArt/glamin-experiment-1},
  note    = {Experimental research software; cite the exact Git commit}
}
```

## License

No software license has been declared for this experiment. Public availability
does not by itself grant permission to copy, modify, or redistribute the work.
Third-party submodules retain their own licenses.
