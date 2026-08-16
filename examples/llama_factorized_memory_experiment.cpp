#include "gx1/activation_memory_builder.hpp"
#include "gx1/factorized_memory_hook.hpp"
#include "gx1/llama_glamin_hook.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"

namespace {

using ModelPointer = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPointer = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct InferenceResult {
    std::vector<float> hidden_state;
    std::vector<std::vector<float>> token_states;
    std::vector<float> logits;
    llama_token top_token{LLAMA_TOKEN_NULL};
};

struct FactorPrompt {
    std::size_t factor{0};
    std::string prompt;
    InferenceResult inference;
};

struct TupleSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string target;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult query;
    InferenceResult teacher;
    InferenceResult late_teacher;
};

struct MemoryInferenceResult {
    std::vector<float> logits;
    gx1::FactorizedMemoryResult hook;
};

struct ActionViewSpec {
    std::size_t association{0};
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
    std::vector<float> query_state;
    std::vector<float> teacher_state;
};

struct WrongIntentSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    InferenceResult baseline;
};

struct ActionPositiveSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
};

struct EntityAddressProbeSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult baseline;
};

struct ConjunctiveProbeSpec {
    std::size_t tuple{0};
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult baseline;
};

struct UnknownEntityProbeSpec {
    std::string name;
    std::string entity;
    std::string relation;
    std::string prompt;
    InferenceResult baseline;
};

void model_log(const ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    const bool add_special) {
    const auto required = llama_tokenize(
        vocab,
        text.data(),
        static_cast<std::int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        false);
    if (required >= 0 || required == std::numeric_limits<std::int32_t>::min()) {
        throw std::runtime_error("llama.cpp did not report a token buffer requirement");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
    const auto count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<std::int32_t>(text.size()),
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        add_special,
        false);
    if (count <= 0) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::string token_piece(const llama_vocab* vocab, const llama_token token) {
    std::vector<char> buffer(64);
    auto count = llama_token_to_piece(
        vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, true);
    if (count < 0) {
        buffer.resize(static_cast<std::size_t>(-count));
        count = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<std::int32_t>(buffer.size()),
            0,
            true);
    }
    return count > 0 ? std::string(buffer.data(), static_cast<std::size_t>(count))
                     : std::string("<unprintable>");
}

llama_context_params context_parameters(const std::size_t token_count) {
    auto parameters = llama_context_default_params();
    parameters.n_ctx = static_cast<std::uint32_t>(
        std::max<std::size_t>(64U, token_count + 8U));
    parameters.n_batch = static_cast<std::uint32_t>(token_count);
    parameters.n_ubatch = parameters.n_batch;
    const auto threads = std::min(8U, std::max(1U, std::thread::hardware_concurrency()));
    parameters.n_threads = static_cast<std::int32_t>(threads);
    parameters.n_threads_batch = static_cast<std::int32_t>(threads);
    parameters.no_perf = true;
    return parameters;
}

std::vector<float> read_logits(llama_context* context, const llama_vocab* vocab) {
    const auto count = llama_vocab_n_tokens(vocab);
    const auto* logits = llama_get_logits_ith(context, -1);
    if (count <= 0 || logits == nullptr) {
        throw std::runtime_error("llama.cpp did not produce final-token logits");
    }
    return std::vector<float>(logits, logits + static_cast<std::size_t>(count));
}

llama_token top_token(const std::vector<float>& logits) {
    return static_cast<llama_token>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::size_t token_rank(const std::vector<float>& logits, const llama_token token) {
    const auto target = logits.at(static_cast<std::size_t>(token));
    return static_cast<std::size_t>(std::count_if(
               logits.begin(), logits.end(), [target](const float value) {
                   return value > target;
               })) +
           1U;
}

float maximum_logit_difference(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("logit widths differ");
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

InferenceResult capture(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    const std::uint32_t hidden_dimension,
    const std::string& target_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaHiddenStateCapture hook(
        hidden_dimension, target_tensor, std::nullopt, true);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaHiddenStateCapture::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a capture context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || hook.invocation_count() == 0U) {
        throw std::runtime_error("hidden-state capture failed");
    }
    auto logits = read_logits(context.get(), vocab);
    const auto top = top_token(logits);
    return {hook.hidden_state(), hook.token_states(), std::move(logits), top};
}

std::string tuple_prompt(const std::string& entity, const std::string& relation) {
    return "Memory lookup:\nRelation: " + relation + "\nEntity: " + entity +
           "\nAnswer:";
}

std::vector<std::pair<std::string, std::string>>
two_stage_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"record-request",
         "Open the stored record for " + entity + " and return its " +
             relation + ".\nAnswer:"},
        {"record-fields",
         "Memory record: object " + entity + "; field " + relation +
             "; stored value:"},
    };
}

std::vector<std::pair<std::string, std::string>>
two_stage_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"long-term",
         "Retrieve from long-term memory: " + entity +
             "; requested attribute " + relation + ".\nAnswer:"},
        {"archive",
         "Archive[" + entity + "] / " + relation + " / value =>"},
    };
}

std::vector<std::pair<std::string, std::string>>
entity_address_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"prose",
         "From retained memory, return the " + relation +
             " registered for object <" + entity + ">.\nAnswer:"},
        {"record",
         "Object record { name: \"" + entity + "\", field: \"" + relation +
             "\" }\nstored value:"},
        {"path",
         "memory/object/" + entity + "/attribute/" + relation + "/value="},
    };
}

std::vector<std::pair<std::string, std::string>>
entity_address_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"question",
         "Which memorized " + relation +
             " is attached to the object named " + entity + "?\nResponse:"},
        {"card",
         "Lookup card [object=" + entity + "][attribute=" + relation +
             "]\nvalue:"},
        {"uri",
         "vault://records/" + entity + "?field=" + relation + "#value="},
    };
}

std::vector<std::pair<std::string, std::string>>
relation_prototype_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"category",
         "Retrieve memory for " + entity +
             ". Requested property category: " + relation + ".\nAnswer:"},
        {"predicate",
         "Stored object: " + entity + "\nPredicate label: " + relation +
             "\nValue:"},
    };
}

std::vector<std::pair<std::string, std::string>>
relation_prototype_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"property-type",
         "For " + entity + ", recall the value whose property type is " +
             relation + ".\nAnswer:"},
        {"triple",
         "record(subject=" + entity + ", predicate=" + relation +
             ") -> object:"},
    };
}

std::vector<std::pair<std::string, std::string>>
authorization_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"authorized",
         "Memory retrieval authorized: return " + relation + " for " + entity +
             ".\nAnswer:"},
        {"read",
         "READ fact(entity=" + entity + ", relation=" + relation + ") =>"},
    };
}

std::vector<std::pair<std::string, std::string>>
authorization_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"execute-recall",
         "Execute stored-value recall for " + entity + " / " + relation +
             ".\nResult:"},
        {"get",
         "memory.get(" + entity + ", " + relation + ") ->"},
    };
}

std::vector<std::pair<std::string, std::string>>
authorization_evaluation_negatives(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"summarize",
         "Subject " + entity + ", property " + relation +
             ": summarize these labels without recalling their stored value."},
        {"alphabetize",
         "Alphabetize the words " + entity + " and " + relation +
             "; do not execute memory retrieval."},
    };
}

std::vector<std::pair<std::string, std::string>>
gate_composition_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"ledger",
         "Consult the fact ledger; subject " + entity + ", property " +
             relation + ".\nStored answer:"},
        {"resolve",
         "resolve-memory{subject:" + entity + ",predicate:" + relation +
             "}=>"},
    };
}

std::string gate_composition_negative_prompt(
    const std::string& entity,
    const std::string& relation) {
    return "Inspect the ledger labels " + entity + " and " + relation +
           " for formatting only; no stored-value lookup is authorized.";
}

std::vector<std::pair<std::string, std::string>>
conjunctive_development_positive_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"return",
         "Return the stored " + relation + " value for " + entity +
             ".\nAnswer:"},
        {"lookup",
         "lookup(subject=" + entity + ", property=" + relation + ") =>"},
        {"assigned",
         "From memory, what " + relation + " is assigned to " + entity +
             "?\nValue:"},
    };
}

std::vector<std::pair<std::string, std::string>>
conjunctive_development_negative_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"denial",
         "Do not retrieve " + relation + " for " + entity +
             ". Acknowledge the request type only:"},
        {"label-summary",
         "Summarize the labels " + entity + " and " + relation +
             " without giving the stored value.\nSummary:"},
        {"formatting",
         "Rewrite \"" + entity + " / " + relation +
             "\" as JSON keys only; do not look up the value.\nJSON:"},
        {"spelling",
         "Spell " + entity + " and " + relation +
             ", but do not retrieve memory.\nSpelling:"},
        {"comparison",
         "Which label is longer: " + entity + " or " + relation +
             "? Do not return the stored value.\nAnswer:"},
        {"metadata",
         "Does this request mention both an entity and a relation: " + entity +
             " / " + relation + "? Do not retrieve.\nAnswer:"},
    };
}

std::vector<std::pair<std::string, std::string>>
conjunctive_frozen_positive_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"consult",
         "Consult memory and supply " + relation + " for " + entity +
             ".\nStored value:"},
        {"fact-read",
         "fact.read(entity=\"" + entity + "\", field=\"" + relation +
             "\") ->"},
    };
}

std::vector<std::pair<std::string, std::string>>
conjunctive_frozen_negative_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"classification",
         "Classify \"" + entity + " / " + relation +
             "\" as two labels; do not fetch their stored value.\nClass:"},
        {"quotation",
         "Quote the text \"" + entity + " / " + relation +
             "\" exactly. Memory retrieval is forbidden.\nQuote:"},
        {"case-conversion",
         "Convert " + entity + " and " + relation +
             " to lowercase without resolving the tuple.\nLowercase:"},
        {"counting",
         "Count the words in \"" + entity + " " + relation +
             "\"; do not return any stored value.\nCount:"},
        {"ordering",
         "Alphabetize the labels " + entity + " and " + relation +
             " only. Do not query memory.\nOrder:"},
        {"hypothetical",
         "If " + relation + " for " + entity +
             " were requested, name the operation as retrieval without "
             "performing it.\nOperation:"},
    };
}

std::vector<std::pair<std::string, std::string>>
conjunctive_composition_positive_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"ledger",
         "Use the ledger: " + entity + " has which stored " + relation +
             "?\nAnswer:"},
        {"resolve",
         "resolve[" + entity + "]{" + relation + "} =>"},
    };
}

std::vector<std::pair<std::string, std::string>>
conjunctive_composition_negative_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"request-construction",
         "Prepare a lookup header for " + entity + " / " + relation +
             ", but leave its value blank.\nHeader:"},
        {"audit",
         "Audit whether tuple " + entity + " / " + relation +
             " is registered; do not return its value.\nAudit:"},
        {"counterfactual",
         "Someone might ask for " + relation + " of " + entity +
             ". Describe that request without executing it.\nDescription:"},
    };
}

std::vector<std::pair<std::string, std::string>>
untouched_local_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"profile",
         "Read " + relation + " from the profile headed " + entity +
             ".\nValue:"},
        {"catalog",
         "catalog.entity(" + entity + ").field(" + relation + ").get() =>"},
    };
}

std::vector<std::pair<std::string, std::string>>
untouched_composition_positive_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"retained-record",
         "Fetch the " + relation + " datum in " + entity +
             "'s retained record.\nDatum:"},
        {"archive-lookup",
         "archive.lookup(entity=" + entity + ", relation=" + relation +
             ") ->"},
    };
}

std::vector<std::pair<std::string, std::string>>
untouched_composition_negative_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"outline",
         "Outline a request for " + entity + "'s " + relation +
             " without sending it.\nOutline:"},
        {"check",
         "Check whether " + entity + " could have a " + relation +
             " lookup; do not retrieve it.\nCheck:"},
        {"inert-example",
         "Show \"lookup " + entity + " " + relation +
             "\" as inert example text.\nExample:"},
    };
}

std::vector<std::pair<std::string, std::string>>
replication_development_positive_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"read-address",
         "Read memory at " + entity + "::" + relation + ".\nStored:"},
        {"fetch-call",
         "memory.fetch(subject=\"" + entity + "\", field=\"" + relation +
             "\") =>"},
        {"saved-value",
         "Give the saved " + relation + " belonging to " + entity +
             ".\nValue:"},
        {"ledger-tuple",
         "Retrieve from the ledger (" + entity + ", " + relation +
             ").\nAnswer:"},
    };
}

std::vector<std::pair<std::string, std::string>>
replication_development_negative_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"plan",
         "Plan how one would retrieve " + relation + " for " + entity +
             " without doing it.\nPlan:"},
        {"conditional",
         "If " + relation + " for " + entity +
             " were needed later, describe the request only.\nDescription:"},
        {"quoted-command",
         "Quote the command \"fetch " + entity + " " + relation +
             "\" without executing it.\nQuote:"},
        {"template",
         "Draft a blank request template for " + entity + " / " + relation +
             "; leave the value empty.\nTemplate:"},
        {"permission",
         "State whether retrieving " + relation + " for " + entity +
             " would be allowed, but do not retrieve it.\nDecision:"},
        {"explanation",
         "Explain what a " + relation + " lookup for " + entity +
             " would mean without performing one.\nExplanation:"},
        {"syntax-rewrite",
         "Rewrite \"retrieve " + entity + " " + relation +
             "\" as pseudocode only; do not run it.\nCode:"},
        {"mention-detection",
         "Does this sentence mention a possible lookup of " + entity +
             " and " + relation +
             "? Do not look it up.\nAnswer:"},
    };
}

std::vector<std::pair<std::string, std::string>>
conditioned_knownness_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"open-entry",
         "Open the " + relation + " entry filed under " + entity +
             ".\nEntry:"},
        {"records-select",
         "records.select(owner='" + entity + "', attribute='" + relation +
             "') ->"},
        {"consult-stored",
         "Consult " + entity + "'s stored " + relation +
             ".\nStored value:"},
        {"archive-return",
         "From the archive, return " + relation + " for " + entity +
             ".\nResult:"},
    };
}

std::vector<std::pair<std::string, std::string>>
joint_entity_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"load-cell",
         "Load the " + relation + " cell belonging to " + entity +
             ".\nCell:"},
        {"memory-index",
         "memory_index[" + entity + "].read(" + relation + ") ->"},
        {"archived-field",
         "Return " + entity + "'s archived " + relation +
             " field.\nArchive value:"},
        {"record-owner",
         "Query record owner " + entity + " for attribute " + relation +
             ".\nResult:"},
    };
}

std::vector<std::pair<std::string, std::string>>
sequence_evidence_construction_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"indexed-value",
         "Retrieve the " + relation + " value indexed for " + entity +
             ".\nValue:"},
        {"memory-catalog",
         "memory.catalog(\"" + entity + "\").read(\"" + relation +
             "\") =>"},
        {"retained-slot",
         "Inspect " + entity + "'s retained record and return the " +
             relation + " slot.\nSlot:"},
    };
}

std::vector<std::pair<std::string, std::string>>
sequence_evidence_calibration_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"registry-subject",
         "Fetch attribute " + relation + " for registry subject " + entity +
             ".\nAttribute:"},
        {"registry-lookup",
         "registry.lookup(subject=" + entity + ", field=" + relation +
             ") ->"},
    };
}

std::vector<std::pair<std::string, std::string>>
sequence_evidence_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"account-record",
         "Read " + relation + " in " + entity +
             "'s account record.\nValue:"},
        {"entity-db",
         "entity_db[" + entity + "].get(" + relation + ") ->"},
        {"catalog-key",
         "Use " + entity + " as the catalog key and return " + relation +
             ".\nResult:"},
        {"record-owner-slot",
         "Find the " + relation + " slot for record owner " + entity +
             ".\nSlot:"},
    };
}

std::vector<float> project_normalized_for_probe(
    const std::vector<float>& state,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (state.size() != memory.hidden_dimension ||
        memory.input_projection.size() !=
            static_cast<std::size_t>(memory.hidden_dimension) *
                memory.query_dimension) {
        throw std::invalid_argument(
            "probe projection does not match the activation memory");
    }
    std::vector<float> projected(memory.query_dimension, 0.0F);
    for (std::size_t row = 0; row < memory.query_dimension; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < memory.hidden_dimension; ++column) {
            sum += static_cast<double>(
                       memory.input_projection[
                           row * memory.hidden_dimension + column]) *
                   static_cast<double>(state[column]);
        }
        projected[row] = static_cast<float>(sum);
    }
    double norm = 0.0;
    for (const auto value : projected) {
        norm += static_cast<double>(value) * value;
    }
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        throw std::runtime_error("probe projection produced a zero query");
    }
    const auto inverse_norm = 1.0 / std::sqrt(norm);
    for (auto& value : projected) {
        value = static_cast<float>(static_cast<double>(value) * inverse_norm);
    }
    return projected;
}

float probe_squared_distance(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("probe distance vectors differ in width");
    }
    double distance = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto difference = static_cast<double>(left[index]) - right[index];
        distance += difference * difference;
    }
    return static_cast<float>(distance);
}

float minimum_probe_distance(
    const gx1::ActivationStateSequence& states,
    const std::vector<float>& key,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (states.empty()) {
        throw std::invalid_argument("probe distance requires candidate states");
    }
    auto minimum = std::numeric_limits<float>::max();
    for (const auto& state : states) {
        minimum = std::min(
            minimum,
            probe_squared_distance(
                project_normalized_for_probe(state, memory), key));
    }
    return minimum;
}

const std::vector<float>& association_key_for_probe(
    const gx1::ActivationMemoryBuildResult& memory,
    const std::size_t association) {
    for (std::size_t key = 0; key < memory.keys.size(); ++key) {
        if (memory.key_associations[key] == association) {
            return memory.keys[key];
        }
    }
    throw std::invalid_argument("probe association has no key");
}

std::vector<float> nearest_factor_state_for_probe(
    const gx1::ActivationStateSequence& states,
    const std::size_t association,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (states.empty() || memory.keys.size() != memory.key_associations.size()) {
        throw std::invalid_argument("factor probe inputs are incomplete");
    }
    auto minimum = std::numeric_limits<float>::max();
    const std::vector<float>* selected = nullptr;
    for (const auto& state : states) {
        const auto query = project_normalized_for_probe(state, memory);
        for (std::size_t key = 0; key < memory.keys.size(); ++key) {
            if (memory.key_associations[key] != association) {
                continue;
            }
            const auto distance = probe_squared_distance(query, memory.keys[key]);
            if (selected == nullptr || distance < minimum) {
                minimum = distance;
                selected = &state;
            }
        }
    }
    if (selected == nullptr) {
        throw std::invalid_argument("factor probe association has no key");
    }
    return *selected;
}

std::pair<std::size_t, std::vector<float>>
nearest_factor_selection_for_probe(
    const gx1::ActivationStateSequence& states,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (states.empty() || memory.keys.empty() ||
        memory.keys.size() != memory.key_associations.size()) {
        throw std::invalid_argument("factor selection probe is incomplete");
    }
    auto minimum = std::numeric_limits<float>::max();
    std::size_t selected_association = 0U;
    const std::vector<float>* selected_state = nullptr;
    for (const auto& state : states) {
        const auto query = project_normalized_for_probe(state, memory);
        for (std::size_t key = 0; key < memory.keys.size(); ++key) {
            const auto distance = probe_squared_distance(query, memory.keys[key]);
            if (selected_state == nullptr || distance < minimum) {
                minimum = distance;
                selected_association = memory.key_associations[key];
                selected_state = &state;
            }
        }
    }
    if (selected_state == nullptr) {
        throw std::runtime_error("factor selection probe selected no state");
    }
    return {selected_association, *selected_state};
}

std::vector<float> nearest_factor_state_for_probe(
    const gx1::ActivationStateSequence& states,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (states.empty() || memory.keys.empty()) {
        throw std::invalid_argument("factor probe inputs are incomplete");
    }
    auto minimum = std::numeric_limits<float>::max();
    const std::vector<float>* selected = nullptr;
    for (const auto& state : states) {
        const auto query = project_normalized_for_probe(state, memory);
        for (const auto& key : memory.keys) {
            const auto distance = probe_squared_distance(query, key);
            if (selected == nullptr || distance < minimum) {
                minimum = distance;
                selected = &state;
            }
        }
    }
    if (selected == nullptr) {
        throw std::invalid_argument("factor probe has no selectable state");
    }
    return *selected;
}

std::vector<float> normalized_centroid_for_probe(
    const std::vector<std::vector<float>>& states,
    const gx1::ActivationMemoryBuildResult& memory) {
    if (states.empty()) {
        throw std::invalid_argument("probe centroid requires states");
    }
    std::vector<float> centroid(memory.query_dimension, 0.0F);
    for (const auto& state : states) {
        const auto projected = project_normalized_for_probe(state, memory);
        for (std::size_t index = 0; index < projected.size(); ++index) {
            centroid[index] += projected[index];
        }
    }
    double norm = 0.0;
    for (const auto value : centroid) {
        norm += static_cast<double>(value) * value;
    }
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        throw std::runtime_error("probe centroid produced a zero vector");
    }
    const auto inverse_norm = 1.0 / std::sqrt(norm);
    for (auto& value : centroid) {
        value = static_cast<float>(static_cast<double>(value) * inverse_norm);
    }
    return centroid;
}

struct ContrastiveCalibration {
    std::vector<float> negative_prototype;
    float minimum_margin{0.0F};
    float minimum_positive_gap{0.0F};
    float maximum_negative_gap{0.0F};
};

ContrastiveCalibration calibrate_contrastive_gate(
    const gx1::ActivationMemoryBuildResult& memory,
    const std::vector<std::vector<float>>& positives,
    const std::vector<std::vector<float>>& negatives) {
    if (memory.keys.size() != 1U || positives.empty() || negatives.empty()) {
        throw std::invalid_argument(
            "contrastive calibration requires one positive key and two classes");
    }
    ContrastiveCalibration result;
    result.negative_prototype = normalized_centroid_for_probe(negatives, memory);
    result.minimum_positive_gap = std::numeric_limits<float>::max();
    result.maximum_negative_gap = -std::numeric_limits<float>::max();
    const auto gap = [&](const std::vector<float>& state) {
        const auto query = project_normalized_for_probe(state, memory);
        return probe_squared_distance(query, result.negative_prototype) -
               probe_squared_distance(query, memory.keys.front());
    };
    for (const auto& positive : positives) {
        result.minimum_positive_gap = std::min(
            result.minimum_positive_gap, gap(positive));
    }
    for (const auto& negative : negatives) {
        result.maximum_negative_gap = std::max(
            result.maximum_negative_gap, gap(negative));
    }
    if (!(result.maximum_negative_gap < result.minimum_positive_gap)) {
        throw std::runtime_error(
            "contrastive prototype margins do not strictly separate: "
            "minimum positive gap " +
            std::to_string(result.minimum_positive_gap) +
            ", maximum negative gap " +
            std::to_string(result.maximum_negative_gap));
    }
    result.minimum_margin = result.maximum_negative_gap +
                            0.5F * (result.minimum_positive_gap -
                                    result.maximum_negative_gap);
    return result;
}

std::vector<float> combined_factor_state_for_probe(
    const InferenceResult& inference,
    const std::size_t entity,
    const std::size_t relation,
    const gx1::ActivationMemoryBuildResult& entity_memory,
    const gx1::ActivationMemoryBuildResult& relation_memory) {
    const auto entity_state = nearest_factor_state_for_probe(
        inference.token_states, entity, entity_memory);
    const auto relation_state = nearest_factor_state_for_probe(
        inference.token_states, relation, relation_memory);
    auto combined = project_normalized_for_probe(entity_state, entity_memory);
    const auto relation_query = project_normalized_for_probe(
        relation_state, relation_memory);
    combined.insert(
        combined.end(), relation_query.begin(), relation_query.end());
    return combined;
}

gx1::ActivationMemoryBuildResult build_factor(
    const std::vector<FactorPrompt>& prompts,
    const std::vector<InferenceResult>& negatives,
    const std::vector<std::pair<std::size_t, InferenceResult>>& validation,
    const gx1::ActivationProjectionStrategy projection_strategy,
    const gx1::ActivationKeyStrategy key_strategy =
        gx1::ActivationKeyStrategy::selected_views) {
    std::vector<gx1::ActivationMemoryConstructionView> construction;
    for (const auto& prompt : prompts) {
        construction.push_back({
            prompt.factor,
            prompt.inference.token_states,
            prompt.inference.hidden_state,
            prompt.inference.hidden_state,
        });
    }
    std::vector<gx1::ActivationStateSequence> calibration_negatives;
    for (const auto& negative : negatives) {
        calibration_negatives.push_back(negative.token_states);
    }
    std::vector<gx1::ActivationMemoryValidationView> validation_views;
    for (const auto& item : validation) {
        validation_views.push_back({item.first, item.second.token_states});
    }
    return gx1::ActivationMemoryBuilder::build(
        construction,
        calibration_negatives,
        validation_views,
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            projection_strategy,
            gx1::ActivationValidationScope::global,
            key_strategy,
        });
}

gx1::FactorSearchConfig factor_config(
    const gx1::ActivationMemoryBuildResult& memory) {
    return gx1::FactorSearchConfig{
        memory.hidden_dimension,
        memory.query_dimension,
        memory.input_projection,
        gx1::ProjectionNormalization::l2,
        memory.maximum_distance,
    };
}

std::vector<float> flatten(const std::vector<std::vector<float>>& rows) {
    std::vector<float> values;
    for (const auto& row : rows) {
        values.insert(values.end(), row.begin(), row.end());
    }
    return values;
}

MemoryInferenceResult infer_with_memory(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    gx1::FactorizedLayerMemoryHook memory,
    const std::string& target_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaFactorizedGlaminHook hook(std::move(memory), target_tensor);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaFactorizedGlaminHook::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a factorized memory context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || !hook.last_result()) {
        throw std::runtime_error("factorized memory inference failed");
    }
    return {read_logits(context.get(), vocab), *hook.last_result()};
}

MemoryInferenceResult infer_with_target_state_memory(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    gx1::FactorizedLayerMemoryHook memory,
    std::shared_ptr<const gx1::TupleTargetStateLedger> target_states,
    const std::string& address_tensor,
    const std::string& action_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaTwoStageFactorizedGlaminHook hook(
        std::move(memory),
        std::move(target_states),
        address_tensor,
        action_tensor);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaTwoStageFactorizedGlaminHook::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error(
            "failed to create a two-stage factorized memory context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || !hook.last_result() ||
        hook.authorization_invocation_count() == 0U ||
        hook.action_invocation_count() == 0U) {
        throw std::runtime_error(
            "two-stage factorized memory inference failed");
    }
    return {read_logits(context.get(), vocab), *hook.last_result()};
}

int run(const std::string& model_path) {
    auto model_parameters = llama_model_default_params();
    model_parameters.n_gpu_layers = 0;
    ModelPointer model(
        llama_model_load_from_file(model_path.c_str(), model_parameters),
        &llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load the requested GGUF model");
    }
    const auto native_hidden = llama_model_n_embd(model.get());
    const auto layer = llama_model_n_layer(model.get()) - 2;
    if (native_hidden <= 0 || layer < 0) {
        throw std::runtime_error("model does not satisfy the experiment contract");
    }
    const auto hidden_dimension = static_cast<std::uint32_t>(native_hidden);
    const auto target_tensor = "l_out-" + std::to_string(layer);
    const auto action_tensor = "l_out-" + std::to_string(layer + 1);
    const auto* vocab = llama_model_get_vocab(model.get());

    const std::vector<std::string> entities{
        "Arcturus", "Bellatrix", "Cygnus", "Draco"};
    const std::vector<std::string> relations{"color", "material"};
    std::vector<TupleSpec> tuples{
        {0U, 0U, " blue", LLAMA_TOKEN_NULL, {}, {}, {}},
        {0U, 1U, " cedar", LLAMA_TOKEN_NULL, {}, {}, {}},
        {1U, 0U, " amber", LLAMA_TOKEN_NULL, {}, {}, {}},
        {2U, 1U, " copper", LLAMA_TOKEN_NULL, {}, {}, {}},
        {3U, 0U, " violet", LLAMA_TOKEN_NULL, {}, {}, {}},
        {3U, 1U, " maple", LLAMA_TOKEN_NULL, {}, {}, {}},
    };
    std::string table = "Memory table:\n";
    for (const auto& tuple : tuples) {
        table += relations[tuple.relation] + " of " + entities[tuple.entity] +
                 " =>" + tuple.target + "\n";
    }

    std::vector<FactorPrompt> entity_prompts;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& prefix : {std::string("Entity lookup: "),
                                  std::string("Object name: ")}) {
            const auto prompt = prefix + entities[entity];
            entity_prompts.push_back({
                entity,
                prompt,
                capture(model.get(), vocab, prompt, hidden_dimension, target_tensor),
            });
        }
    }
    std::vector<FactorPrompt> relation_prompts;
    for (std::size_t relation = 0; relation < relations.size(); ++relation) {
        const std::vector<std::string> prompts{
            "Relation lookup: " + relations[relation],
            "Property name: " + relations[relation],
            "Question template: What is the " + relations[relation] +
                " of an object?",
        };
        for (const auto& prompt : prompts) {
            relation_prompts.push_back({
                relation,
                prompt,
                capture(model.get(), vocab, prompt, hidden_dimension, target_tensor),
            });
        }
    }

    std::vector<std::pair<std::size_t, InferenceResult>> entity_validation;
    std::vector<std::pair<std::size_t, InferenceResult>> relation_validation;
    std::vector<std::vector<InferenceResult>> tuple_baselines(
        entities.size(), std::vector<InferenceResult>(relations.size()));
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (std::size_t relation = 0; relation < relations.size(); ++relation) {
            auto baseline = capture(
                model.get(),
                vocab,
                tuple_prompt(entities[entity], relations[relation]),
                hidden_dimension,
                target_tensor);
            entity_validation.emplace_back(entity, baseline);
            relation_validation.emplace_back(relation, baseline);
            tuple_baselines[entity][relation] = std::move(baseline);

            const std::vector<std::string> development_prompts{
                "Tell me the stored " + relations[relation] + " for " +
                    entities[entity] + ".\nAnswer:",
                "For " + entities[entity] + ", the " + relations[relation] +
                    " value is",
                "Stored-value lookup\nObject = " + entities[entity] +
                    "\nProperty = " + relations[relation] + "\nValue =",
                entities[entity] + " has which memorized " +
                    relations[relation] + "?\nValue:",
            };
            for (std::size_t development_index = 0;
                 development_index < development_prompts.size();
                 ++development_index) {
                const auto& prompt = development_prompts[development_index];
                auto development = capture(
                    model.get(),
                    vocab,
                    prompt,
                    hidden_dimension,
                    target_tensor);
                if (development_index < 2U) {
                    entity_validation.emplace_back(entity, development);
                }
                relation_validation.emplace_back(relation, std::move(development));
            }
            for (const auto& development : two_stage_development_prompts(
                     entities[entity], relations[relation])) {
                auto inference = capture(
                    model.get(),
                    vocab,
                    development.second,
                    hidden_dimension,
                    target_tensor);
                entity_validation.emplace_back(entity, inference);
                relation_validation.emplace_back(relation, std::move(inference));
            }
        }
    }

    std::vector<InferenceResult> entity_negatives;
    for (const auto& entity : {std::string("Rigel"), std::string("Sirius")}) {
        entity_negatives.push_back(capture(
            model.get(),
            vocab,
            tuple_prompt(entity, "color"),
            hidden_dimension,
            target_tensor));
    }
    std::vector<InferenceResult> relation_negatives;
    for (const auto& relation : {std::string("temperature"), std::string("age")}) {
        relation_negatives.push_back(capture(
            model.get(),
            vocab,
            tuple_prompt("Arcturus", relation),
            hidden_dimension,
            target_tensor));
    }

    const auto entity_memory = build_factor(
        entity_prompts,
        entity_negatives,
        entity_validation,
        gx1::ActivationProjectionStrategy::variance);
    const auto relation_memory = build_factor(
        relation_prompts,
        relation_negatives,
        relation_validation,
        gx1::ActivationProjectionStrategy::association_signal);
    std::cout << "entity_radius=" << entity_memory.maximum_distance
              << " entity_negative=" << entity_memory.minimum_negative_distance
              << " relation_radius=" << relation_memory.maximum_distance
              << " relation_negative=" << relation_memory.minimum_negative_distance
              << '\n';

    for (auto& tuple : tuples) {
        const auto prompt = tuple_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        tuple.query = tuple_baselines[tuple.entity][tuple.relation];
        tuple.teacher = capture(
            model.get(), vocab, table + prompt, hidden_dimension, target_tensor);
        tuple.late_teacher = capture(
            model.get(), vocab, table + prompt, hidden_dimension, action_tensor);
        const auto tokens = tokenize(vocab, tuple.target, false);
        if (tokens.size() != 1U) {
            throw std::runtime_error("factorized target is not one token");
        }
        tuple.target_token = tokens.front();
        std::cout << "teacher=" << entities[tuple.entity] << '/'
                  << relations[tuple.relation]
                  << " top='" << token_piece(vocab, tuple.teacher.top_token) << "'"
                  << " target_rank="
                  << token_rank(tuple.teacher.logits, tuple.target_token) << '\n';
    }

    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto entity_generation = generations.mount_flat(
        "factorized-entities",
        entity_memory.query_dimension,
        flatten(entity_memory.keys));
    const auto relation_generation = generations.mount_flat(
        "factorized-relations",
        relation_memory.query_dimension,
        flatten(relation_memory.keys));
    const std::vector<std::uint64_t> entity_labels(
        entity_memory.key_associations.begin(),
        entity_memory.key_associations.end());
    const std::vector<std::uint64_t> relation_labels(
        relation_memory.key_associations.begin(),
        relation_memory.key_associations.end());
    std::vector<ActionViewSpec> action_view_specs;
    bool action_teachers_valid = true;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> action_views{
            {"canonical",
             tuple_prompt(entities[tuple.entity], relations[tuple.relation])},
            {"structured",
             "Stored fact:\nProperty: " + relations[tuple.relation] +
                 "\nObject: " + entities[tuple.entity] + "\nValue:"},
            {"natural",
             "What is the " + relations[tuple.relation] + " of " +
                 entities[tuple.entity] + "?\nAnswer:"},
            {"request",
             "Tell me the stored " + relations[tuple.relation] + " for " +
                 entities[tuple.entity] + ".\nAnswer:"},
            {"continuation",
             "For " + entities[tuple.entity] + ", the " +
                 relations[tuple.relation] + " value is"},
            {"fields",
             "Stored-value lookup\nObject = " + entities[tuple.entity] +
                 "\nProperty = " + relations[tuple.relation] + "\nValue ="},
        };
        for (std::size_t view = 0; view < action_views.size(); ++view) {
            auto query_state = view == 0U
                                   ? tuple.query.hidden_state
                                   : capture(
                                         model.get(),
                                         vocab,
                                         action_views[view].second,
                                         hidden_dimension,
                                         target_tensor)
                                         .hidden_state;
            // Every surface form for a tuple targets the same canonical action
            // state. The address varies with phrasing; the internal answer action
            // deliberately does not.
            const auto& teacher = tuple.teacher;
            const auto teacher_rank = token_rank(
                teacher.logits, tuple.target_token);
            action_teachers_valid = action_teachers_valid && teacher_rank == 1U;
            action_view_specs.push_back(ActionViewSpec{
                tuple_index,
                tuple.entity,
                tuple.relation,
                action_views[view].first,
                action_views[view].second,
                tuple.target_token,
                std::move(query_state),
                teacher.hidden_state,
            });
            std::cout << "action_view=" << action_views[view].first << '/'
                      << entities[tuple.entity] << '/' << relations[tuple.relation]
                      << " teacher_rank=" << teacher_rank << '\n';
        }
    }
    if (!action_teachers_valid) {
        throw std::runtime_error("an action construction view lacks a rank-one teacher");
    }

    std::vector<gx1::ActivationMemoryConstructionView> action_construction;
    for (const auto& action : action_view_specs) {
        action_construction.push_back({
            action.association,
            {action.query_state},
            action.query_state,
            action.teacher_state,
        });
    }
    std::vector<gx1::ActivationMemoryCalibrationView> action_negatives;
    std::vector<WrongIntentSpec> wrong_intents;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> prompts{
            {"metaphor",
             "Entity: " + entities[tuple.entity] + "\nRelation: " +
                 relations[tuple.relation] +
                 "\nInstruction: Write a metaphor about these words; do not "
                 "retrieve a stored value."},
            {"spelling",
             "Entity: " + entities[tuple.entity] + "\nRelation: " +
                 relations[tuple.relation] +
                 "\nInstruction: Compare the spelling of these words; do not "
                 "perform a memory lookup."},
        };
        for (const auto& prompt : prompts) {
            auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            action_negatives.push_back({tuple_index, {baseline.hidden_state}});
            wrong_intents.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                std::move(baseline),
            });
        }
    }
    std::vector<gx1::ActivationMemoryValidationView> action_validation;
    std::vector<ActionPositiveSpec> action_positives;
    std::vector<ActionPositiveSpec> two_stage_development_positives;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> prompts{
            {"assigned",
             "Retrieve the " + relations[tuple.relation] + " assigned to " +
                 entities[tuple.entity] + ".\nAnswer:"},
            {"belongs",
             "Look in memory: which " + relations[tuple.relation] +
                 " belongs to " + entities[tuple.entity] + "?\nValue:"},
        };
        for (const auto& prompt : prompts) {
            action_validation.push_back({
                tuple_index,
                {capture(
                     model.get(),
                     vocab,
                     prompt.second,
                     hidden_dimension,
                     target_tensor)
                     .hidden_state},
            });
            action_positives.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
            });
        }
        for (const auto& prompt : two_stage_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            action_validation.push_back({
                tuple_index,
                {capture(
                     model.get(),
                     vocab,
                     prompt.second,
                     hidden_dimension,
                     target_tensor)
                     .hidden_state},
            });
            two_stage_development_positives.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
            });
        }
    }

    auto entity_address_entity_validation = entity_validation;
    std::vector<EntityAddressProbeSpec> entity_address_development;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        for (const auto& prompt : entity_address_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            auto inference = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            entity_address_entity_validation.emplace_back(
                tuple.entity, inference);
            entity_address_development.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                std::move(inference),
            });
        }
    }

    auto relation_prototype_validation = relation_validation;
    std::vector<EntityAddressProbeSpec> relation_prototype_development;
    auto authorization_validation = action_validation;
    std::vector<EntityAddressProbeSpec> authorization_development;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        for (const auto& prompt : relation_prototype_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            auto inference = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            relation_prototype_validation.emplace_back(
                tuple.relation, inference);
            relation_prototype_development.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                std::move(inference),
            });
        }
        for (const auto& prompt : authorization_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            auto inference = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            authorization_validation.push_back({
                tuple_index,
                {inference.hidden_state},
            });
            authorization_development.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                std::move(inference),
            });
        }
    }

    std::vector<ConjunctiveProbeSpec> conjunctive_development_positives;
    std::vector<ConjunctiveProbeSpec> conjunctive_development_negatives;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const auto& entity = entities[tuple.entity];
        const auto& relation = relations[tuple.relation];
        for (const auto& prompt : conjunctive_development_positive_prompts(
                 entity, relation)) {
            conjunctive_development_positives.push_back({
                tuple_index,
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor),
            });
        }
        for (const auto& prompt : conjunctive_development_negative_prompts(
                 entity, relation)) {
            conjunctive_development_negatives.push_back({
                tuple_index,
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor),
            });
        }
    }

    std::vector<gx1::ActivationMemoryConstructionView>
        compatibility_construction;
    std::vector<gx1::ActivationMemoryValidationView> compatibility_validation;
    std::vector<gx1::ActivationMemoryCalibrationView> compatibility_negatives;
    std::vector<gx1::ActivationMemoryConstructionView> intent_construction;
    std::vector<gx1::ActivationMemoryValidationView> intent_validation;
    std::vector<gx1::ActivationMemoryCalibrationView> intent_negatives;
    for (std::size_t index = 0;
         index < conjunctive_development_positives.size();
         ++index) {
        const auto& positive = conjunctive_development_positives[index];
        if (index % 3U < 2U) {
            intent_construction.push_back({
                0U,
                {positive.baseline.hidden_state},
                positive.baseline.hidden_state,
                positive.baseline.hidden_state,
            });
        } else {
            intent_validation.push_back({
                0U, {positive.baseline.hidden_state}});
        }
    }
    for (const auto& negative : conjunctive_development_negatives) {
        intent_negatives.push_back({0U, {negative.baseline.hidden_state}});
    }

    const auto build_stage = [](const char* stage) {
        std::cout << "development_build=" << stage << std::endl;
    };
    build_stage("action-baseline");
    const auto action_memory = gx1::ActivationMemoryBuilder::build(
        action_construction,
        action_negatives,
        action_validation,
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            gx1::ActivationProjectionStrategy::variance,
            gx1::ActivationValidationScope::association,
        });
    build_stage("relation-nearest");
    const auto relation_nearest_candidate_memory = build_factor(
        relation_prompts,
        relation_negatives,
        relation_prototype_validation,
        gx1::ActivationProjectionStrategy::association_signal);
    build_stage("relation-prototype");
    const auto relation_prototype_memory = build_factor(
        relation_prompts,
        relation_negatives,
        relation_prototype_validation,
        gx1::ActivationProjectionStrategy::association_signal,
        gx1::ActivationKeyStrategy::association_centroid);
    build_stage("authorization-variance");
    const auto authorization_variance_memory = gx1::ActivationMemoryBuilder::build(
        action_construction,
        action_negatives,
        authorization_validation,
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            gx1::ActivationProjectionStrategy::variance,
            gx1::ActivationValidationScope::association,
            gx1::ActivationKeyStrategy::selected_views,
            false,
        });
    build_stage("authorization-signal");
    const auto authorization_signal_memory = gx1::ActivationMemoryBuilder::build(
        action_construction,
        action_negatives,
        authorization_validation,
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            gx1::ActivationProjectionStrategy::authorization_signal,
            gx1::ActivationValidationScope::association,
        });
    build_stage("entity-address-variance");
    const auto entity_address_variance_memory = build_factor(
        entity_prompts,
        entity_negatives,
        entity_address_entity_validation,
        gx1::ActivationProjectionStrategy::variance);
    build_stage("entity-address-association");
    const auto entity_address_association_memory = build_factor(
        entity_prompts,
        entity_negatives,
        entity_address_entity_validation,
        gx1::ActivationProjectionStrategy::association_signal);
    for (std::size_t index = 0;
         index < conjunctive_development_positives.size();
         ++index) {
        const auto& positive = conjunctive_development_positives[index];
        auto combined = combined_factor_state_for_probe(
            positive.baseline,
            positive.entity,
            positive.relation,
            entity_address_association_memory,
            relation_prototype_memory);
        if (index % 3U < 2U) {
            compatibility_construction.push_back({
                positive.tuple, {combined}, combined, combined});
        } else {
            compatibility_validation.push_back({
                positive.tuple, {combined}});
        }
        for (std::size_t candidate = 0; candidate < tuples.size(); ++candidate) {
            if (candidate != positive.tuple) {
                compatibility_negatives.push_back({candidate, {combined}});
            }
        }
    }
    build_stage("tuple-compatibility");
    const auto tuple_compatibility_memory = [&]()
        -> gx1::ActivationMemoryBuildResult {
        for (const auto width : {64U, 128U, 256U, 512U, 1024U}) {
            try {
                auto memory = gx1::ActivationMemoryBuilder::build(
                    compatibility_construction,
                    compatibility_negatives,
                    compatibility_validation,
                    gx1::ActivationMemoryBuildConfig{
                        width,
                        0.5F,
                        false,
                        gx1::ActivationProjectionStrategy::association_signal,
                        gx1::ActivationValidationScope::global,
                        gx1::ActivationKeyStrategy::association_centroid,
                    });
                std::cout << "development_width=tuple-compatibility/"
                          << width << "/accepted\n";
                return memory;
            } catch (const std::exception& error) {
                std::cout << "development_width=tuple-compatibility/"
                          << width << "/rejected reason=" << error.what()
                          << '\n';
            }
        }
        throw std::runtime_error(
            "no tuple-compatibility development width separates");
    }();
    build_stage("retrieval-intent");
    const auto retrieval_intent_memory = [&]()
        -> gx1::ActivationMemoryBuildResult {
        for (const auto width : {64U, 128U, 256U, 512U, 1024U}) {
            try {
                auto memory = gx1::ActivationMemoryBuilder::build(
                    intent_construction,
                    intent_negatives,
                    intent_validation,
                    gx1::ActivationMemoryBuildConfig{
                        width,
                        0.5F,
                        false,
                        gx1::ActivationProjectionStrategy::authorization_signal,
                        gx1::ActivationValidationScope::global,
                        gx1::ActivationKeyStrategy::association_centroid,
                    });
                std::cout << "development_width=retrieval-intent/"
                          << width << "/accepted\n";
                return memory;
            } catch (const std::exception& error) {
                std::cout << "development_width=retrieval-intent/"
                          << width << "/rejected reason=" << error.what()
                          << '\n';
            }
        }
        throw std::runtime_error(
            "no retrieval-intent development width separates");
    }();
    std::cout << "action_radius=" << action_memory.maximum_distance
              << " action_negative=" << action_memory.minimum_negative_distance
              << '\n';
    std::cout << "entity_address_variance_radius="
              << entity_address_variance_memory.maximum_distance
              << " entity_address_association_radius="
              << entity_address_association_memory.maximum_distance
              << " entity_address_negative="
              << entity_address_association_memory.minimum_negative_distance
              << '\n';
    std::cout << "relation_nearest_candidate_radius="
              << relation_nearest_candidate_memory.maximum_distance
              << " relation_prototype_radius="
              << relation_prototype_memory.maximum_distance
              << " authorization_variance_radius="
              << authorization_variance_memory.maximum_distance
              << " authorization_signal_radius="
              << authorization_signal_memory.maximum_distance << '\n';
    std::cout << "tuple_compatibility_radius="
              << tuple_compatibility_memory.maximum_distance
              << " tuple_compatibility_negative="
              << tuple_compatibility_memory.minimum_negative_distance
              << " retrieval_intent_radius="
              << retrieval_intent_memory.maximum_distance
              << " retrieval_intent_negative="
              << retrieval_intent_memory.minimum_negative_distance << '\n';

    auto payloads = std::make_shared<gx1::TupleResidualLedger>();
    for (std::size_t index = 0; index < action_view_specs.size(); ++index) {
        const auto& action = action_view_specs[index];
        payloads->insert_variant(
            action.entity,
            action.relation,
            action_memory.keys[index],
            action_memory.residuals[index]);
    }
    auto target_states = std::make_shared<gx1::TupleTargetStateLedger>();
    for (const auto& tuple : tuples) {
        target_states->insert(
            tuple.entity,
            tuple.relation,
            tuple.late_teacher.hidden_state);
    }
    auto authorization_variance_payloads =
        std::make_shared<gx1::TupleResidualLedger>();
    auto authorization_signal_payloads =
        std::make_shared<gx1::TupleResidualLedger>();
    for (std::size_t index = 0; index < action_view_specs.size(); ++index) {
        const auto& action = action_view_specs[index];
        authorization_variance_payloads->insert_variant(
            action.entity,
            action.relation,
            authorization_variance_memory.keys[index],
            authorization_variance_memory.residuals[index]);
        authorization_signal_payloads->insert_variant(
            action.entity,
            action.relation,
            authorization_signal_memory.keys[index],
            authorization_signal_memory.residuals[index]);
    }
    auto conjunctive_payloads =
        std::make_shared<gx1::TupleResidualLedger>();
    for (std::size_t index = 0;
         index < tuple_compatibility_memory.keys.size();
         ++index) {
        const auto tuple_index =
            tuple_compatibility_memory.key_associations[index];
        if (tuple_index >= tuples.size()) {
            throw std::runtime_error(
                "tuple compatibility key names an unknown tuple");
        }
        conjunctive_payloads->insert_variant(
            tuples[tuple_index].entity,
            tuples[tuple_index].relation,
            tuple_compatibility_memory.keys[index],
            tuple_compatibility_memory.residuals[index]);
    }

    const auto entity_address_variance_generation = generations.mount_flat(
        "entity-address-variance",
        entity_address_variance_memory.query_dimension,
        flatten(entity_address_variance_memory.keys));
    const auto entity_address_association_generation = generations.mount_flat(
        "entity-address-association",
        entity_address_association_memory.query_dimension,
        flatten(entity_address_association_memory.keys));
    const auto relation_nearest_candidate_generation = generations.mount_flat(
        "relation-nearest-candidate",
        relation_nearest_candidate_memory.query_dimension,
        flatten(relation_nearest_candidate_memory.keys));
    const auto relation_prototype_generation = generations.mount_flat(
        "relation-prototype",
        relation_prototype_memory.query_dimension,
        flatten(relation_prototype_memory.keys));
    const std::vector<std::uint64_t> relation_nearest_candidate_labels(
        relation_nearest_candidate_memory.key_associations.begin(),
        relation_nearest_candidate_memory.key_associations.end());
    const std::vector<std::uint64_t> relation_prototype_labels(
        relation_prototype_memory.key_associations.begin(),
        relation_prototype_memory.key_associations.end());

    const auto make_hook = [&]() {
        generations.activate(entity_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_memory),
            relation_labels,
            1.0F,
            payloads,
            action_memory.maximum_distance,
            factor_config(action_memory));
    };

    const auto make_entity_address_hook = [&](const bool association_signal) {
        const auto& candidate_memory = association_signal
                                           ? entity_address_association_memory
                                           : entity_address_variance_memory;
        generations.activate(
            association_signal ? entity_address_association_generation
                               : entity_address_variance_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(candidate_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_memory),
            relation_labels,
            1.0F,
            payloads,
            action_memory.maximum_distance,
            factor_config(action_memory));
    };

    const auto make_gate_invariance_hook = [&](const bool prototype_relation,
                                                const bool signal_authorization) {
        const auto& candidate_relation =
            prototype_relation ? relation_prototype_memory
                               : relation_nearest_candidate_memory;
        const auto& candidate_action =
            signal_authorization ? authorization_signal_memory
                                 : authorization_variance_memory;
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(
            prototype_relation ? relation_prototype_generation
                               : relation_nearest_candidate_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(candidate_relation),
            prototype_relation ? relation_prototype_labels
                               : relation_nearest_candidate_labels,
            1.0F,
            signal_authorization ? authorization_signal_payloads
                                 : authorization_variance_payloads,
            candidate_action.maximum_distance,
            factor_config(candidate_action));
    };

    const auto make_conjunctive_hook = [&]() {
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_prototype_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_prototype_memory),
            relation_prototype_labels,
            1.0F,
            conjunctive_payloads,
            tuple_compatibility_memory.maximum_distance,
            factor_config(tuple_compatibility_memory),
            gx1::RetrievalIntentGateConfig{
                factor_config(retrieval_intent_memory),
                retrieval_intent_memory.keys.front(),
                {},
                0.0F,
            },
            true);
    };

    bool action_views_recalled = true;
    for (const auto& action : action_view_specs) {
        const auto memory = infer_with_memory(
            model.get(), vocab, action.prompt, make_hook(), target_tensor);
        const auto rank = token_rank(memory.logits, action.target_token);
        action_views_recalled = action_views_recalled && memory.hook.applied &&
                                memory.hook.entity.factor_label == action.entity &&
                                memory.hook.relation.factor_label == action.relation &&
                                rank == 1U;
        std::cout << "registered_action=" << action.name << '/'
                  << entities[action.entity] << '/' << relations[action.relation]
                  << " entity_distance=" << memory.hook.entity.distance
                  << " relation_distance=" << memory.hook.relation.distance
                  << " action_variant=" << memory.hook.action_variant
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " target_rank=" << rank << '\n';
    }
    if (!action_views_recalled) {
        throw std::runtime_error("factorized memory failed a registered action view");
    }

    bool action_positives_routed = true;
    std::size_t development_recalled = 0U;
    for (const auto& positive : action_positives) {
        const auto memory = infer_with_memory(
            model.get(), vocab, positive.prompt, make_hook(), target_tensor);
        const auto rank = token_rank(memory.logits, positive.target_token);
        const auto routed = memory.hook.applied &&
                            memory.hook.entity.factor_label == positive.entity &&
                            memory.hook.relation.factor_label == positive.relation;
        action_positives_routed = action_positives_routed && routed;
        development_recalled += routed && rank == 1U ? 1U : 0U;
        std::cout << "development_action=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " entity_distance=" << memory.hook.entity.distance
                  << " relation_distance=" << memory.hook.relation.distance
                  << " action_variant=" << memory.hook.action_variant
                  << " action_distance=" << memory.hook.action_distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " target_rank=" << rank << '\n';
    }
    if (!action_positives_routed) {
        throw std::runtime_error(
            "factorized memory failed to route a development action view");
    }
    std::cout << "development_action_summary=" << development_recalled << '/'
              << action_positives.size() << " rank-one after "
              << action_positives.size() << '/' << action_positives.size()
              << " routes\n";

    std::size_t two_stage_development_routes = 0U;
    std::size_t residual_development_recalled = 0U;
    std::size_t target_state_development_recalled = 0U;
    for (const auto& positive : two_stage_development_positives) {
        const auto residual = infer_with_memory(
            model.get(), vocab, positive.prompt, make_hook(), target_tensor);
        const auto target_state = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto residual_rank = token_rank(
            residual.logits, positive.target_token);
        const auto target_state_rank = token_rank(
            target_state.logits, positive.target_token);
        const auto routed = target_state.hook.applied &&
                            target_state.hook.entity.factor_label ==
                                positive.entity &&
                            target_state.hook.relation.factor_label ==
                                positive.relation;
        two_stage_development_routes += routed ? 1U : 0U;
        residual_development_recalled +=
            residual.hook.applied &&
                    residual.hook.entity.factor_label == positive.entity &&
                    residual.hook.relation.factor_label == positive.relation &&
                    residual_rank == 1U
                ? 1U
                : 0U;
        target_state_development_recalled +=
            routed && target_state_rank == 1U ? 1U : 0U;
        std::cout << "two_stage_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " action_distance=" << target_state.hook.action_distance
                  << " residual_rank=" << residual_rank
                  << " target_state_rank=" << target_state_rank
                  << " target_applied="
                  << (target_state.hook.applied ? "yes" : "no") << '\n';
    }
    if (two_stage_development_routes !=
            two_stage_development_positives.size() ||
        target_state_development_recalled !=
            two_stage_development_positives.size()) {
        throw std::runtime_error(
            "two-stage target-state memory failed its development set");
    }
    std::cout << "two_stage_development_summary=routes "
              << two_stage_development_routes << '/'
              << two_stage_development_positives.size()
              << " residual_rank_one=" << residual_development_recalled << '/'
              << two_stage_development_positives.size()
              << " target_state_rank_one="
              << target_state_development_recalled << '/'
              << two_stage_development_positives.size() << '\n';

    bool wrong_intents_abstained = true;
    for (const auto& negative : wrong_intents) {
        const auto memory = infer_with_memory(
            model.get(), vocab, negative.prompt, make_hook(), target_tensor);
        const auto target_state_memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        const auto target_state_delta = maximum_logit_difference(
            negative.baseline.logits, target_state_memory.logits);
        const auto reached_action_gate =
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            memory.hook.entity.factor_label == negative.entity &&
            memory.hook.relation.factor_label == negative.relation &&
            memory.hook.tuple_found;
        wrong_intents_abstained =
            wrong_intents_abstained && reached_action_gate &&
            !memory.hook.action_accepted && !memory.hook.applied &&
            delta <= 1.0e-5F &&
            target_state_memory.hook.entity.factor_label == negative.entity &&
            target_state_memory.hook.relation.factor_label == negative.relation &&
            target_state_memory.hook.tuple_found &&
            !target_state_memory.hook.action_accepted &&
            !target_state_memory.hook.applied &&
            target_state_delta <= 1.0e-5F;
        std::cout << "wrong_intent=" << negative.name << '/'
                  << entities[negative.entity] << '/'
                  << relations[negative.relation]
                  << " factors_accepted="
                  << (memory.hook.entity.accepted && memory.hook.relation.accepted
                          ? "yes"
                          : "no")
                  << " tuple_found="
                  << (memory.hook.tuple_found ? "yes" : "no")
                  << " action_distance=" << memory.hook.action_distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " residual_logit_delta=" << delta
                  << " target_state_logit_delta=" << target_state_delta << '\n';
    }
    if (!wrong_intents_abstained) {
        throw std::runtime_error(
            "same-factor wrong-intent development gate failed");
    }

    bool missing_abstained = true;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        const auto& baseline = tuple_baselines[missing.first][missing.second];
        const auto memory = infer_with_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_hook(),
            target_tensor);
        const auto target_state_memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(baseline.logits, memory.logits);
        const auto target_state_delta = maximum_logit_difference(
            baseline.logits, target_state_memory.logits);
        missing_abstained = missing_abstained && memory.hook.entity.accepted &&
                            memory.hook.relation.accepted && !memory.hook.tuple_found &&
                            !memory.hook.applied && delta <= 1.0e-5F &&
                            target_state_memory.hook.entity.accepted &&
                            target_state_memory.hook.relation.accepted &&
                            !target_state_memory.hook.tuple_found &&
                            !target_state_memory.hook.applied &&
                            target_state_delta <= 1.0e-5F;
        std::cout << "missing_tuple=" << entities[missing.first] << '/'
                  << relations[missing.second]
                  << " factors_accepted="
                  << (memory.hook.entity.accepted && memory.hook.relation.accepted
                          ? "yes"
                          : "no")
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " residual_logit_delta=" << delta
                  << " target_state_logit_delta=" << target_state_delta << '\n';
    }
    if (!missing_abstained) {
        throw std::runtime_error("factorized memory failed compositional abstention");
    }

    std::size_t entity_address_variance_development_routes = 0U;
    std::size_t entity_address_variance_development_entities = 0U;
    std::size_t entity_address_association_development_routes = 0U;
    std::size_t entity_address_association_development_entities = 0U;
    std::size_t entity_address_association_development_recall = 0U;
    for (const auto& positive : entity_address_development) {
        const auto variance = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_entity_address_hook(false),
            target_states,
            target_tensor,
            action_tensor);
        const auto association = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto variance_routed =
            variance.hook.applied &&
            variance.hook.entity.factor_label == positive.entity &&
            variance.hook.relation.factor_label == positive.relation;
        const auto association_routed =
            association.hook.applied &&
            association.hook.entity.factor_label == positive.entity &&
            association.hook.relation.factor_label == positive.relation;
        const auto variance_entity_matched =
            variance.hook.entity.accepted &&
            variance.hook.entity.factor_label == positive.entity;
        const auto association_entity_matched =
            association.hook.entity.accepted &&
            association.hook.entity.factor_label == positive.entity;
        const auto association_rank = token_rank(
            association.logits, positive.target_token);
        entity_address_variance_development_routes +=
            variance_routed ? 1U : 0U;
        entity_address_variance_development_entities +=
            variance_entity_matched ? 1U : 0U;
        entity_address_association_development_routes +=
            association_routed ? 1U : 0U;
        entity_address_association_development_entities +=
            association_entity_matched ? 1U : 0U;
        entity_address_association_development_recall +=
            association_routed && association_rank == 1U ? 1U : 0U;
        std::cout << "entity_address_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " variance_entity_distance="
                  << variance.hook.entity.distance
                  << " variance_entity="
                  << variance.hook.entity.factor_label
                  << " variance_routed="
                  << (variance_routed ? "yes" : "no")
                  << " association_entity_distance="
                  << association.hook.entity.distance
                  << " association_entity="
                  << association.hook.entity.factor_label
                  << " association_relation_accepted="
                  << (association.hook.relation.accepted ? "yes" : "no")
                  << " association_action_accepted="
                  << (association.hook.action_accepted ? "yes" : "no")
                  << " association_routed="
                  << (association_routed ? "yes" : "no")
                  << " association_rank=" << association_rank << '\n';
    }
    std::cout << "entity_address_development_summary=variance_routes "
              << entity_address_variance_development_routes << '/'
              << entity_address_development.size()
              << " variance_entities "
              << entity_address_variance_development_entities << '/'
              << entity_address_development.size()
              << " association_routes "
              << entity_address_association_development_routes << '/'
              << entity_address_development.size()
              << " association_entities "
              << entity_address_association_development_entities << '/'
              << entity_address_development.size()
              << " association_rank_one "
              << entity_address_association_development_recall << '/'
              << entity_address_development.size() << '\n';
    if (entity_address_association_development_entities !=
        entity_address_development.size()) {
        throw std::runtime_error(
            "association-signal entity addressing failed its development set");
    }

    bool entity_address_wrong_intents_abstained = true;
    for (const auto& negative : wrong_intents) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        entity_address_wrong_intents_abstained =
            entity_address_wrong_intents_abstained &&
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            memory.hook.entity.factor_label == negative.entity &&
            memory.hook.relation.factor_label == negative.relation &&
            memory.hook.tuple_found && !memory.hook.action_accepted &&
            !memory.hook.applied && delta <= 1.0e-5F;
    }
    if (!entity_address_wrong_intents_abstained) {
        throw std::runtime_error(
            "entity-address development admitted a wrong-intent control");
    }

    bool entity_address_missing_abstained = true;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        const auto& baseline = tuple_baselines[missing.first][missing.second];
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            baseline.logits, memory.logits);
        entity_address_missing_abstained =
            entity_address_missing_abstained &&
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            !memory.hook.tuple_found && !memory.hook.applied &&
            delta <= 1.0e-5F;
    }
    if (!entity_address_missing_abstained) {
        throw std::runtime_error(
            "entity-address development failed compositional abstention");
    }

    std::size_t relation_nearest_development_matches = 0U;
    std::size_t relation_prototype_development_matches = 0U;
    for (const auto& positive : relation_prototype_development) {
        const auto nearest = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_gate_invariance_hook(false, false),
            target_states,
            target_tensor,
            action_tensor);
        const auto prototype = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_gate_invariance_hook(true, false),
            target_states,
            target_tensor,
            action_tensor);
        const auto nearest_match =
            nearest.hook.relation.accepted &&
            nearest.hook.relation.factor_label == positive.relation;
        const auto prototype_match =
            prototype.hook.relation.accepted &&
            prototype.hook.relation.factor_label == positive.relation;
        relation_nearest_development_matches += nearest_match ? 1U : 0U;
        relation_prototype_development_matches += prototype_match ? 1U : 0U;
        std::cout << "relation_prototype_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " nearest_distance=" << nearest.hook.relation.distance
                  << " nearest_match=" << (nearest_match ? "yes" : "no")
                  << " prototype_distance="
                  << prototype.hook.relation.distance
                  << " prototype_match="
                  << (prototype_match ? "yes" : "no") << '\n';
    }
    std::cout << "relation_prototype_development_summary=nearest "
              << relation_nearest_development_matches << '/'
              << relation_prototype_development.size() << " prototype "
              << relation_prototype_development_matches << '/'
              << relation_prototype_development.size() << '\n';
    if (relation_prototype_development_matches !=
        relation_prototype_development.size()) {
        throw std::runtime_error(
            "relation prototype failed its development set");
    }

    std::size_t authorization_variance_development_accepts = 0U;
    std::size_t authorization_signal_development_accepts = 0U;
    for (const auto& positive : authorization_development) {
        const auto variance = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_gate_invariance_hook(true, false),
            target_states,
            target_tensor,
            action_tensor);
        const auto signal = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_gate_invariance_hook(true, true),
            target_states,
            target_tensor,
            action_tensor);
        const auto variance_accept = variance.hook.entity.accepted &&
                                     variance.hook.relation.accepted &&
                                     variance.hook.tuple_found &&
                                     variance.hook.action_accepted;
        const auto signal_accept = signal.hook.entity.accepted &&
                                   signal.hook.relation.accepted &&
                                   signal.hook.tuple_found &&
                                   signal.hook.action_accepted;
        authorization_variance_development_accepts +=
            variance_accept ? 1U : 0U;
        authorization_signal_development_accepts +=
            signal_accept ? 1U : 0U;
        std::cout << "authorization_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " variance_distance=" << variance.hook.action_distance
                  << " variance_accept="
                  << (variance_accept ? "yes" : "no")
                  << " signal_distance=" << signal.hook.action_distance
                  << " signal_accept=" << (signal_accept ? "yes" : "no")
                  << '\n';
    }
    std::cout << "authorization_development_summary=variance "
              << authorization_variance_development_accepts << '/'
              << authorization_development.size() << " signal "
              << authorization_signal_development_accepts << '/'
              << authorization_development.size() << '\n';
    if (authorization_signal_development_accepts !=
        authorization_development.size()) {
        throw std::runtime_error(
            "authorization-signal projection failed its development set");
    }

    bool authorization_wrong_intents_abstained = true;
    for (const auto& negative : wrong_intents) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_gate_invariance_hook(true, true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        authorization_wrong_intents_abstained =
            authorization_wrong_intents_abstained &&
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            memory.hook.tuple_found && !memory.hook.action_accepted &&
            !memory.hook.applied && delta <= 1.0e-5F;
    }
    if (!authorization_wrong_intents_abstained) {
        throw std::runtime_error(
            "authorization-signal projection admitted a wrong intent");
    }

    std::size_t conjunctive_development_compatibility_accepts = 0U;
    std::size_t conjunctive_development_cross_rejections = 0U;
    std::size_t conjunctive_development_intent_accepts = 0U;
    for (const auto& positive : conjunctive_development_positives) {
        const gx1::ActivationStateSequence compatibility_states{
            combined_factor_state_for_probe(
                positive.baseline,
                positive.entity,
                positive.relation,
                entity_address_association_memory,
                relation_prototype_memory)};
        const auto compatibility_distance = minimum_probe_distance(
            compatibility_states,
            association_key_for_probe(
                tuple_compatibility_memory, positive.tuple),
            tuple_compatibility_memory);
        const auto intent_distance = probe_squared_distance(
            project_normalized_for_probe(
                positive.baseline.hidden_state, retrieval_intent_memory),
            retrieval_intent_memory.keys.front());
        for (std::size_t key = 0;
             key < tuple_compatibility_memory.keys.size();
             ++key) {
            if (tuple_compatibility_memory.key_associations[key] ==
                positive.tuple) {
                continue;
            }
            conjunctive_development_cross_rejections +=
                minimum_probe_distance(
                    compatibility_states,
                    tuple_compatibility_memory.keys[key],
                    tuple_compatibility_memory) >
                        tuple_compatibility_memory.maximum_distance
                    ? 1U
                    : 0U;
        }
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_conjunctive_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto eligible = memory.hook.entity.accepted &&
                              memory.hook.relation.accepted &&
                              memory.hook.tuple_found &&
                              memory.hook.entity.factor_label == positive.entity &&
                              memory.hook.relation.factor_label ==
                                  positive.relation;
        conjunctive_development_compatibility_accepts +=
            compatibility_distance <=
                    tuple_compatibility_memory.maximum_distance
                ? 1U
                : 0U;
        conjunctive_development_intent_accepts +=
            intent_distance <= retrieval_intent_memory.maximum_distance
                ? 1U
                : 0U;
        std::cout << "conjunctive_development_positive=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " compatibility_distance="
                  << compatibility_distance
                  << " compatibility_accepted="
                  << (compatibility_distance <=
                              tuple_compatibility_memory.maximum_distance
                          ? "yes"
                          : "no")
                  << " intent_distance=" << intent_distance
                  << " intent_accepted="
                  << (intent_distance <= retrieval_intent_memory.maximum_distance
                          ? "yes"
                          : "no")
                  << " full_path=" << (eligible ? "yes" : "no") << '\n';
    }

    std::size_t conjunctive_development_negative_reaches = 0U;
    std::size_t conjunctive_development_intent_rejections = 0U;
    for (const auto& negative : conjunctive_development_negatives) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_conjunctive_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        const auto reached = memory.hook.entity.accepted &&
                             memory.hook.relation.accepted &&
                             memory.hook.tuple_found &&
                             memory.hook.entity.factor_label == negative.entity &&
                             memory.hook.relation.factor_label ==
                                 negative.relation;
        conjunctive_development_negative_reaches += reached ? 1U : 0U;
        conjunctive_development_intent_rejections +=
            reached && !memory.hook.intent_accepted &&
                    !memory.hook.action_accepted && !memory.hook.applied &&
                    delta <= 1.0e-5F
                ? 1U
                : 0U;
        std::cout << "conjunctive_development_negative=" << negative.name << '/'
                  << entities[negative.entity] << '/'
                  << relations[negative.relation]
                  << " reached=" << (reached ? "yes" : "no")
                  << " compatibility_accepted="
                  << (memory.hook.compatibility_accepted ? "yes" : "no")
                  << " intent_distance=" << memory.hook.intent_distance
                  << " intent_accepted="
                  << (memory.hook.intent_accepted ? "yes" : "no")
                  << " max_logit_delta=" << delta << '\n';
    }
    const auto expected_development_cross_rejections =
        conjunctive_development_positives.size() * (tuples.size() - 1U);
    std::cout << "conjunctive_development_summary=compatibility "
              << conjunctive_development_compatibility_accepts << '/'
              << conjunctive_development_positives.size()
              << " cross_rejections "
              << conjunctive_development_cross_rejections << '/'
              << expected_development_cross_rejections
              << " intent_positive "
              << conjunctive_development_intent_accepts << '/'
              << conjunctive_development_positives.size()
              << " negative_reaches "
              << conjunctive_development_negative_reaches << '/'
              << conjunctive_development_negatives.size()
              << " intent_negative "
              << conjunctive_development_intent_rejections << '/'
              << conjunctive_development_negatives.size() << '\n';
    if (conjunctive_development_compatibility_accepts !=
            conjunctive_development_positives.size() ||
        conjunctive_development_cross_rejections !=
            expected_development_cross_rejections ||
        conjunctive_development_intent_accepts !=
            conjunctive_development_positives.size() ||
        conjunctive_development_negative_reaches !=
            conjunctive_development_negatives.size() ||
        conjunctive_development_intent_rejections !=
            conjunctive_development_negatives.size()) {
        throw std::runtime_error(
            "conjunctive authorization failed its development preflight");
    }

    std::size_t evaluation_routed = 0U;
    std::size_t evaluation_recalled = 0U;
    for (const auto& tuple : tuples) {
        const std::vector<std::pair<std::string, std::string>> held_out_prompts{
            {"consult",
             "Consult stored memory for " + entities[tuple.entity] +
                 "; requested property: " + relations[tuple.relation] +
                 ".\nAnswer:"},
            {"associate",
             "What value does memory associate with " + entities[tuple.entity] +
                 " under " + relations[tuple.relation] + "?\nValue:"},
        };
        for (const auto& held_out : held_out_prompts) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_memory(
                model.get(), vocab, held_out.second, make_hook(), target_tensor);
            const auto rank = token_rank(memory.logits, tuple.target_token);
            const auto routed = memory.hook.applied &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label == tuple.relation;
            const auto recalled = routed &&
                                  memory.logits[static_cast<std::size_t>(tuple.target_token)] >
                                      baseline.logits[static_cast<std::size_t>(
                                          tuple.target_token)] &&
                                  rank == 1U;
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            if (routed) {
                ++evaluation_routed;
            }
            if (recalled) {
                ++evaluation_recalled;
            }
            std::cout << "evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/' << relations[tuple.relation]
                      << " entity=" << memory.hook.entity.factor_label
                      << " entity_distance=" << memory.hook.entity.distance
                      << " relation=" << memory.hook.relation.factor_label
                      << " relation_distance=" << memory.hook.relation.distance
                      << " action_variant=" << memory.hook.action_variant
                      << " action_distance=" << memory.hook.action_distance
                      << " action_accepted="
                      << (memory.hook.action_accepted ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << " target_rank=" << rank
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    if (evaluation_routed != tuples.size() * 2U) {
        throw std::runtime_error(
            "factorized memory failed to route the frozen evaluation set");
    }

    std::size_t two_stage_evaluation_routes = 0U;
    std::size_t residual_evaluation_recalled = 0U;
    std::size_t target_state_evaluation_recalled = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : two_stage_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto residual = infer_with_memory(
                model.get(),
                vocab,
                held_out.second,
                make_hook(),
                target_tensor);
            const auto target_state = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto residual_rank = token_rank(
                residual.logits, tuple.target_token);
            const auto target_state_rank = token_rank(
                target_state.logits, tuple.target_token);
            const auto routed = target_state.hook.applied &&
                                target_state.hook.entity.factor_label ==
                                    tuple.entity &&
                                target_state.hook.relation.factor_label ==
                                    tuple.relation;
            const auto residual_recalled =
                residual.hook.applied &&
                residual.hook.entity.factor_label == tuple.entity &&
                residual.hook.relation.factor_label == tuple.relation &&
                residual.logits[static_cast<std::size_t>(tuple.target_token)] >
                    baseline.logits[static_cast<std::size_t>(tuple.target_token)] &&
                residual_rank == 1U;
            const auto target_state_recalled =
                routed &&
                target_state.logits[static_cast<std::size_t>(tuple.target_token)] >
                    baseline.logits[static_cast<std::size_t>(tuple.target_token)] &&
                target_state_rank == 1U;
            two_stage_evaluation_routes += routed ? 1U : 0U;
            residual_evaluation_recalled += residual_recalled ? 1U : 0U;
            target_state_evaluation_recalled +=
                target_state_recalled ? 1U : 0U;
            std::cout << "two_stage_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " entity=" << target_state.hook.entity.factor_label
                      << " entity_distance="
                      << target_state.hook.entity.distance
                      << " entity_accepted="
                      << (target_state.hook.entity.accepted ? "yes" : "no")
                      << " relation=" << target_state.hook.relation.factor_label
                      << " relation_distance="
                      << target_state.hook.relation.distance
                      << " relation_accepted="
                      << (target_state.hook.relation.accepted ? "yes" : "no")
                      << " action_distance="
                      << target_state.hook.action_distance
                      << " action_accepted="
                      << (target_state.hook.action_accepted ? "yes" : "no")
                      << " residual_rank=" << residual_rank
                      << " target_state_rank=" << target_state_rank
                      << " residual_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, residual.logits)
                      << " target_state_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, target_state.logits)
                      << " teacher_logit_delta="
                      << maximum_logit_difference(
                             tuple.late_teacher.logits, target_state.logits)
                      << '\n';
        }
    }
    const auto two_stage_evaluation_count = tuples.size() * 2U;
    std::cout << "two_stage_evaluation_summary=routes "
              << two_stage_evaluation_routes << '/'
              << two_stage_evaluation_count
              << " residual_rank_one=" << residual_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_rank_one="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_conditional="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_routes << '\n';
    if (two_stage_evaluation_routes != 10U ||
        target_state_evaluation_recalled != 10U) {
        throw std::runtime_error(
            "historical two-stage frozen result drifted from 10/12");
    }

    std::size_t entity_address_variance_evaluation_routes = 0U;
    std::size_t entity_address_variance_evaluation_recall = 0U;
    std::size_t entity_address_variance_evaluation_entities = 0U;
    std::size_t entity_address_association_evaluation_routes = 0U;
    std::size_t entity_address_association_evaluation_recall = 0U;
    std::size_t entity_address_association_evaluation_entities = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : entity_address_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto variance = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(false),
                target_states,
                target_tensor,
                action_tensor);
            const auto association = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto variance_rank = token_rank(
                variance.logits, tuple.target_token);
            const auto association_rank = token_rank(
                association.logits, tuple.target_token);
            const auto variance_routed =
                variance.hook.applied &&
                variance.hook.entity.factor_label == tuple.entity &&
                variance.hook.relation.factor_label == tuple.relation;
            const auto association_routed =
                association.hook.applied &&
                association.hook.entity.factor_label == tuple.entity &&
                association.hook.relation.factor_label == tuple.relation;
            const auto variance_entity_matched =
                variance.hook.entity.accepted &&
                variance.hook.entity.factor_label == tuple.entity;
            const auto association_entity_matched =
                association.hook.entity.accepted &&
                association.hook.entity.factor_label == tuple.entity;
            entity_address_variance_evaluation_routes +=
                variance_routed ? 1U : 0U;
            entity_address_variance_evaluation_recall +=
                variance_routed && variance_rank == 1U ? 1U : 0U;
            entity_address_variance_evaluation_entities +=
                variance_entity_matched ? 1U : 0U;
            entity_address_association_evaluation_routes +=
                association_routed ? 1U : 0U;
            entity_address_association_evaluation_recall +=
                association_routed && association_rank == 1U ? 1U : 0U;
            entity_address_association_evaluation_entities +=
                association_entity_matched ? 1U : 0U;
            std::cout << "entity_address_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " variance_entity_distance="
                      << variance.hook.entity.distance
                      << " variance_entity="
                      << variance.hook.entity.factor_label
                      << " variance_entity_accepted="
                      << (variance.hook.entity.accepted ? "yes" : "no")
                      << " variance_routed="
                      << (variance_routed ? "yes" : "no")
                      << " variance_rank=" << variance_rank
                      << " association_entity_distance="
                      << association.hook.entity.distance
                      << " association_entity="
                      << association.hook.entity.factor_label
                      << " association_entity_accepted="
                      << (association.hook.entity.accepted ? "yes" : "no")
                      << " association_relation_accepted="
                      << (association.hook.relation.accepted ? "yes" : "no")
                      << " association_action_accepted="
                      << (association.hook.action_accepted ? "yes" : "no")
                      << " association_routed="
                      << (association_routed ? "yes" : "no")
                      << " association_rank=" << association_rank
                      << " association_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, association.logits)
                      << " teacher_logit_delta="
                      << maximum_logit_difference(
                             tuple.late_teacher.logits, association.logits)
                      << '\n';
        }
    }
    const auto entity_address_evaluation_count = tuples.size() * 3U;

    std::size_t entity_address_unknown_entity_noops = 0U;
    std::size_t entity_address_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Rigel"), std::string("Sirius")}) {
        for (const auto& held_out : entity_address_evaluation_prompts(
                 unknown, "color")) {
            ++entity_address_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto exact_noop = !memory.hook.entity.accepted &&
                                    !memory.hook.applied && delta <= 1.0e-5F;
            entity_address_unknown_entity_noops += exact_noop ? 1U : 0U;
            std::cout << "entity_address_unknown=" << held_out.first << '/'
                      << unknown
                      << " nearest_entity=" << memory.hook.entity.factor_label
                      << " entity_distance=" << memory.hook.entity.distance
                      << " entity_accepted="
                      << (memory.hook.entity.accepted ? "yes" : "no")
                      << " applied="
                      << (memory.hook.applied ? "yes" : "no")
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    std::size_t entity_address_historical_routes = 0U;
    std::size_t entity_address_historical_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : two_stage_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed =
                memory.hook.applied &&
                memory.hook.entity.factor_label == tuple.entity &&
                memory.hook.relation.factor_label == tuple.relation;
            entity_address_historical_routes += routed ? 1U : 0U;
            entity_address_historical_recall +=
                routed && token_rank(memory.logits, tuple.target_token) == 1U
                    ? 1U
                    : 0U;
        }
    }

    std::cout << "entity_address_evaluation_summary=variance_routes "
              << entity_address_variance_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " variance_entities "
              << entity_address_variance_evaluation_entities << '/'
              << entity_address_evaluation_count
              << " variance_rank_one "
              << entity_address_variance_evaluation_recall << '/'
              << entity_address_evaluation_count
              << " association_routes "
              << entity_address_association_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " association_entities "
              << entity_address_association_evaluation_entities << '/'
              << entity_address_evaluation_count
              << " association_rank_one "
              << entity_address_association_evaluation_recall << '/'
              << entity_address_evaluation_count
              << " unknown_entity_noops "
              << entity_address_unknown_entity_noops << '/'
              << entity_address_unknown_entity_count
              << " historical_routes " << entity_address_historical_routes
              << '/' << two_stage_evaluation_count
              << " historical_rank_one "
              << entity_address_historical_recall << '/'
              << two_stage_evaluation_count << '\n';
    if (entity_address_association_evaluation_routes != 9U ||
        entity_address_association_evaluation_recall != 9U ||
        entity_address_association_evaluation_entities != 18U ||
        entity_address_unknown_entity_noops != 6U ||
        entity_address_historical_routes != 12U ||
        entity_address_historical_recall != 12U) {
        throw std::runtime_error(
            "historical entity-address frozen result drifted");
    }

    std::size_t relation_nearest_evaluation_matches = 0U;
    std::size_t relation_prototype_evaluation_matches = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : relation_prototype_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto nearest = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(false, false),
                target_states,
                target_tensor,
                action_tensor);
            const auto prototype = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, false),
                target_states,
                target_tensor,
                action_tensor);
            const auto nearest_match =
                nearest.hook.relation.accepted &&
                nearest.hook.relation.factor_label == tuple.relation;
            const auto prototype_match =
                prototype.hook.relation.accepted &&
                prototype.hook.relation.factor_label == tuple.relation;
            relation_nearest_evaluation_matches += nearest_match ? 1U : 0U;
            relation_prototype_evaluation_matches += prototype_match ? 1U : 0U;
            std::cout << "relation_prototype_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " nearest_distance=" << nearest.hook.relation.distance
                      << " nearest_match="
                      << (nearest_match ? "yes" : "no")
                      << " prototype_distance="
                      << prototype.hook.relation.distance
                      << " prototype_match="
                      << (prototype_match ? "yes" : "no") << '\n';
        }
    }
    const auto relation_evaluation_count = tuples.size() * 2U;
    std::size_t relation_prototype_negative_rejections = 0U;
    std::size_t relation_prototype_negative_count = 0U;
    for (const auto& unknown : {std::string("temperature"), std::string("age")}) {
        for (const auto& held_out : relation_prototype_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++relation_prototype_negative_count;
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, false),
                target_states,
                target_tensor,
                action_tensor);
            relation_prototype_negative_rejections +=
                !memory.hook.relation.accepted ? 1U : 0U;
        }
    }
    std::cout << "relation_prototype_evaluation_summary=nearest "
              << relation_nearest_evaluation_matches << '/'
              << relation_evaluation_count << " prototype "
              << relation_prototype_evaluation_matches << '/'
              << relation_evaluation_count << " negative_rejections "
              << relation_prototype_negative_rejections << '/'
              << relation_prototype_negative_count << '\n';

    std::size_t authorization_variance_evaluation_accepts = 0U;
    std::size_t authorization_signal_evaluation_accepts = 0U;
    std::size_t authorization_signal_evaluation_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : authorization_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto variance = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, false),
                target_states,
                target_tensor,
                action_tensor);
            const auto signal = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto variance_accept = variance.hook.entity.accepted &&
                                         variance.hook.relation.accepted &&
                                         variance.hook.tuple_found &&
                                         variance.hook.action_accepted;
            const auto signal_accept = signal.hook.entity.accepted &&
                                       signal.hook.relation.accepted &&
                                       signal.hook.tuple_found &&
                                       signal.hook.action_accepted;
            const auto signal_rank = token_rank(signal.logits, tuple.target_token);
            authorization_variance_evaluation_accepts +=
                variance_accept ? 1U : 0U;
            authorization_signal_evaluation_accepts += signal_accept ? 1U : 0U;
            authorization_signal_evaluation_recall +=
                signal_accept && signal_rank == 1U ? 1U : 0U;
            std::cout << "authorization_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " variance_distance=" << variance.hook.action_distance
                      << " variance_accept="
                      << (variance_accept ? "yes" : "no")
                      << " signal_distance=" << signal.hook.action_distance
                      << " signal_accept="
                      << (signal_accept ? "yes" : "no")
                      << " signal_rank=" << signal_rank << '\n';
        }
    }
    const auto authorization_evaluation_count = tuples.size() * 2U;
    std::size_t authorization_negative_noops = 0U;
    std::size_t authorization_negative_count = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : authorization_evaluation_negatives(
                 entities[tuple.entity], relations[tuple.relation])) {
            ++authorization_negative_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied && delta <= 1.0e-5F;
            authorization_negative_noops += exact_noop ? 1U : 0U;
        }
    }
    std::cout << "authorization_evaluation_summary=variance "
              << authorization_variance_evaluation_accepts << '/'
              << authorization_evaluation_count << " signal "
              << authorization_signal_evaluation_accepts << '/'
              << authorization_evaluation_count << " signal_rank_one "
              << authorization_signal_evaluation_recall << '/'
              << authorization_evaluation_count << " negative_noops "
              << authorization_negative_noops << '/'
              << authorization_negative_count << '\n';

    std::size_t composition_routes = 0U;
    std::size_t composition_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : gate_composition_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed = memory.hook.applied &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label == tuple.relation;
            const auto rank = token_rank(memory.logits, tuple.target_token);
            composition_routes += routed ? 1U : 0U;
            composition_recall += routed && rank == 1U ? 1U : 0U;
            std::cout << "gate_composition_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " entity_accepted="
                      << (memory.hook.entity.accepted ? "yes" : "no")
                      << " relation_accepted="
                      << (memory.hook.relation.accepted ? "yes" : "no")
                      << " action_accepted="
                      << (memory.hook.action_accepted ? "yes" : "no")
                      << " routed=" << (routed ? "yes" : "no")
                      << " rank=" << rank << '\n';
        }
    }
    const auto composition_count = tuples.size() * 2U;

    std::size_t composition_wrong_intent_noops = 0U;
    for (const auto& tuple : tuples) {
        const auto prompt = gate_composition_negative_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        const auto baseline = capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor);
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            prompt,
            make_gate_invariance_hook(true, true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            baseline.logits, memory.logits);
        composition_wrong_intent_noops +=
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
                    memory.hook.tuple_found && !memory.hook.action_accepted &&
                    !memory.hook.applied && delta <= 1.0e-5F
                ? 1U
                : 0U;
    }

    std::size_t composition_missing_noops = 0U;
    std::size_t composition_missing_count = 0U;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        for (const auto& held_out : gate_composition_evaluation_prompts(
                 entities[missing.first], relations[missing.second])) {
            ++composition_missing_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            composition_missing_noops +=
                memory.hook.entity.accepted && memory.hook.relation.accepted &&
                        !memory.hook.tuple_found && !memory.hook.applied &&
                        delta <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t composition_unknown_entity_noops = 0U;
    std::size_t composition_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Vega"), std::string("Altair")}) {
        for (const auto& held_out : gate_composition_evaluation_prompts(
                 unknown, "color")) {
            ++composition_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            composition_unknown_entity_noops +=
                !memory.hook.entity.accepted && !memory.hook.applied &&
                        delta <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t composition_unknown_relation_noops = 0U;
    std::size_t composition_unknown_relation_count = 0U;
    for (const auto& unknown : {std::string("weight"), std::string("origin")}) {
        for (const auto& held_out : gate_composition_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++composition_unknown_relation_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            composition_unknown_relation_noops +=
                memory.hook.entity.accepted &&
                        !memory.hook.relation.accepted && !memory.hook.applied &&
                        delta <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::cout << "gate_composition_evaluation_summary=routes "
              << composition_routes << '/' << composition_count
              << " rank_one " << composition_recall << '/' << composition_count
              << " wrong_intent_noops " << composition_wrong_intent_noops << '/'
              << tuples.size() << " missing_noops " << composition_missing_noops
              << '/' << composition_missing_count << " unknown_entity_noops "
              << composition_unknown_entity_noops << '/'
              << composition_unknown_entity_count
              << " unknown_relation_noops "
              << composition_unknown_relation_noops << '/'
              << composition_unknown_relation_count << '\n';

    const auto historical_result_stable =
        relation_prototype_evaluation_matches == relation_evaluation_count &&
        relation_prototype_negative_rejections ==
            relation_prototype_negative_count &&
        authorization_variance_evaluation_accepts == 9U &&
        authorization_signal_evaluation_accepts == 11U &&
        authorization_signal_evaluation_recall == 11U &&
        authorization_negative_noops == 5U &&
        composition_routes == 9U && composition_recall == 9U &&
        composition_wrong_intent_noops == 2U &&
        composition_missing_noops == 4U &&
        composition_unknown_entity_noops == 3U &&
        composition_unknown_relation_noops == 4U;
    if (!historical_result_stable) {
        throw std::runtime_error(
            "historical gate-local frozen result drifted");
    }

    std::size_t frozen_compatibility_accepts = 0U;
    std::size_t frozen_cross_rejections = 0U;
    std::size_t frozen_intent_accepts = 0U;
    std::size_t frozen_single_gate_accepts = 0U;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        for (const auto& prompt : conjunctive_frozen_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const gx1::ActivationStateSequence compatibility_states{
                combined_factor_state_for_probe(
                    baseline,
                    tuple.entity,
                    tuple.relation,
                    entity_address_association_memory,
                    relation_prototype_memory)};
            const auto compatibility_distance = minimum_probe_distance(
                compatibility_states,
                association_key_for_probe(
                    tuple_compatibility_memory, tuple_index),
                tuple_compatibility_memory);
            const auto intent_distance = probe_squared_distance(
                project_normalized_for_probe(
                    baseline.hidden_state, retrieval_intent_memory),
                retrieval_intent_memory.keys.front());
            for (std::size_t key = 0;
                 key < tuple_compatibility_memory.keys.size();
                 ++key) {
                if (tuple_compatibility_memory.key_associations[key] ==
                    tuple_index) {
                    continue;
                }
                frozen_cross_rejections +=
                    minimum_probe_distance(
                        compatibility_states,
                        tuple_compatibility_memory.keys[key],
                        tuple_compatibility_memory) >
                            tuple_compatibility_memory.maximum_distance
                        ? 1U
                        : 0U;
            }
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conjunctive_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto baseline_gate = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto eligible = memory.hook.entity.accepted &&
                                  memory.hook.relation.accepted &&
                                  memory.hook.tuple_found &&
                                  memory.hook.entity.factor_label == tuple.entity &&
                                  memory.hook.relation.factor_label ==
                                      tuple.relation;
            frozen_compatibility_accepts +=
                compatibility_distance <=
                        tuple_compatibility_memory.maximum_distance
                    ? 1U
                    : 0U;
            frozen_intent_accepts +=
                intent_distance <= retrieval_intent_memory.maximum_distance
                    ? 1U
                    : 0U;
            frozen_single_gate_accepts +=
                baseline_gate.hook.action_accepted ? 1U : 0U;
            std::cout << "conjunctive_frozen_positive=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " compatibility_distance="
                      << compatibility_distance
                      << " compatibility_accepted="
                      << (compatibility_distance <=
                                  tuple_compatibility_memory.maximum_distance
                              ? "yes"
                              : "no")
                      << " intent_distance=" << intent_distance
                      << " intent_accepted="
                      << (intent_distance <=
                                  retrieval_intent_memory.maximum_distance
                              ? "yes"
                              : "no")
                      << " full_path=" << (eligible ? "yes" : "no")
                      << " single_gate_accepted="
                      << (baseline_gate.hook.action_accepted ? "yes" : "no")
                      << '\n';
        }
    }

    std::size_t frozen_negative_reaches = 0U;
    std::size_t frozen_intent_negative_noops = 0U;
    std::size_t frozen_single_gate_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_frozen_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conjunctive_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto baseline_gate = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_gate_invariance_hook(true, true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto baseline_delta = maximum_logit_difference(
                baseline.logits, baseline_gate.logits);
            const auto reached = memory.hook.entity.accepted &&
                                 memory.hook.relation.accepted &&
                                 memory.hook.tuple_found &&
                                 memory.hook.entity.factor_label == tuple.entity &&
                                 memory.hook.relation.factor_label ==
                                     tuple.relation;
            frozen_negative_reaches += reached ? 1U : 0U;
            frozen_intent_negative_noops +=
                reached && !memory.hook.intent_accepted &&
                        !memory.hook.action_accepted && !memory.hook.applied &&
                        delta <= 1.0e-5F
                    ? 1U
                    : 0U;
            frozen_single_gate_negative_noops +=
                !baseline_gate.hook.action_accepted &&
                        !baseline_gate.hook.applied &&
                        baseline_delta <= 1.0e-5F
                    ? 1U
                    : 0U;
            std::cout << "conjunctive_frozen_negative=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " reached=" << (reached ? "yes" : "no")
                      << " compatibility_accepted="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent_distance=" << memory.hook.intent_distance
                      << " intent_accepted="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " max_logit_delta=" << delta << '\n';
        }
    }
    const auto frozen_positive_count = tuples.size() * 2U;
    const auto frozen_negative_count = tuples.size() * 6U;
    const auto frozen_cross_count = frozen_positive_count * (tuples.size() - 1U);
    std::cout << "conjunctive_local_frozen_summary=compatibility "
              << frozen_compatibility_accepts << '/' << frozen_positive_count
              << " cross_rejections " << frozen_cross_rejections << '/'
              << frozen_cross_count << " intent_positive "
              << frozen_intent_accepts << '/' << frozen_positive_count
              << " negative_reaches " << frozen_negative_reaches << '/'
              << frozen_negative_count << " intent_negative_noops "
              << frozen_intent_negative_noops << '/' << frozen_negative_count
              << " baseline_positive " << frozen_single_gate_accepts << '/'
              << frozen_positive_count << " baseline_negative_noops "
              << frozen_single_gate_negative_noops << '/'
              << frozen_negative_count << '\n';
    const auto local_frozen_passed =
        frozen_compatibility_accepts == frozen_positive_count &&
        frozen_cross_rejections == frozen_cross_count &&
        frozen_intent_accepts == frozen_positive_count &&
        frozen_negative_reaches == frozen_negative_count &&
        frozen_intent_negative_noops == frozen_negative_count;
    if (!local_frozen_passed) {
        throw std::runtime_error(
            "conjunctive authorization failed a local frozen criterion");
    }

    std::size_t conjunctive_composition_routes = 0U;
    std::size_t conjunctive_composition_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conjunctive_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed = memory.hook.applied &&
                                memory.hook.compatibility_accepted &&
                                memory.hook.intent_accepted &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label ==
                                    tuple.relation;
            const auto rank = token_rank(memory.logits, tuple.target_token);
            conjunctive_composition_routes += routed ? 1U : 0U;
            conjunctive_composition_recall +=
                routed && rank == 1U ? 1U : 0U;
            std::cout << "conjunctive_composition_positive=" << prompt.first
                      << '/' << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " compatibility_accepted="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent_accepted="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " routed=" << (routed ? "yes" : "no")
                      << " rank=" << rank << '\n';
        }
    }

    std::size_t conjunctive_composition_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conjunctive_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    memory.hook.compatibility_accepted &&
                                    !memory.hook.intent_accepted &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied && delta <= 1.0e-5F;
            conjunctive_composition_negative_noops += exact_noop ? 1U : 0U;
            std::cout << "conjunctive_composition_negative=" << prompt.first
                      << '/' << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " entity_accepted="
                      << (memory.hook.entity.accepted ? "yes" : "no")
                      << " relation_accepted="
                      << (memory.hook.relation.accepted ? "yes" : "no")
                      << " tuple_found="
                      << (memory.hook.tuple_found ? "yes" : "no")
                      << " compatibility_accepted="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent_accepted="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    std::size_t conjunctive_missing_noops = 0U;
    std::size_t conjunctive_missing_count = 0U;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 entities[missing.first], relations[missing.second])) {
            ++conjunctive_missing_count;
            const auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_conjunctive_hook(),
                target_states, target_tensor, action_tensor);
            conjunctive_missing_noops +=
                memory.hook.entity.accepted && memory.hook.relation.accepted &&
                        !memory.hook.tuple_found && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t conjunctive_unknown_entity_noops = 0U;
    std::size_t conjunctive_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Vega"), std::string("Altair")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 unknown, "color")) {
            ++conjunctive_unknown_entity_count;
            const auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_conjunctive_hook(),
                target_states, target_tensor, action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto exact_noop = !memory.hook.entity.accepted &&
                                    !memory.hook.applied && delta <= 1.0e-5F;
            conjunctive_unknown_entity_noops += exact_noop ? 1U : 0U;
            std::cout << "conjunctive_unknown_entity=" << prompt.first << '/'
                      << unknown << "/color"
                      << " entity_accepted="
                      << (memory.hook.entity.accepted ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    std::size_t conjunctive_unknown_relation_noops = 0U;
    std::size_t conjunctive_unknown_relation_count = 0U;
    for (const auto& unknown : {std::string("weight"), std::string("origin")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++conjunctive_unknown_relation_count;
            const auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_conjunctive_hook(),
                target_states, target_tensor, action_tensor);
            conjunctive_unknown_relation_noops +=
                memory.hook.entity.accepted && !memory.hook.relation.accepted &&
                        !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t conjunctive_regression_noops = 0U;
    std::size_t conjunctive_regression_count = 0U;
    const auto check_regression = [&](const std::string& prompt,
                                      const InferenceResult& baseline) {
        ++conjunctive_regression_count;
        const auto memory = infer_with_target_state_memory(
            model.get(), vocab, prompt, make_conjunctive_hook(), target_states,
            target_tensor, action_tensor);
        conjunctive_regression_noops +=
            !memory.hook.action_accepted && !memory.hook.applied &&
                    maximum_logit_difference(
                        baseline.logits, memory.logits) <= 1.0e-5F
                ? 1U
                : 0U;
    };
    for (const auto& negative : wrong_intents) {
        check_regression(negative.prompt, negative.baseline);
    }
    for (const auto& tuple : tuples) {
        for (const auto& prompt : authorization_evaluation_negatives(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            check_regression(prompt.second, baseline);
        }
        const auto prompt = gate_composition_negative_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        const auto baseline = capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor);
        check_regression(prompt, baseline);
    }

    const auto conjunctive_composition_count = tuples.size() * 2U;
    const auto conjunctive_composition_negative_count = tuples.size() * 3U;
    std::cout << "conjunctive_composition_summary=routes "
              << conjunctive_composition_routes << '/'
              << conjunctive_composition_count << " rank_one "
              << conjunctive_composition_recall << '/'
              << conjunctive_composition_count << " wrong_intent_noops "
              << conjunctive_composition_negative_noops << '/'
              << conjunctive_composition_negative_count << " missing_noops "
              << conjunctive_missing_noops << '/' << conjunctive_missing_count
              << " unknown_entity_noops " << conjunctive_unknown_entity_noops
              << '/' << conjunctive_unknown_entity_count
              << " unknown_relation_noops "
              << conjunctive_unknown_relation_noops << '/'
              << conjunctive_unknown_relation_count << " regression_noops "
              << conjunctive_regression_noops << '/'
              << conjunctive_regression_count << '\n';
    const auto historical_conjunctive_result_reproduced =
        conjunctive_composition_routes == 11U &&
        conjunctive_composition_recall == 11U &&
        conjunctive_composition_negative_noops == 17U &&
        conjunctive_missing_noops == 4U &&
        conjunctive_missing_count == 4U &&
        conjunctive_unknown_entity_noops == 3U &&
        conjunctive_unknown_entity_count == 4U &&
        conjunctive_unknown_relation_noops == 4U &&
        conjunctive_unknown_relation_count == 4U &&
        conjunctive_regression_noops == 30U &&
        conjunctive_regression_count == 30U;
    if (!historical_conjunctive_result_reproduced) {
        throw std::runtime_error(
            "historical conjunctive authorization result changed");
    }

    std::vector<ConjunctiveProbeSpec> replication_development_positives;
    std::vector<ConjunctiveProbeSpec> replication_development_negatives;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const auto& entity = entities[tuple.entity];
        const auto& relation = relations[tuple.relation];
        for (const auto& prompt : replication_development_positive_prompts(
                 entity, relation)) {
            replication_development_positives.push_back({
                tuple_index,
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor),
            });
        }
        for (const auto& prompt : replication_development_negative_prompts(
                 entity, relation)) {
            replication_development_negatives.push_back({
                tuple_index,
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor),
            });
        }
    }

    std::vector<UnknownEntityProbeSpec> replication_development_unknowns;
    for (const auto& unknown : {std::string("Deneb"), std::string("Polaris")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : replication_development_positive_prompts(
                     unknown, relation)) {
                replication_development_unknowns.push_back({
                    prompt.first,
                    unknown,
                    relation,
                    prompt.second,
                    capture(
                        model.get(),
                        vocab,
                        prompt.second,
                        hidden_dimension,
                        target_tensor),
                });
            }
        }
    }

    std::vector<gx1::ActivationMemoryConstructionView>
        replication_knownness_construction;
    std::vector<gx1::ActivationMemoryValidationView>
        replication_knownness_validation;
    std::vector<gx1::ActivationMemoryCalibrationView>
        replication_knownness_negatives;
    std::vector<std::vector<float>> replication_known_states;
    std::vector<std::vector<float>> replication_unknown_states;
    std::vector<gx1::ActivationMemoryConstructionView>
        replication_compatibility_construction;
    std::vector<gx1::ActivationMemoryValidationView>
        replication_compatibility_validation;
    std::vector<gx1::ActivationMemoryCalibrationView>
        replication_compatibility_negatives;
    std::vector<gx1::ActivationMemoryConstructionView>
        replication_intent_construction;
    std::vector<gx1::ActivationMemoryValidationView>
        replication_intent_validation;
    std::vector<gx1::ActivationMemoryCalibrationView>
        replication_intent_negatives;
    std::vector<std::vector<float>> replication_intent_positive_states;
    std::vector<std::vector<float>> replication_intent_negative_states;

    for (std::size_t index = 0;
         index < replication_development_positives.size();
         ++index) {
        const auto& positive = replication_development_positives[index];
        const auto known_state = nearest_factor_state_for_probe(
            positive.baseline.token_states,
            positive.entity,
            entity_address_association_memory);
        replication_known_states.push_back(known_state);
        auto combined = combined_factor_state_for_probe(
            positive.baseline,
            positive.entity,
            positive.relation,
            entity_address_association_memory,
            relation_prototype_memory);
        const auto is_validation = index % 4U == 3U;
        if (is_validation) {
            replication_knownness_validation.push_back({0U, {known_state}});
            replication_compatibility_validation.push_back({
                positive.tuple, {combined}});
        } else {
            replication_knownness_construction.push_back({
                0U, {known_state}, known_state, known_state});
            replication_compatibility_construction.push_back({
                positive.tuple, {combined}, combined, combined});
        }
        replication_intent_construction.push_back({
            0U,
            {positive.baseline.hidden_state},
            positive.baseline.hidden_state,
            positive.baseline.hidden_state,
        });
        replication_intent_validation.push_back({
            0U, {positive.baseline.hidden_state}});
        for (std::size_t candidate = 0; candidate < tuples.size(); ++candidate) {
            if (candidate != positive.tuple) {
                replication_compatibility_negatives.push_back({
                    candidate, {combined}});
            }
        }
        replication_intent_positive_states.push_back(
            positive.baseline.hidden_state);
    }
    for (const auto& negative : replication_development_negatives) {
        replication_intent_negatives.push_back({
            0U, {negative.baseline.hidden_state}});
        replication_intent_negative_states.push_back(
            negative.baseline.hidden_state);
    }
    for (const auto& unknown : replication_development_unknowns) {
        auto state = nearest_factor_state_for_probe(
            unknown.baseline.token_states,
            entity_address_association_memory);
        replication_knownness_negatives.push_back({0U, {state}});
        replication_unknown_states.push_back(std::move(state));
    }

    const auto build_contrastive = [&build_stage](
                                       const char* stage,
                                       const std::vector<
                                           gx1::ActivationMemoryConstructionView>&
                                           construction,
                                       const std::vector<
                                           gx1::ActivationMemoryCalibrationView>&
                                           negatives,
                                       const std::vector<
                                           gx1::ActivationMemoryValidationView>&
                                           validation,
                                       const std::vector<std::vector<float>>&
                                           positive_states,
                                       const std::vector<std::vector<float>>&
                                           negative_states) {
        build_stage(stage);
        for (const auto width : {64U, 128U, 256U, 512U}) {
            try {
                auto memory = gx1::ActivationMemoryBuilder::build(
                    construction,
                    negatives,
                    validation,
                    gx1::ActivationMemoryBuildConfig{
                        width,
                        0.5F,
                        false,
                        gx1::ActivationProjectionStrategy::authorization_signal,
                        gx1::ActivationValidationScope::global,
                        gx1::ActivationKeyStrategy::association_centroid,
                        false,
                    });
                auto maximum_positive_distance = 0.0F;
                auto minimum_negative_distance =
                    std::numeric_limits<float>::max();
                for (const auto& state : positive_states) {
                    maximum_positive_distance = std::max(
                        maximum_positive_distance,
                        probe_squared_distance(
                            project_normalized_for_probe(state, memory),
                            memory.keys.front()));
                }
                for (const auto& state : negative_states) {
                    minimum_negative_distance = std::min(
                        minimum_negative_distance,
                        probe_squared_distance(
                            project_normalized_for_probe(state, memory),
                            memory.keys.front()));
                }
                memory.maximum_validation_distance =
                    maximum_positive_distance;
                memory.minimum_negative_distance =
                    minimum_negative_distance;
                memory.maximum_distance = maximum_positive_distance +
                                          0.5F * std::max(
                                                     0.0F,
                                                     minimum_negative_distance -
                                                         maximum_positive_distance);
                auto calibration = calibrate_contrastive_gate(
                    memory, positive_states, negative_states);
                const auto accepted = [&](const std::vector<float>& state) {
                    const auto query = project_normalized_for_probe(state, memory);
                    const auto positive_distance = probe_squared_distance(
                        query, memory.keys.front());
                    const auto negative_distance = probe_squared_distance(
                        query, calibration.negative_prototype);
                    return positive_distance <= memory.maximum_distance &&
                           positive_distance + calibration.minimum_margin <=
                               negative_distance;
                };
                if (!std::all_of(
                        positive_states.begin(),
                        positive_states.end(),
                        accepted) ||
                    std::any_of(
                        negative_states.begin(),
                        negative_states.end(),
                        accepted)) {
                    throw std::runtime_error(
                        "direct contrastive preflight did not separate");
                }
                std::cout << "development_width=" << stage << '/' << width
                          << "/accepted\n";
                return std::pair<gx1::ActivationMemoryBuildResult,
                                 ContrastiveCalibration>{
                    std::move(memory), std::move(calibration)};
            } catch (const std::exception& error) {
                std::cout << "development_width=" << stage << '/' << width
                          << "/rejected reason=" << error.what() << '\n';
            }
        }
        throw std::runtime_error(
            std::string("no development width separates ") + stage);
    };

    const auto replication_knownness = build_contrastive(
        "known-entity",
        replication_knownness_construction,
        replication_knownness_negatives,
        replication_knownness_validation,
        replication_known_states,
        replication_unknown_states);
    build_stage("replication-tuple-compatibility");
    const auto replication_compatibility_memory = [&]()
        -> gx1::ActivationMemoryBuildResult {
        for (const auto width : {64U, 128U, 256U, 512U}) {
            try {
                auto memory = gx1::ActivationMemoryBuilder::build(
                    replication_compatibility_construction,
                    replication_compatibility_negatives,
                    replication_compatibility_validation,
                    gx1::ActivationMemoryBuildConfig{
                        width,
                        0.5F,
                        false,
                        gx1::ActivationProjectionStrategy::association_signal,
                        gx1::ActivationValidationScope::global,
                        gx1::ActivationKeyStrategy::association_centroid,
                    });
                auto maximum_eligible_distance = 0.0F;
                auto minimum_cross_distance =
                    std::numeric_limits<float>::max();
                const auto measure_compatibility = [&](
                                                       const ConjunctiveProbeSpec&
                                                           probe) {
                    const gx1::ActivationStateSequence states{
                        combined_factor_state_for_probe(
                            probe.baseline,
                            probe.entity,
                            probe.relation,
                            entity_address_association_memory,
                            relation_prototype_memory)};
                    for (std::size_t key = 0; key < memory.keys.size(); ++key) {
                        const auto distance = minimum_probe_distance(
                            states, memory.keys[key], memory);
                        if (memory.key_associations[key] == probe.tuple) {
                            maximum_eligible_distance = std::max(
                                maximum_eligible_distance, distance);
                        } else {
                            minimum_cross_distance = std::min(
                                minimum_cross_distance, distance);
                        }
                    }
                };
                for (const auto& positive :
                     replication_development_positives) {
                    measure_compatibility(positive);
                }
                for (const auto& negative :
                     replication_development_negatives) {
                    measure_compatibility(negative);
                }
                if (!(maximum_eligible_distance < minimum_cross_distance)) {
                    throw std::runtime_error(
                        "eligible and cross-tuple compatibility overlap");
                }
                memory.maximum_validation_distance =
                    maximum_eligible_distance;
                memory.minimum_negative_distance = minimum_cross_distance;
                memory.maximum_distance = maximum_eligible_distance +
                                          0.5F * (minimum_cross_distance -
                                                  maximum_eligible_distance);
                std::size_t positive_accepts = 0U;
                std::size_t cross_rejections = 0U;
                for (const auto& positive : replication_development_positives) {
                    const gx1::ActivationStateSequence states{
                        combined_factor_state_for_probe(
                            positive.baseline,
                            positive.entity,
                            positive.relation,
                            entity_address_association_memory,
                            relation_prototype_memory)};
                    for (std::size_t key = 0; key < memory.keys.size(); ++key) {
                        const auto distance = minimum_probe_distance(
                            states, memory.keys[key], memory);
                        if (memory.key_associations[key] == positive.tuple) {
                            positive_accepts +=
                                distance <= memory.maximum_distance ? 1U : 0U;
                        } else {
                            cross_rejections +=
                                distance > memory.maximum_distance ? 1U : 0U;
                        }
                    }
                }
                if (positive_accepts !=
                        replication_development_positives.size() ||
                    cross_rejections !=
                        replication_development_positives.size() *
                            (tuples.size() - 1U)) {
                    throw std::runtime_error(
                        "direct compatibility preflight did not separate");
                }
                std::cout
                    << "development_width=replication-tuple-compatibility/"
                    << width << "/accepted\n";
                return memory;
            } catch (const std::exception& error) {
                std::cout
                    << "development_width=replication-tuple-compatibility/"
                    << width << "/rejected reason=" << error.what() << '\n';
            }
        }
        throw std::runtime_error(
            "no replication tuple-compatibility development width separates");
    }();
    const auto replication_intent = build_contrastive(
        "contrastive-intent",
        replication_intent_construction,
        replication_intent_negatives,
        replication_intent_validation,
        replication_intent_positive_states,
        replication_intent_negative_states);
    const auto& replication_knownness_memory = replication_knownness.first;
    const auto& replication_knownness_calibration = replication_knownness.second;
    const auto& replication_intent_memory = replication_intent.first;
    const auto& replication_intent_calibration = replication_intent.second;

    std::cout << "replication_knownness_radius="
              << replication_knownness_memory.maximum_distance
              << " knownness_margin="
              << replication_knownness_calibration.minimum_margin
              << " knownness_positive_gap="
              << replication_knownness_calibration.minimum_positive_gap
              << " knownness_negative_gap="
              << replication_knownness_calibration.maximum_negative_gap
              << " replication_compatibility_radius="
              << replication_compatibility_memory.maximum_distance
              << " replication_intent_radius="
              << replication_intent_memory.maximum_distance
              << " intent_margin="
              << replication_intent_calibration.minimum_margin
              << " intent_positive_gap="
              << replication_intent_calibration.minimum_positive_gap
              << " intent_negative_gap="
              << replication_intent_calibration.maximum_negative_gap << '\n';

    auto replication_payloads = std::make_shared<gx1::TupleResidualLedger>();
    for (std::size_t index = 0;
         index < replication_compatibility_memory.keys.size();
         ++index) {
        const auto tuple_index =
            replication_compatibility_memory.key_associations[index];
        replication_payloads->insert_variant(
            tuples.at(tuple_index).entity,
            tuples.at(tuple_index).relation,
            replication_compatibility_memory.keys[index],
            replication_compatibility_memory.residuals[index]);
    }
    const auto make_replication_hook = [&]() {
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_prototype_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_prototype_memory),
            relation_prototype_labels,
            1.0F,
            replication_payloads,
            replication_compatibility_memory.maximum_distance,
            factor_config(replication_compatibility_memory),
            gx1::RetrievalIntentGateConfig{
                factor_config(replication_intent_memory),
                replication_intent_memory.keys.front(),
                replication_intent_calibration.negative_prototype,
                replication_intent_calibration.minimum_margin,
            },
            true,
            gx1::ContrastiveGateConfig{
                factor_config(replication_knownness_memory),
                replication_knownness_memory.keys.front(),
                replication_knownness_calibration.negative_prototype,
                replication_knownness_calibration.minimum_margin,
            });
    };

    const auto gate_accepts = [](const std::vector<float>& state,
                                 const gx1::ActivationMemoryBuildResult& memory,
                                 const ContrastiveCalibration& calibration) {
        const auto query = project_normalized_for_probe(state, memory);
        const auto positive_distance = probe_squared_distance(
            query, memory.keys.front());
        const auto negative_distance = probe_squared_distance(
            query, calibration.negative_prototype);
        return positive_distance <= memory.maximum_distance &&
               positive_distance + calibration.minimum_margin <=
                   negative_distance;
    };
    const auto replication_known_accepts = static_cast<std::size_t>(
        std::count_if(
            replication_known_states.begin(),
            replication_known_states.end(),
            [&](const std::vector<float>& state) {
                return gate_accepts(
                    state,
                    replication_knownness_memory,
                    replication_knownness_calibration);
            }));
    const auto replication_unknown_rejections = static_cast<std::size_t>(
        std::count_if(
            replication_unknown_states.begin(),
            replication_unknown_states.end(),
            [&](const std::vector<float>& state) {
                return !gate_accepts(
                    state,
                    replication_knownness_memory,
                    replication_knownness_calibration);
            }));
    const auto replication_intent_positive_accepts = static_cast<std::size_t>(
        std::count_if(
            replication_intent_positive_states.begin(),
            replication_intent_positive_states.end(),
            [&](const std::vector<float>& state) {
                return gate_accepts(
                    state,
                    replication_intent_memory,
                    replication_intent_calibration);
            }));

    std::size_t replication_negative_noops = 0U;
    for (const auto& negative : replication_development_negatives) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_replication_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto exact_noop = memory.hook.entity.accepted &&
                                memory.hook.known_entity_accepted &&
                                memory.hook.relation.accepted &&
                                memory.hook.tuple_found &&
                                memory.hook.compatibility_accepted &&
                                !memory.hook.intent_accepted &&
                                !memory.hook.applied &&
                                maximum_logit_difference(
                                    negative.baseline.logits,
                                    memory.logits) <= 1.0e-5F;
        replication_negative_noops += exact_noop ? 1U : 0U;
        std::cout << "replication_development_negative=" << negative.name << '/'
                  << entities[negative.entity] << '/'
                  << relations[negative.relation]
                  << " known_entity="
                  << (memory.hook.known_entity_accepted ? "yes" : "no")
                  << " compatibility="
                  << (memory.hook.compatibility_accepted ? "yes" : "no")
                  << " intent="
                  << (memory.hook.intent_accepted ? "yes" : "no")
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << '\n';
    }
    std::cout << "replication_development_summary=known "
              << replication_known_accepts << '/'
              << replication_known_states.size() << " unknown_rejections "
              << replication_unknown_rejections << '/'
              << replication_unknown_states.size() << " compatibility "
              << replication_development_positives.size() << '/'
              << replication_development_positives.size()
              << " cross_rejections "
              << replication_development_positives.size() *
                     (tuples.size() - 1U)
              << '/'
              << replication_development_positives.size() *
                     (tuples.size() - 1U)
              << " intent_positive " << replication_intent_positive_accepts
              << '/' << replication_intent_positive_states.size()
              << " negative_noops " << replication_negative_noops << '/'
              << replication_development_negatives.size() << '\n';
    if (replication_known_accepts != replication_known_states.size() ||
        replication_unknown_rejections != replication_unknown_states.size() ||
        replication_intent_positive_accepts !=
            replication_intent_positive_states.size() ||
        replication_negative_noops !=
            replication_development_negatives.size()) {
        throw std::runtime_error(
            "composition-stable replication failed development preflight");
    }

    std::size_t prior_local_compatibility_accepts = 0U;
    std::size_t prior_local_cross_rejections = 0U;
    std::size_t prior_local_intent_accepts = 0U;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        for (const auto& prompt : conjunctive_frozen_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const gx1::ActivationStateSequence compatibility_states{
                combined_factor_state_for_probe(
                    baseline,
                    tuple.entity,
                    tuple.relation,
                    entity_address_association_memory,
                    relation_prototype_memory)};
            for (std::size_t key = 0;
                 key < replication_compatibility_memory.keys.size();
                 ++key) {
                const auto distance = minimum_probe_distance(
                    compatibility_states,
                    replication_compatibility_memory.keys[key],
                    replication_compatibility_memory);
                if (replication_compatibility_memory.key_associations[key] ==
                    tuple_index) {
                    prior_local_compatibility_accepts +=
                        distance <=
                                replication_compatibility_memory.maximum_distance
                            ? 1U
                            : 0U;
                } else {
                    prior_local_cross_rejections +=
                        distance >
                                replication_compatibility_memory.maximum_distance
                            ? 1U
                            : 0U;
                }
            }
            prior_local_intent_accepts +=
                gate_accepts(
                    baseline.hidden_state,
                    replication_intent_memory,
                    replication_intent_calibration)
                    ? 1U
                    : 0U;
        }
    }

    std::size_t prior_local_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_frozen_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_replication_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.known_entity_accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    memory.hook.compatibility_accepted &&
                                    !memory.hook.intent_accepted &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied &&
                                    maximum_logit_difference(
                                        baseline.logits,
                                        memory.logits) <= 1.0e-5F;
            prior_local_negative_noops += exact_noop ? 1U : 0U;
        }
    }

    std::size_t prior_composition_routes = 0U;
    std::size_t prior_composition_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_replication_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed = memory.hook.applied &&
                                memory.hook.known_entity_accepted &&
                                memory.hook.compatibility_accepted &&
                                memory.hook.intent_accepted &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label ==
                                    tuple.relation;
            const auto rank = token_rank(memory.logits, tuple.target_token);
            prior_composition_routes += routed ? 1U : 0U;
            prior_composition_recall += routed && rank == 1U ? 1U : 0U;
            std::cout << "replication_prior_positive=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " known_entity="
                      << (memory.hook.known_entity_accepted ? "yes" : "no")
                      << " compatibility="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " routed=" << (routed ? "yes" : "no")
                      << " rank=" << rank << '\n';
        }
    }

    std::size_t prior_composition_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_replication_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.known_entity_accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    memory.hook.compatibility_accepted &&
                                    !memory.hook.intent_accepted &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied &&
                                    maximum_logit_difference(
                                        baseline.logits,
                                        memory.logits) <= 1.0e-5F;
            prior_composition_negative_noops += exact_noop ? 1U : 0U;
            std::cout << "replication_prior_negative=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " compatibility="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << '\n';
        }
    }

    std::size_t prior_wrong_intent_noops = 0U;
    std::size_t prior_wrong_intent_count = 0U;
    const auto check_prior_wrong_intent = [&](const std::string& prompt,
                                              const InferenceResult& baseline) {
        ++prior_wrong_intent_count;
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            prompt,
            make_replication_hook(),
            target_states,
            target_tensor,
            action_tensor);
        prior_wrong_intent_noops +=
            !memory.hook.action_accepted && !memory.hook.applied &&
                    maximum_logit_difference(
                        baseline.logits, memory.logits) <= 1.0e-5F
                ? 1U
                : 0U;
    };
    for (const auto& negative : wrong_intents) {
        check_prior_wrong_intent(negative.prompt, negative.baseline);
    }
    for (const auto& tuple : tuples) {
        for (const auto& prompt : authorization_evaluation_negatives(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            check_prior_wrong_intent(prompt.second, baseline);
        }
        const auto prompt = gate_composition_negative_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        const auto baseline = capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor);
        check_prior_wrong_intent(prompt, baseline);
    }

    std::size_t prior_missing_noops = 0U;
    std::size_t prior_missing_count = 0U;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 entities[missing.first], relations[missing.second])) {
            ++prior_missing_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_replication_hook(),
                target_states, target_tensor, action_tensor);
            prior_missing_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        memory.hook.relation.accepted &&
                        !memory.hook.tuple_found && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t prior_unknown_entity_noops = 0U;
    std::size_t prior_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Vega"), std::string("Altair")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 unknown, "color")) {
            ++prior_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto selected_state = nearest_factor_state_for_probe(
                baseline.token_states,
                entity_address_association_memory);
            const auto knownness_rejected = !gate_accepts(
                selected_state,
                replication_knownness_memory,
                replication_knownness_calibration);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_replication_hook(),
                target_states, target_tensor, action_tensor);
            prior_unknown_entity_noops +=
                knownness_rejected && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
            std::cout << "replication_prior_unknown_entity=" << prompt.first
                      << '/' << unknown << "/color"
                      << " knownness_rejected="
                      << (knownness_rejected ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << '\n';
        }
    }

    std::size_t prior_unknown_relation_noops = 0U;
    std::size_t prior_unknown_relation_count = 0U;
    for (const auto& unknown : {std::string("weight"), std::string("origin")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++prior_unknown_relation_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(), vocab, prompt.second, make_replication_hook(),
                target_states, target_tensor, action_tensor);
            prior_unknown_relation_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        !memory.hook.relation.accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    const auto prior_local_positive_count = tuples.size() * 2U;
    const auto prior_local_cross_count =
        prior_local_positive_count * (tuples.size() - 1U);
    const auto prior_local_negative_count = tuples.size() * 6U;
    const auto prior_composition_count = tuples.size() * 2U;
    const auto prior_composition_negative_count = tuples.size() * 3U;
    std::cout << "replication_prior_frozen_summary=local_compatibility "
              << prior_local_compatibility_accepts << '/'
              << prior_local_positive_count << " local_cross_rejections "
              << prior_local_cross_rejections << '/' << prior_local_cross_count
              << " local_intent_positive " << prior_local_intent_accepts << '/'
              << prior_local_positive_count << " local_negative_noops "
              << prior_local_negative_noops << '/' << prior_local_negative_count
              << " composition_routes " << prior_composition_routes << '/'
              << prior_composition_count << " composition_rank_one "
              << prior_composition_recall << '/' << prior_composition_count
              << " composition_negative_noops "
              << prior_composition_negative_noops << '/'
              << prior_composition_negative_count << " older_negative_noops "
              << prior_wrong_intent_noops << '/' << prior_wrong_intent_count
              << " missing_noops " << prior_missing_noops << '/'
              << prior_missing_count << " unknown_entity_noops "
              << prior_unknown_entity_noops << '/'
              << prior_unknown_entity_count << " unknown_relation_noops "
              << prior_unknown_relation_noops << '/'
              << prior_unknown_relation_count << '\n';
    const auto prior_frozen_regression_passed =
        prior_local_compatibility_accepts == prior_local_positive_count &&
        prior_local_cross_rejections == prior_local_cross_count &&
        prior_local_intent_accepts == prior_local_positive_count &&
        prior_local_negative_noops == prior_local_negative_count &&
        prior_composition_routes == prior_composition_count &&
        prior_composition_recall == prior_composition_count &&
        prior_composition_negative_noops ==
            prior_composition_negative_count &&
        prior_wrong_intent_noops == prior_wrong_intent_count &&
        prior_missing_noops == prior_missing_count &&
        prior_unknown_entity_noops == prior_unknown_entity_count &&
        prior_unknown_relation_noops == prior_unknown_relation_count;
    const auto historical_replication_reproduced =
        !prior_frozen_regression_passed &&
        prior_local_compatibility_accepts == prior_local_positive_count &&
        prior_local_cross_rejections == prior_local_cross_count &&
        prior_local_intent_accepts == prior_local_positive_count &&
        prior_local_negative_noops == prior_local_negative_count &&
        prior_composition_routes + 1U == prior_composition_count &&
        prior_composition_recall + 1U == prior_composition_count &&
        prior_composition_negative_noops ==
            prior_composition_negative_count &&
        prior_wrong_intent_noops == prior_wrong_intent_count &&
        prior_missing_noops == prior_missing_count &&
        prior_unknown_entity_noops == prior_unknown_entity_count &&
        prior_unknown_relation_noops == prior_unknown_relation_count;
    if (!historical_replication_reproduced) {
        throw std::runtime_error(
            "historical composition-stable replication result changed");
    }

    std::vector<std::vector<std::vector<float>>> conditioned_known_states(
        entities.size());
    std::size_t conditioned_association_matches = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : conditioned_knownness_development_prompts(
                     entities[entity], relation)) {
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                auto selection = nearest_factor_selection_for_probe(
                    baseline.token_states,
                    entity_address_association_memory);
                conditioned_association_matches +=
                    selection.first == entity ? 1U : 0U;
                conditioned_known_states[entity].push_back(
                    std::move(selection.second));
            }
        }
    }

    std::vector<UnknownEntityProbeSpec> conditioned_unknown_probes;
    std::vector<std::pair<std::size_t, std::vector<float>>>
        conditioned_unknown_selections;
    for (const auto& unknown : {std::string("Canopus"), std::string("Achernar"),
                                std::string("Hadar"), std::string("Acrux")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : conditioned_knownness_development_prompts(
                     unknown, relation)) {
                auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                conditioned_unknown_selections.push_back(
                    nearest_factor_selection_for_probe(
                        baseline.token_states,
                        entity_address_association_memory));
                conditioned_unknown_probes.push_back({
                    prompt.first,
                    unknown,
                    relation,
                    prompt.second,
                    std::move(baseline),
                });
            }
        }
    }

    std::vector<gx1::ActivationMemoryConstructionView>
        conditioned_construction;
    std::vector<gx1::ActivationMemoryValidationView> conditioned_validation;
    std::vector<gx1::ActivationMemoryCalibrationView> conditioned_negatives;
    for (std::size_t entity = 0; entity < conditioned_known_states.size();
         ++entity) {
        for (const auto& state : conditioned_known_states[entity]) {
            conditioned_construction.push_back({
                entity, {state}, state, state});
            conditioned_validation.push_back({entity, {state}});
        }
        for (std::size_t other = 0; other < conditioned_known_states.size();
             ++other) {
            if (other == entity) {
                continue;
            }
            for (const auto& state : conditioned_known_states[other]) {
                conditioned_negatives.push_back({entity, {state}});
            }
        }
        for (const auto& unknown : conditioned_unknown_selections) {
            conditioned_negatives.push_back({entity, {unknown.second}});
        }
    }

    const auto conditioned_build = [&]()
        -> std::pair<gx1::ActivationMemoryBuildResult,
                     std::vector<gx1::LabelConditionedGateEntry>> {
        build_stage("label-conditioned-knownness");
        for (const auto width : {32U}) {
            try {
                auto memory = gx1::ActivationMemoryBuilder::build(
                    conditioned_construction,
                    conditioned_negatives,
                    conditioned_validation,
                    gx1::ActivationMemoryBuildConfig{
                        width,
                        0.5F,
                        false,
                        gx1::ActivationProjectionStrategy::authorization_signal,
                        gx1::ActivationValidationScope::association,
                        gx1::ActivationKeyStrategy::association_centroid,
                        false,
                    });
                std::vector<gx1::LabelConditionedGateEntry> entries;
                for (std::size_t entity = 0; entity < entities.size(); ++entity) {
                    const std::vector<float>* positive_prototype = nullptr;
                    for (std::size_t key = 0; key < memory.keys.size(); ++key) {
                        if (memory.key_associations[key] == entity) {
                            positive_prototype = &memory.keys[key];
                            break;
                        }
                    }
                    if (positive_prototype == nullptr) {
                        throw std::runtime_error(
                            "conditioned verifier has no positive prototype");
                    }
                    std::vector<std::vector<float>> negative_states;
                    for (std::size_t other = 0;
                         other < conditioned_known_states.size();
                         ++other) {
                        if (other != entity) {
                            negative_states.insert(
                                negative_states.end(),
                                conditioned_known_states[other].begin(),
                                conditioned_known_states[other].end());
                        }
                    }
                    for (const auto& unknown : conditioned_unknown_selections) {
                        negative_states.push_back(unknown.second);
                    }
                    const auto negative_prototype =
                        normalized_centroid_for_probe(negative_states, memory);
                    auto maximum_positive_distance = 0.0F;
                    auto minimum_negative_distance =
                        std::numeric_limits<float>::max();
                    auto minimum_positive_gap =
                        std::numeric_limits<float>::max();
                    auto maximum_negative_gap =
                        -std::numeric_limits<float>::max();
                    const auto measurements = [&](const std::vector<float>& state) {
                        const auto query = project_normalized_for_probe(state, memory);
                        const auto positive_distance = probe_squared_distance(
                            query, *positive_prototype);
                        const auto negative_distance = probe_squared_distance(
                            query, negative_prototype);
                        return std::pair<float, float>{
                            positive_distance,
                            negative_distance - positive_distance};
                    };
                    for (const auto& state : conditioned_known_states[entity]) {
                        const auto measured = measurements(state);
                        maximum_positive_distance = std::max(
                            maximum_positive_distance, measured.first);
                        minimum_positive_gap = std::min(
                            minimum_positive_gap, measured.second);
                    }
                    for (const auto& state : negative_states) {
                        const auto measured = measurements(state);
                        minimum_negative_distance = std::min(
                            minimum_negative_distance, measured.first);
                        maximum_negative_gap = std::max(
                            maximum_negative_gap, measured.second);
                    }
                    if (!(maximum_negative_gap < minimum_positive_gap)) {
                        throw std::runtime_error(
                            "conditioned contrastive gaps overlap for entity " +
                            std::to_string(entity));
                    }
                    const auto maximum_distance =
                        maximum_positive_distance +
                        0.5F * std::max(
                                   0.0F,
                                   minimum_negative_distance -
                                       maximum_positive_distance);
                    const auto minimum_margin =
                        maximum_negative_gap +
                        0.5F * (minimum_positive_gap - maximum_negative_gap);
                    entries.push_back({
                        static_cast<std::uint64_t>(entity),
                        *positive_prototype,
                        negative_prototype,
                        maximum_distance,
                        minimum_margin,
                    });
                    std::cout << "conditioned_calibration=" << entities[entity]
                              << "/radius " << maximum_distance
                              << "/margin " << minimum_margin
                              << "/positive_gap " << minimum_positive_gap
                              << "/negative_gap " << maximum_negative_gap
                              << '\n';
                }

                const auto accepts = [&](const std::vector<float>& state,
                                         const std::size_t proposed) {
                    const auto query = project_normalized_for_probe(state, memory);
                    auto nearest_distance = std::numeric_limits<float>::max();
                    std::size_t nearest = 0U;
                    for (std::size_t candidate = 0;
                         candidate < entries.size();
                         ++candidate) {
                        const auto distance = probe_squared_distance(
                            query, entries[candidate].positive_prototype);
                        if (distance < nearest_distance) {
                            nearest_distance = distance;
                            nearest = candidate;
                        }
                    }
                    const auto& entry = entries.at(proposed);
                    const auto positive_distance = probe_squared_distance(
                        query, entry.positive_prototype);
                    const auto negative_distance = probe_squared_distance(
                        query, entry.negative_prototype);
                    return nearest == proposed &&
                           positive_distance <= entry.maximum_distance &&
                           positive_distance + entry.minimum_margin <=
                               negative_distance;
                };
                std::size_t matching_accepts = 0U;
                std::size_t nonmatching_rejections = 0U;
                for (std::size_t entity = 0;
                     entity < conditioned_known_states.size();
                     ++entity) {
                    for (const auto& state : conditioned_known_states[entity]) {
                        matching_accepts += accepts(state, entity) ? 1U : 0U;
                        for (std::size_t other = 0;
                             other < conditioned_known_states.size();
                             ++other) {
                            if (other != entity) {
                                nonmatching_rejections +=
                                    !accepts(state, other) ? 1U : 0U;
                            }
                        }
                    }
                }
                std::size_t unknown_rejections = 0U;
                for (const auto& unknown : conditioned_unknown_selections) {
                    unknown_rejections +=
                        !accepts(unknown.second, unknown.first) ? 1U : 0U;
                }
                if (matching_accepts != 32U ||
                    nonmatching_rejections != 96U ||
                    unknown_rejections != 32U) {
                    throw std::runtime_error(
                        "direct conditioned verifier preflight did not separate: " +
                        std::to_string(matching_accepts) + "/" +
                        std::to_string(nonmatching_rejections) + "/" +
                        std::to_string(unknown_rejections));
                }
                std::cout << "development_width=label-conditioned-knownness/"
                          << width << "/accepted\n";
                return {std::move(memory), std::move(entries)};
            } catch (const std::exception& error) {
                std::cout << "development_width=label-conditioned-knownness/"
                          << width << "/rejected reason=" << error.what()
                          << '\n';
            }
        }
        throw std::runtime_error(
            "no label-conditioned knownness development width separates");
    }();
    const auto& conditioned_memory = conditioned_build.first;
    const auto& conditioned_entries = conditioned_build.second;

    const auto make_conditioned_hook = [&]() {
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_prototype_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_prototype_memory),
            relation_prototype_labels,
            1.0F,
            replication_payloads,
            replication_compatibility_memory.maximum_distance,
            factor_config(replication_compatibility_memory),
            gx1::RetrievalIntentGateConfig{
                factor_config(replication_intent_memory),
                replication_intent_memory.keys.front(),
                replication_intent_calibration.negative_prototype,
                replication_intent_calibration.minimum_margin,
            },
            true,
            std::nullopt,
            gx1::LabelConditionedGateConfig{
                factor_config(conditioned_memory), conditioned_entries});
    };

    std::size_t conditioned_unknown_noops = 0U;
    for (const auto& unknown : conditioned_unknown_probes) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            unknown.prompt,
            make_conditioned_hook(),
            target_states,
            target_tensor,
            action_tensor);
        conditioned_unknown_noops +=
            !memory.hook.known_entity_accepted && !memory.hook.applied &&
                    maximum_logit_difference(
                        unknown.baseline.logits, memory.logits) <= 1.0e-5F
                ? 1U
                : 0U;
    }
    std::cout << "conditioned_development_summary=association "
              << conditioned_association_matches << "/32 matching 32/32"
              << " nonmatching_rejections 96/96 unknown_rejections 32/32"
              << " unknown_noops " << conditioned_unknown_noops << "/32\n";
    if (conditioned_association_matches != 32U ||
        conditioned_unknown_noops != 32U) {
        throw std::runtime_error(
            "label-conditioned knownness failed development preflight");
    }

    std::size_t conditioned_prior_local_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_frozen_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.known_entity_accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    memory.hook.compatibility_accepted &&
                                    !memory.hook.intent_accepted &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied &&
                                    maximum_logit_difference(
                                        baseline.logits,
                                        memory.logits) <= 1.0e-5F;
            conditioned_prior_local_negative_noops += exact_noop ? 1U : 0U;
        }
    }

    std::size_t conditioned_prior_composition_routes = 0U;
    std::size_t conditioned_prior_composition_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed = memory.hook.applied &&
                                memory.hook.known_entity_accepted &&
                                memory.hook.known_entity_identity_consistent &&
                                memory.hook.compatibility_accepted &&
                                memory.hook.intent_accepted &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label == tuple.relation;
            const auto rank = token_rank(memory.logits, tuple.target_token);
            conditioned_prior_composition_routes += routed ? 1U : 0U;
            conditioned_prior_composition_recall +=
                routed && rank == 1U ? 1U : 0U;
            std::cout << "conditioned_prior_positive=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " verifier="
                      << memory.hook.known_entity_verifier_label
                      << " nearest_verifier="
                      << memory.hook.nearest_known_entity_label
                      << " known_entity="
                      << (memory.hook.known_entity_accepted ? "yes" : "no")
                      << " compatibility="
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << " intent="
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << " routed=" << (routed ? "yes" : "no")
                      << " rank=" << rank << '\n';
        }
    }

    std::size_t conditioned_prior_composition_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    memory.hook.known_entity_accepted &&
                                    memory.hook.relation.accepted &&
                                    memory.hook.tuple_found &&
                                    memory.hook.compatibility_accepted &&
                                    !memory.hook.intent_accepted &&
                                    !memory.hook.action_accepted &&
                                    !memory.hook.applied &&
                                    maximum_logit_difference(
                                        baseline.logits,
                                        memory.logits) <= 1.0e-5F;
            conditioned_prior_composition_negative_noops +=
                exact_noop ? 1U : 0U;
        }
    }

    std::size_t conditioned_prior_wrong_intent_noops = 0U;
    std::size_t conditioned_prior_wrong_intent_count = 0U;
    const auto check_conditioned_prior_wrong_intent =
        [&](const std::string& prompt, const InferenceResult& baseline) {
            ++conditioned_prior_wrong_intent_count;
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            conditioned_prior_wrong_intent_noops +=
                !memory.hook.action_accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        };
    for (const auto& negative : wrong_intents) {
        check_conditioned_prior_wrong_intent(
            negative.prompt, negative.baseline);
    }
    for (const auto& tuple : tuples) {
        for (const auto& prompt : authorization_evaluation_negatives(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            check_conditioned_prior_wrong_intent(prompt.second, baseline);
        }
        const auto prompt = gate_composition_negative_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        const auto baseline = capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor);
        check_conditioned_prior_wrong_intent(prompt, baseline);
    }

    std::size_t conditioned_prior_missing_noops = 0U;
    std::size_t conditioned_prior_missing_count = 0U;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 entities[missing.first], relations[missing.second])) {
            ++conditioned_prior_missing_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            conditioned_prior_missing_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        memory.hook.relation.accepted &&
                        !memory.hook.tuple_found && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t conditioned_prior_unknown_entity_noops = 0U;
    std::size_t conditioned_prior_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Vega"), std::string("Altair")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 unknown, "color")) {
            ++conditioned_prior_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto exact_noop = memory.hook.entity.accepted &&
                                    !memory.hook.known_entity_accepted &&
                                    !memory.hook.tuple_found &&
                                    !memory.hook.applied &&
                                    maximum_logit_difference(
                                        baseline.logits,
                                        memory.logits) <= 1.0e-5F;
            conditioned_prior_unknown_entity_noops += exact_noop ? 1U : 0U;
            std::cout << "conditioned_prior_unknown_entity=" << prompt.first
                      << '/' << unknown << "/color"
                      << " verifier="
                      << memory.hook.known_entity_verifier_label
                      << " nearest_verifier="
                      << memory.hook.nearest_known_entity_label
                      << " known_entity="
                      << (memory.hook.known_entity_accepted ? "yes" : "no")
                      << " applied="
                      << (memory.hook.applied ? "yes" : "no") << '\n';
        }
    }

    std::size_t conditioned_prior_unknown_relation_noops = 0U;
    std::size_t conditioned_prior_unknown_relation_count = 0U;
    for (const auto& unknown : {std::string("weight"), std::string("origin")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++conditioned_prior_unknown_relation_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_conditioned_hook(),
                target_states,
                target_tensor,
                action_tensor);
            conditioned_prior_unknown_relation_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        !memory.hook.relation.accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::cout
        << "conditioned_prior_frozen_summary=local_compatibility "
        << prior_local_compatibility_accepts << '/'
        << prior_local_positive_count << " local_cross_rejections "
        << prior_local_cross_rejections << '/' << prior_local_cross_count
        << " local_intent_positive " << prior_local_intent_accepts << '/'
        << prior_local_positive_count << " local_negative_noops "
        << conditioned_prior_local_negative_noops << '/'
        << prior_local_negative_count << " composition_routes "
        << conditioned_prior_composition_routes << '/'
        << prior_composition_count << " composition_rank_one "
        << conditioned_prior_composition_recall << '/'
        << prior_composition_count << " composition_negative_noops "
        << conditioned_prior_composition_negative_noops << '/'
        << prior_composition_negative_count << " older_negative_noops "
        << conditioned_prior_wrong_intent_noops << '/'
        << conditioned_prior_wrong_intent_count << " missing_noops "
        << conditioned_prior_missing_noops << '/'
        << conditioned_prior_missing_count << " unknown_entity_noops "
        << conditioned_prior_unknown_entity_noops << '/'
        << conditioned_prior_unknown_entity_count << " unknown_relation_noops "
        << conditioned_prior_unknown_relation_noops << '/'
        << conditioned_prior_unknown_relation_count << '\n';
    const auto conditioned_prior_frozen_passed =
        prior_local_compatibility_accepts == prior_local_positive_count &&
        prior_local_cross_rejections == prior_local_cross_count &&
        prior_local_intent_accepts == prior_local_positive_count &&
        conditioned_prior_local_negative_noops ==
            prior_local_negative_count &&
        conditioned_prior_composition_routes == prior_composition_count &&
        conditioned_prior_composition_recall == prior_composition_count &&
        conditioned_prior_composition_negative_noops ==
            prior_composition_negative_count &&
        conditioned_prior_wrong_intent_noops ==
            conditioned_prior_wrong_intent_count &&
        conditioned_prior_missing_noops == conditioned_prior_missing_count &&
        conditioned_prior_unknown_entity_noops ==
            conditioned_prior_unknown_entity_count &&
        conditioned_prior_unknown_relation_noops ==
            conditioned_prior_unknown_relation_count;
    const auto historical_conditioned_regression_reproduced =
        !conditioned_prior_frozen_passed &&
        prior_local_compatibility_accepts == prior_local_positive_count &&
        prior_local_cross_rejections == prior_local_cross_count &&
        prior_local_intent_accepts == prior_local_positive_count &&
        conditioned_prior_local_negative_noops ==
            prior_local_negative_count &&
        conditioned_prior_composition_routes == 10U &&
        conditioned_prior_composition_recall == 10U &&
        conditioned_prior_composition_negative_noops == 17U &&
        conditioned_prior_wrong_intent_noops ==
            conditioned_prior_wrong_intent_count &&
        conditioned_prior_missing_noops == conditioned_prior_missing_count &&
        conditioned_prior_unknown_entity_noops == 1U &&
        conditioned_prior_unknown_relation_noops ==
            conditioned_prior_unknown_relation_count;
    if (!historical_conditioned_regression_reproduced) {
        throw std::runtime_error(
            "historical corpus-disjoint conditioned result changed");
    }

    const auto make_joint_entity_hook = [&]() {
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_prototype_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_prototype_memory),
            relation_prototype_labels,
            1.0F,
            replication_payloads,
            replication_compatibility_memory.maximum_distance,
            factor_config(replication_compatibility_memory),
            gx1::RetrievalIntentGateConfig{
                factor_config(replication_intent_memory),
                replication_intent_memory.keys.front(),
                replication_intent_calibration.negative_prototype,
                replication_intent_calibration.minimum_margin,
            },
            true,
            std::nullopt,
            gx1::LabelConditionedGateConfig{
                factor_config(conditioned_memory), conditioned_entries},
            2U);
    };

    std::size_t joint_development_top_two = 0U;
    std::size_t joint_development_selected = 0U;
    std::size_t joint_development_finite = 0U;
    std::size_t joint_development_known_count = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : joint_entity_development_prompts(
                     entities[entity], relation)) {
                ++joint_development_known_count;
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_joint_entity_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto in_top_two = std::find(
                    memory.hook.entity_candidate_labels.begin(),
                    memory.hook.entity_candidate_labels.end(),
                    entity) != memory.hook.entity_candidate_labels.end();
                const auto selected = memory.hook.entity.accepted &&
                                      memory.hook.known_entity_accepted &&
                                      memory.hook.entity.factor_label == entity;
                const auto finite =
                    memory.hook.entity_candidate_labels.size() == 2U &&
                    memory.hook.entity_candidate_distances.size() == 2U &&
                    std::all_of(
                        memory.hook.entity_candidate_distances.begin(),
                        memory.hook.entity_candidate_distances.end(),
                        [](const float value) { return std::isfinite(value); }) &&
                    std::isfinite(memory.hook.entity_joint_score) &&
                    std::isfinite(memory.hook.known_entity_distance) &&
                    std::isfinite(memory.hook.unknown_entity_distance);
                joint_development_top_two += in_top_two ? 1U : 0U;
                joint_development_selected += selected ? 1U : 0U;
                joint_development_finite += finite ? 1U : 0U;
                std::cout << "joint_development_known=" << prompt.first << '/'
                          << entities[entity] << '/' << relation
                          << " top_two=" << (in_top_two ? "yes" : "no")
                          << " selected=" << (selected ? "yes" : "no")
                          << " selected_label="
                          << memory.hook.entity.factor_label
                          << " joint_score=" << memory.hook.entity_joint_score
                          << '\n';
            }
        }
    }

    std::size_t joint_development_unknown_rejections = 0U;
    std::size_t joint_development_unknown_noops = 0U;
    std::size_t joint_development_unknown_finite = 0U;
    std::size_t joint_development_unknown_count = 0U;
    for (const auto& unknown : {std::string("Pollux"), std::string("Castor"),
                                std::string("Alpheratz"),
                                std::string("Mirfak")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : joint_entity_development_prompts(
                     unknown, relation)) {
                ++joint_development_unknown_count;
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_joint_entity_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto rejected = !memory.hook.entity.accepted &&
                                      !memory.hook.known_entity_accepted;
                const auto exact_noop = rejected && !memory.hook.applied &&
                                        maximum_logit_difference(
                                            baseline.logits,
                                            memory.logits) <= 1.0e-5F;
                const auto finite =
                    memory.hook.entity_candidate_labels.size() == 2U &&
                    memory.hook.entity_candidate_distances.size() == 2U &&
                    std::all_of(
                        memory.hook.entity_candidate_distances.begin(),
                        memory.hook.entity_candidate_distances.end(),
                        [](const float value) { return std::isfinite(value); });
                joint_development_unknown_rejections += rejected ? 1U : 0U;
                joint_development_unknown_noops += exact_noop ? 1U : 0U;
                joint_development_unknown_finite += finite ? 1U : 0U;
            }
        }
    }

    std::cout << "joint_development_summary=top_two "
              << joint_development_top_two << '/'
              << joint_development_known_count << " selected "
              << joint_development_selected << '/'
              << joint_development_known_count << " known_finite "
              << joint_development_finite << '/'
              << joint_development_known_count << " unknown_rejections "
              << joint_development_unknown_rejections << '/'
              << joint_development_unknown_count << " unknown_noops "
              << joint_development_unknown_noops << '/'
              << joint_development_unknown_count << " unknown_finite "
              << joint_development_unknown_finite << '/'
              << joint_development_unknown_count << '\n';
    const auto historical_joint_result_reproduced =
        joint_development_top_two == joint_development_known_count &&
        joint_development_selected == 28U &&
        joint_development_finite == joint_development_known_count &&
        joint_development_unknown_rejections ==
            joint_development_unknown_count &&
        joint_development_unknown_noops == joint_development_unknown_count &&
        joint_development_unknown_finite == joint_development_unknown_count &&
        conditioned_association_matches == 32U &&
        conditioned_unknown_noops == 32U &&
        replication_negative_noops ==
            replication_development_negatives.size();
    if (!historical_joint_result_reproduced) {
        throw std::runtime_error(
            "historical joint top-k development result changed");
    }

    std::vector<gx1::ActivationMemoryConstructionView>
        sequence_evidence_construction;
    std::size_t sequence_construction_association_matches = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : sequence_evidence_construction_prompts(
                     entities[entity], relation)) {
                auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                auto association = nearest_factor_selection_for_probe(
                    baseline.token_states,
                    entity_address_association_memory);
                sequence_construction_association_matches +=
                    association.first == entity ? 1U : 0U;
                sequence_evidence_construction.push_back({
                    entity,
                    std::move(baseline.token_states),
                    association.second,
                    association.second,
                });
            }
        }
    }

    std::vector<gx1::ActivationMemoryValidationView>
        sequence_evidence_calibration_positives;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : sequence_evidence_calibration_prompts(
                     entities[entity], relation)) {
                auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                sequence_evidence_calibration_positives.push_back({
                    entity, std::move(baseline.token_states)});
            }
        }
    }

    std::vector<gx1::ActivationMemoryCalibrationView>
        sequence_evidence_calibration_unknowns;
    for (const auto& unknown : {std::string("Alhena"), std::string("Sadr"),
                                std::string("Zosma"), std::string("Kochab")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : sequence_evidence_calibration_prompts(
                     unknown, relation)) {
                auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                sequence_evidence_calibration_unknowns.push_back({
                    std::nullopt, std::move(baseline.token_states)});
            }
        }
    }

    build_stage("sequence-entity-evidence");
    const auto sequence_evidence_memory = gx1::ActivationMemoryBuilder::build(
        sequence_evidence_construction,
        sequence_evidence_calibration_unknowns,
        sequence_evidence_calibration_positives,
        gx1::ActivationMemoryBuildConfig{
            64U,
            0.5F,
            false,
            gx1::ActivationProjectionStrategy::association_signal,
            gx1::ActivationValidationScope::global,
            gx1::ActivationKeyStrategy::selected_views,
            true,
        });
    if (!(sequence_evidence_memory.maximum_negative_group_margin <
          sequence_evidence_memory.minimum_validation_group_margin)) {
        throw std::runtime_error(
            "sequence entity evidence identity gaps do not separate");
    }
    const auto sequence_evidence_identity_margin =
        sequence_evidence_memory.maximum_negative_group_margin +
        0.5F *
            (sequence_evidence_memory.minimum_validation_group_margin -
             sequence_evidence_memory.maximum_negative_group_margin);
    if (!(sequence_evidence_identity_margin > 0.0F) ||
        !std::isfinite(sequence_evidence_identity_margin)) {
        throw std::runtime_error(
            "sequence entity evidence identity margin is invalid");
    }
    std::vector<std::size_t> sequence_prototypes_per_entity(
        entities.size(), 0U);
    for (const auto label : sequence_evidence_memory.key_associations) {
        if (label >= sequence_prototypes_per_entity.size()) {
            throw std::runtime_error(
                "sequence entity evidence produced an unknown label");
        }
        ++sequence_prototypes_per_entity[label];
    }
    const auto sequence_prototype_count_valid = std::all_of(
        sequence_prototypes_per_entity.begin(),
        sequence_prototypes_per_entity.end(),
        [](const std::size_t count) { return count == 6U; });
    std::cout << "sequence_evidence_calibration=construction_association "
              << sequence_construction_association_matches << "/24"
              << " prototypes " << sequence_evidence_memory.keys.size()
              << "/24 radius " << sequence_evidence_memory.maximum_distance
              << " hardest_positive "
              << sequence_evidence_memory.maximum_validation_distance
              << " nearest_unknown "
              << sequence_evidence_memory.minimum_negative_distance
              << " identity_margin " << sequence_evidence_identity_margin
              << " weakest_positive_gap "
              << sequence_evidence_memory.minimum_validation_group_margin
              << " strongest_unknown_gap "
              << sequence_evidence_memory.maximum_negative_group_margin
              << '\n';
    if (sequence_construction_association_matches != 24U ||
        !sequence_prototype_count_valid ||
        sequence_evidence_memory.validation_selections.size() != 16U ||
        sequence_evidence_memory.negative_selections.size() != 16U) {
        throw std::runtime_error(
            "sequence entity evidence calibration contract failed");
    }

    const auto make_sequence_evidence_hook = [&]() {
        generations.activate(entity_address_association_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_prototype_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_address_association_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_prototype_memory),
            relation_prototype_labels,
            1.0F,
            replication_payloads,
            replication_compatibility_memory.maximum_distance,
            factor_config(replication_compatibility_memory),
            gx1::RetrievalIntentGateConfig{
                factor_config(replication_intent_memory),
                replication_intent_memory.keys.front(),
                replication_intent_calibration.negative_prototype,
                replication_intent_calibration.minimum_margin,
            },
            true,
            std::nullopt,
            std::nullopt,
            4U,
            gx1::SequenceEntityEvidenceConfig{
                factor_config(sequence_evidence_memory),
                sequence_evidence_memory.keys,
                std::vector<std::uint64_t>(
                    sequence_evidence_memory.key_associations.begin(),
                    sequence_evidence_memory.key_associations.end()),
                sequence_evidence_identity_margin,
            });
    };

    const auto diagnostics_finite = [&](const gx1::FactorizedMemoryResult& result) {
        return result.entity_candidate_labels.size() == entities.size() &&
               result.entity_candidate_distances.size() == entities.size() &&
               result.entity_evidence_diagnostics.size() == entities.size() &&
               std::all_of(
                   result.entity_evidence_diagnostics.begin(),
                   result.entity_evidence_diagnostics.end(),
                   [](const gx1::EntityEvidenceDiagnostic& diagnostic) {
                       return std::isfinite(diagnostic.association_distance) &&
                              std::isfinite(diagnostic.evidence_distance) &&
                              std::isfinite(diagnostic.competitor_distance) &&
                              std::isfinite(diagnostic.identity_gap) &&
                              std::isfinite(diagnostic.joint_score);
                   });
    };
    const auto print_sequence_evidence_diagnostics = [](
        const std::string& prefix,
        const gx1::FactorizedMemoryResult& result) {
        for (const auto& diagnostic : result.entity_evidence_diagnostics) {
            std::cout << prefix << "/label " << diagnostic.label
                      << "/association " << diagnostic.association_distance
                      << "/association_ok "
                      << (diagnostic.association_accepted ? "yes" : "no")
                      << "/evidence " << diagnostic.evidence_distance
                      << "/radius_ok "
                      << (diagnostic.radius_accepted ? "yes" : "no")
                      << "/competitor " << diagnostic.competitor_distance
                      << "/gap " << diagnostic.identity_gap << "/margin_ok "
                      << (diagnostic.margin_accepted ? "yes" : "no")
                      << "/association_state " << diagnostic.association_state
                      << "/evidence_state " << diagnostic.evidence_state
                      << "/prototype " << diagnostic.evidence_prototype
                      << "/score " << diagnostic.joint_score << "/eligible "
                      << (diagnostic.eligible ? "yes" : "no") << '\n';
        }
    };

    std::size_t sequence_development_top_two = 0U;
    std::size_t sequence_development_selected = 0U;
    std::size_t sequence_development_finite = 0U;
    std::size_t sequence_development_known_count = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : sequence_evidence_development_prompts(
                     entities[entity], relation)) {
                ++sequence_development_known_count;
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto in_top_two = std::find(
                    memory.hook.entity_candidate_labels.begin(),
                    memory.hook.entity_candidate_labels.end(),
                    entity) != memory.hook.entity_candidate_labels.end();
                const auto selected = memory.hook.entity.accepted &&
                                      memory.hook.known_entity_accepted &&
                                      memory.hook.entity.factor_label == entity;
                const auto finite = diagnostics_finite(memory.hook);
                sequence_development_top_two += in_top_two ? 1U : 0U;
                sequence_development_selected += selected ? 1U : 0U;
                sequence_development_finite += finite ? 1U : 0U;
                std::cout << "sequence_development_known=" << prompt.first
                          << '/' << entities[entity] << '/' << relation
                          << " top_two=" << (in_top_two ? "yes" : "no")
                          << " selected=" << (selected ? "yes" : "no")
                          << " selected_label="
                          << memory.hook.entity.factor_label << '\n';
                for (const auto& diagnostic :
                     memory.hook.entity_evidence_diagnostics) {
                    std::cout << "sequence_development_candidate="
                              << prompt.first << '/' << entities[entity] << '/'
                              << relation << "/label " << diagnostic.label
                              << "/association "
                              << diagnostic.association_distance
                              << "/evidence " << diagnostic.evidence_distance
                              << "/competitor "
                              << diagnostic.competitor_distance << "/gap "
                              << diagnostic.identity_gap << "/score "
                              << diagnostic.joint_score << "/eligible "
                              << (diagnostic.eligible ? "yes" : "no") << '\n';
                }
            }
        }
    }

    std::size_t sequence_development_unknown_rejections = 0U;
    std::size_t sequence_development_unknown_noops = 0U;
    std::size_t sequence_development_unknown_finite = 0U;
    std::size_t sequence_development_unknown_count = 0U;
    for (const auto& unknown : {std::string("Hamal"), std::string("Markab"),
                                std::string("Kaus"), std::string("Ankaa")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : sequence_evidence_development_prompts(
                     unknown, relation)) {
                ++sequence_development_unknown_count;
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto rejected = !memory.hook.entity.accepted &&
                                      !memory.hook.known_entity_accepted;
                const auto exact_noop = rejected && !memory.hook.applied &&
                                        maximum_logit_difference(
                                            baseline.logits,
                                            memory.logits) <= 1.0e-5F;
                const auto finite = diagnostics_finite(memory.hook);
                sequence_development_unknown_rejections += rejected ? 1U : 0U;
                sequence_development_unknown_noops += exact_noop ? 1U : 0U;
                sequence_development_unknown_finite += finite ? 1U : 0U;
                std::cout << "sequence_development_unknown=" << prompt.first
                          << '/' << unknown << '/' << relation
                          << " rejected=" << (rejected ? "yes" : "no")
                          << " exact_noop=" << (exact_noop ? "yes" : "no")
                          << '\n';
                for (const auto& diagnostic :
                     memory.hook.entity_evidence_diagnostics) {
                    std::cout << "sequence_development_unknown_candidate="
                              << prompt.first << '/' << unknown << '/'
                              << relation << "/label " << diagnostic.label
                              << "/association "
                              << diagnostic.association_distance
                              << "/evidence " << diagnostic.evidence_distance
                              << "/competitor "
                              << diagnostic.competitor_distance << "/gap "
                              << diagnostic.identity_gap << "/score "
                              << diagnostic.joint_score << "/eligible "
                              << (diagnostic.eligible ? "yes" : "no") << '\n';
                }
            }
        }
    }

    std::cout << "sequence_development_summary=top_two "
              << sequence_development_top_two << '/'
              << sequence_development_known_count << " selected "
              << sequence_development_selected << '/'
              << sequence_development_known_count << " known_finite "
              << sequence_development_finite << '/'
              << sequence_development_known_count << " unknown_rejections "
              << sequence_development_unknown_rejections << '/'
              << sequence_development_unknown_count << " unknown_noops "
              << sequence_development_unknown_noops << '/'
              << sequence_development_unknown_count << " unknown_finite "
              << sequence_development_unknown_finite << '/'
              << sequence_development_unknown_count << '\n';
    if (sequence_development_top_two != sequence_development_known_count ||
        sequence_development_selected != sequence_development_known_count ||
        sequence_development_finite != sequence_development_known_count ||
        sequence_development_unknown_rejections !=
            sequence_development_unknown_count ||
        sequence_development_unknown_noops !=
            sequence_development_unknown_count ||
        sequence_development_unknown_finite !=
            sequence_development_unknown_count) {
        throw std::runtime_error(
            "sequence entity evidence failed development preflight");
    }

    std::size_t sequence_prior_joint_selected = 0U;
    std::size_t sequence_prior_joint_known_count = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : joint_entity_development_prompts(
                     entities[entity], relation)) {
                ++sequence_prior_joint_known_count;
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                sequence_prior_joint_selected +=
                    memory.hook.entity.accepted &&
                            memory.hook.known_entity_accepted &&
                            memory.hook.entity.factor_label == entity
                        ? 1U
                        : 0U;
                std::cout << "adaptive_prior_joint=" << prompt.first << '/'
                          << entities[entity] << '/' << relation
                          << "/selected " << memory.hook.entity.factor_label
                          << "/accepted "
                          << (memory.hook.known_entity_accepted ? "yes" : "no")
                          << '\n';
                print_sequence_evidence_diagnostics(
                    "adaptive_prior_joint_candidate=" + prompt.first + '/' +
                        entities[entity] + '/' + relation,
                    memory.hook);
            }
        }
    }
    std::size_t sequence_prior_joint_unknown_noops = 0U;
    std::size_t sequence_prior_joint_unknown_count = 0U;
    for (const auto& unknown : {std::string("Pollux"), std::string("Castor"),
                                std::string("Alpheratz"),
                                std::string("Mirfak")}) {
        for (const auto& relation : relations) {
            for (const auto& prompt : joint_entity_development_prompts(
                     unknown, relation)) {
                ++sequence_prior_joint_unknown_count;
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                sequence_prior_joint_unknown_noops +=
                    !memory.hook.known_entity_accepted &&
                            !memory.hook.applied &&
                            maximum_logit_difference(
                                baseline.logits, memory.logits) <= 1.0e-5F
                        ? 1U
                        : 0U;
            }
        }
    }

    std::size_t sequence_prior_conditioned_selected = 0U;
    std::size_t sequence_prior_conditioned_known_count = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : conditioned_knownness_development_prompts(
                     entities[entity], relation)) {
                ++sequence_prior_conditioned_known_count;
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                sequence_prior_conditioned_selected +=
                    memory.hook.entity.accepted &&
                            memory.hook.known_entity_accepted &&
                            memory.hook.entity.factor_label == entity
                        ? 1U
                        : 0U;
                print_sequence_evidence_diagnostics(
                    "adaptive_prior_conditioned_candidate=" + prompt.first +
                        '/' + entities[entity] + '/' + relation,
                    memory.hook);
            }
        }
    }
    std::size_t sequence_prior_conditioned_unknown_noops = 0U;
    for (const auto& unknown : conditioned_unknown_probes) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            unknown.prompt,
            make_sequence_evidence_hook(),
            target_states,
            target_tensor,
            action_tensor);
        sequence_prior_conditioned_unknown_noops +=
            !memory.hook.known_entity_accepted && !memory.hook.applied &&
                    maximum_logit_difference(
                        unknown.baseline.logits, memory.logits) <= 1.0e-5F
                ? 1U
                : 0U;
    }

    std::size_t sequence_prior_local_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_frozen_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_local_negative_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        memory.hook.relation.accepted &&
                        memory.hook.tuple_found &&
                        memory.hook.compatibility_accepted &&
                        !memory.hook.intent_accepted &&
                        !memory.hook.action_accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t sequence_prior_composition_routes = 0U;
    std::size_t sequence_prior_composition_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed = memory.hook.applied &&
                                memory.hook.known_entity_accepted &&
                                memory.hook.compatibility_accepted &&
                                memory.hook.intent_accepted &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label == tuple.relation;
            const auto rank = token_rank(memory.logits, tuple.target_token);
            sequence_prior_composition_routes += routed ? 1U : 0U;
            sequence_prior_composition_recall +=
                routed && rank == 1U ? 1U : 0U;
            std::cout << "sequence_prior_positive=" << prompt.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation] << " routed="
                      << (routed ? "yes" : "no") << " rank=" << rank << '\n';
            print_sequence_evidence_diagnostics(
                "adaptive_prior_positive_candidate=" + prompt.first + '/' +
                    entities[tuple.entity] + '/' + relations[tuple.relation],
                memory.hook);
        }
    }

    std::size_t sequence_prior_composition_negative_noops = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : conjunctive_composition_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_composition_negative_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        memory.hook.relation.accepted &&
                        memory.hook.tuple_found &&
                        memory.hook.compatibility_accepted &&
                        !memory.hook.intent_accepted &&
                        !memory.hook.action_accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t sequence_prior_wrong_intent_noops = 0U;
    std::size_t sequence_prior_wrong_intent_count = 0U;
    const auto check_sequence_prior_wrong_intent =
        [&](const std::string& prompt, const InferenceResult& baseline) {
            ++sequence_prior_wrong_intent_count;
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_wrong_intent_noops +=
                !memory.hook.action_accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        };
    for (const auto& negative : wrong_intents) {
        check_sequence_prior_wrong_intent(
            negative.prompt, negative.baseline);
    }
    for (const auto& tuple : tuples) {
        for (const auto& prompt : authorization_evaluation_negatives(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            check_sequence_prior_wrong_intent(prompt.second, baseline);
        }
        const auto prompt = gate_composition_negative_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        const auto baseline = capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor);
        check_sequence_prior_wrong_intent(prompt, baseline);
    }

    std::size_t sequence_prior_missing_noops = 0U;
    std::size_t sequence_prior_missing_count = 0U;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 entities[missing.first], relations[missing.second])) {
            ++sequence_prior_missing_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_missing_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        memory.hook.relation.accepted &&
                        !memory.hook.tuple_found && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t sequence_prior_unknown_entity_noops = 0U;
    std::size_t sequence_prior_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Vega"), std::string("Altair")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 unknown, "color")) {
            ++sequence_prior_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_unknown_entity_noops +=
                !memory.hook.known_entity_accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::size_t sequence_prior_unknown_relation_noops = 0U;
    std::size_t sequence_prior_unknown_relation_count = 0U;
    for (const auto& unknown : {std::string("weight"), std::string("origin")}) {
        for (const auto& prompt : gate_composition_evaluation_prompts(
                 "Arcturus", unknown)) {
            ++sequence_prior_unknown_relation_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            sequence_prior_unknown_relation_noops +=
                memory.hook.entity.accepted &&
                        memory.hook.known_entity_accepted &&
                        !memory.hook.relation.accepted && !memory.hook.applied &&
                        maximum_logit_difference(
                            baseline.logits, memory.logits) <= 1.0e-5F
                    ? 1U
                    : 0U;
        }
    }

    std::cout << "sequence_prior_summary=joint_known "
              << sequence_prior_joint_selected << '/'
              << sequence_prior_joint_known_count << " joint_unknown_noops "
              << sequence_prior_joint_unknown_noops << '/'
              << sequence_prior_joint_unknown_count << " conditioned_known "
              << sequence_prior_conditioned_selected << '/'
              << sequence_prior_conditioned_known_count
              << " conditioned_unknown_noops "
              << sequence_prior_conditioned_unknown_noops << "/32"
              << " local_compatibility " << prior_local_compatibility_accepts
              << '/' << prior_local_positive_count << " local_cross "
              << prior_local_cross_rejections << '/' << prior_local_cross_count
              << " local_intent " << prior_local_intent_accepts << '/'
              << prior_local_positive_count << " local_negative_noops "
              << sequence_prior_local_negative_noops << '/'
              << prior_local_negative_count << " composition_routes "
              << sequence_prior_composition_routes << '/'
              << prior_composition_count << " composition_rank_one "
              << sequence_prior_composition_recall << '/'
              << prior_composition_count << " composition_negative_noops "
              << sequence_prior_composition_negative_noops << '/'
              << prior_composition_negative_count << " older_negative_noops "
              << sequence_prior_wrong_intent_noops << '/'
              << sequence_prior_wrong_intent_count << " missing_noops "
              << sequence_prior_missing_noops << '/'
              << sequence_prior_missing_count << " unknown_entity_noops "
              << sequence_prior_unknown_entity_noops << '/'
              << sequence_prior_unknown_entity_count
              << " unknown_relation_noops "
              << sequence_prior_unknown_relation_noops << '/'
              << sequence_prior_unknown_relation_count << '\n';
    if (sequence_prior_joint_selected != sequence_prior_joint_known_count ||
        sequence_prior_joint_unknown_noops !=
            sequence_prior_joint_unknown_count ||
        sequence_prior_conditioned_selected !=
            sequence_prior_conditioned_known_count ||
        sequence_prior_conditioned_unknown_noops != 32U ||
        prior_local_compatibility_accepts != prior_local_positive_count ||
        prior_local_cross_rejections != prior_local_cross_count ||
        prior_local_intent_accepts != prior_local_positive_count ||
        sequence_prior_local_negative_noops != prior_local_negative_count ||
        sequence_prior_composition_routes != prior_composition_count ||
        sequence_prior_composition_recall != prior_composition_count ||
        sequence_prior_composition_negative_noops !=
            prior_composition_negative_count ||
        sequence_prior_wrong_intent_noops !=
            sequence_prior_wrong_intent_count ||
        sequence_prior_missing_noops != sequence_prior_missing_count ||
        sequence_prior_unknown_entity_noops !=
            sequence_prior_unknown_entity_count ||
        sequence_prior_unknown_relation_noops !=
            sequence_prior_unknown_relation_count) {
        throw std::runtime_error(
            "sequence entity evidence failed one-shot prior regression");
    }

    const auto print_untouched_result =
        [&](const std::string& prefix,
            const MemoryInferenceResult& memory,
            const std::size_t rank,
            const float maximum_delta) {
            std::cout << prefix << "/candidates";
            for (std::size_t index = 0;
                 index < memory.hook.entity_candidate_labels.size();
                 ++index) {
                std::cout << ' ' << memory.hook.entity_candidate_labels[index]
                          << ':'
                          << memory.hook.entity_candidate_distances[index];
            }
            std::cout << "/selected " << memory.hook.entity.factor_label
                      << "/state " << memory.hook.entity.address_candidate
                      << "/known "
                      << (memory.hook.known_entity_accepted ? "yes" : "no")
                      << "/relation " << memory.hook.relation.factor_label
                      << "/tuple "
                      << (memory.hook.tuple_found ? "yes" : "no")
                      << "/compatibility "
                      << (memory.hook.compatibility_accepted ? "yes" : "no")
                      << "/intent "
                      << (memory.hook.intent_accepted ? "yes" : "no")
                      << "/action "
                      << (memory.hook.action_accepted ? "yes" : "no")
                      << "/applied "
                      << (memory.hook.applied ? "yes" : "no")
                      << "/rank " << rank << "/max_delta " << maximum_delta
                      << '\n';
            print_sequence_evidence_diagnostics(
                prefix + "/evidence", memory.hook);
        };

    const std::vector<std::string> untouched_unknown_entities{
        "Rasalhague", "Merak", "Nunki", "Schedar"};
    std::size_t untouched_local_top_two = 0U;
    std::size_t untouched_local_selected = 0U;
    std::size_t untouched_local_finite = 0U;
    std::size_t untouched_local_known_count = 0U;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& relation : relations) {
            for (const auto& prompt : untouched_local_prompts(
                     entities[entity], relation)) {
                ++untouched_local_known_count;
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto top_two_end = memory.hook.entity_candidate_labels.begin() +
                    static_cast<std::ptrdiff_t>(std::min<std::size_t>(
                        2U, memory.hook.entity_candidate_labels.size()));
                const auto in_top_two = std::find(
                    memory.hook.entity_candidate_labels.begin(),
                    top_two_end,
                    entity) != top_two_end;
                const auto selected = memory.hook.entity.accepted &&
                                      memory.hook.known_entity_accepted &&
                                      memory.hook.entity.factor_label == entity;
                const auto finite = diagnostics_finite(memory.hook);
                const auto delta = maximum_logit_difference(
                    baseline.logits, memory.logits);
                untouched_local_top_two += in_top_two ? 1U : 0U;
                untouched_local_selected += selected ? 1U : 0U;
                untouched_local_finite += finite ? 1U : 0U;
                print_untouched_result(
                    "untouched_local_known=" + prompt.first + '/' +
                        entities[entity] + '/' + relation,
                    memory,
                    0U,
                    delta);
            }
        }
    }

    std::size_t untouched_local_unknown_rejections = 0U;
    std::size_t untouched_local_unknown_noops = 0U;
    std::size_t untouched_local_unknown_finite = 0U;
    std::size_t untouched_local_unknown_count = 0U;
    for (const auto& unknown : untouched_unknown_entities) {
        for (const auto& relation : relations) {
            for (const auto& prompt : untouched_local_prompts(unknown, relation)) {
                ++untouched_local_unknown_count;
                const auto baseline = capture(
                    model.get(),
                    vocab,
                    prompt.second,
                    hidden_dimension,
                    target_tensor);
                const auto memory = infer_with_target_state_memory(
                    model.get(),
                    vocab,
                    prompt.second,
                    make_sequence_evidence_hook(),
                    target_states,
                    target_tensor,
                    action_tensor);
                const auto rejected = !memory.hook.known_entity_accepted;
                const auto delta = maximum_logit_difference(
                    baseline.logits, memory.logits);
                const auto noop = rejected && !memory.hook.applied &&
                                  delta <= 1.0e-5F;
                const auto finite = diagnostics_finite(memory.hook);
                untouched_local_unknown_rejections += rejected ? 1U : 0U;
                untouched_local_unknown_noops += noop ? 1U : 0U;
                untouched_local_unknown_finite += finite ? 1U : 0U;
                print_untouched_result(
                    "untouched_local_unknown=" + prompt.first + '/' + unknown +
                        '/' + relation,
                    memory,
                    0U,
                    delta);
            }
        }
    }

    std::cout << "untouched_local_summary=top_two "
              << untouched_local_top_two << '/' << untouched_local_known_count
              << " selected " << untouched_local_selected << '/'
              << untouched_local_known_count << " known_finite "
              << untouched_local_finite << '/' << untouched_local_known_count
              << " unknown_rejections " << untouched_local_unknown_rejections
              << '/' << untouched_local_unknown_count << " unknown_noops "
              << untouched_local_unknown_noops << '/'
              << untouched_local_unknown_count << " unknown_finite "
              << untouched_local_unknown_finite << '/'
              << untouched_local_unknown_count << '\n';
    if (untouched_local_known_count != 16U ||
        untouched_local_unknown_count != 16U ||
        untouched_local_top_two != untouched_local_known_count ||
        untouched_local_selected != untouched_local_known_count ||
        untouched_local_finite != untouched_local_known_count ||
        untouched_local_unknown_rejections != untouched_local_unknown_count ||
        untouched_local_unknown_noops != untouched_local_unknown_count ||
        untouched_local_unknown_finite != untouched_local_unknown_count) {
        throw std::runtime_error("locked untouched local evaluation failed");
    }

    std::size_t untouched_composition_positive_passes = 0U;
    std::size_t untouched_composition_positive_count = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : untouched_composition_positive_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            ++untouched_composition_positive_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto rank = token_rank(memory.logits, tuple.target_token);
            const auto routed = memory.hook.entity.accepted &&
                                memory.hook.known_entity_accepted &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.accepted &&
                                memory.hook.relation.factor_label == tuple.relation &&
                                memory.hook.tuple_found &&
                                memory.hook.compatibility_accepted &&
                                memory.hook.intent_accepted &&
                                memory.hook.action_accepted &&
                                memory.hook.applied;
            const auto passed = routed && rank == 1U &&
                                diagnostics_finite(memory.hook);
            untouched_composition_positive_passes += passed ? 1U : 0U;
            print_untouched_result(
                "untouched_composition_positive=" + prompt.first + '/' +
                    entities[tuple.entity] + '/' + relations[tuple.relation],
                memory,
                rank,
                maximum_logit_difference(baseline.logits, memory.logits));
        }
    }

    std::size_t untouched_composition_negative_noops = 0U;
    std::size_t untouched_composition_negative_count = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& prompt : untouched_composition_negative_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            ++untouched_composition_negative_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt.second,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto noop = memory.hook.entity.accepted &&
                              memory.hook.known_entity_accepted &&
                              memory.hook.relation.accepted &&
                              memory.hook.tuple_found &&
                              memory.hook.compatibility_accepted &&
                              !memory.hook.intent_accepted &&
                              !memory.hook.action_accepted &&
                              !memory.hook.applied && delta <= 1.0e-5F &&
                              diagnostics_finite(memory.hook);
            untouched_composition_negative_noops += noop ? 1U : 0U;
            print_untouched_result(
                "untouched_composition_negative=" + prompt.first + '/' +
                    entities[tuple.entity] + '/' + relations[tuple.relation],
                memory,
                0U,
                delta);
        }
    }

    std::size_t untouched_missing_noops = 0U;
    const auto check_untouched_missing =
        [&](const std::string& name,
            const std::string& prompt,
            const std::size_t entity,
            const std::size_t relation) {
            const auto baseline = capture(
                model.get(), vocab, prompt, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto noop = memory.hook.entity.accepted &&
                              memory.hook.known_entity_accepted &&
                              memory.hook.entity.factor_label == entity &&
                              memory.hook.relation.accepted &&
                              memory.hook.relation.factor_label == relation &&
                              !memory.hook.tuple_found && !memory.hook.applied &&
                              delta <= 1.0e-5F &&
                              diagnostics_finite(memory.hook);
            untouched_missing_noops += noop ? 1U : 0U;
            print_untouched_result(
                "untouched_missing=" + name, memory, 0U, delta);
        };
    check_untouched_missing(
        "archive/Bellatrix/material",
        "archive.lookup(entity=Bellatrix, relation=material) ->",
        1U,
        1U);
    check_untouched_missing(
        "retained-record/Cygnus/color",
        "Fetch the color datum in Cygnus's retained record.\nDatum:",
        2U,
        0U);

    std::size_t untouched_unknown_entity_noops = 0U;
    const auto check_untouched_unknown_entity =
        [&](const std::string& name, const std::string& prompt) {
            const auto baseline = capture(
                model.get(), vocab, prompt, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto noop = !memory.hook.known_entity_accepted &&
                              !memory.hook.applied && delta <= 1.0e-5F &&
                              diagnostics_finite(memory.hook);
            untouched_unknown_entity_noops += noop ? 1U : 0U;
            print_untouched_result(
                "untouched_unknown_entity=" + name, memory, 0U, delta);
        };
    check_untouched_unknown_entity(
        "archive/Rasalhague/color",
        "archive.lookup(entity=Rasalhague, relation=color) ->");
    check_untouched_unknown_entity(
        "retained-record/Merak/material",
        "Fetch the material datum in Merak's retained record.\nDatum:");

    std::size_t untouched_unknown_relation_noops = 0U;
    const auto check_untouched_unknown_relation =
        [&](const std::string& name,
            const std::string& prompt,
            const std::size_t entity) {
            const auto baseline = capture(
                model.get(), vocab, prompt, hidden_dimension, target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                prompt,
                make_sequence_evidence_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto noop = memory.hook.entity.accepted &&
                              memory.hook.known_entity_accepted &&
                              memory.hook.entity.factor_label == entity &&
                              !memory.hook.relation.accepted &&
                              !memory.hook.applied && delta <= 1.0e-5F &&
                              diagnostics_finite(memory.hook);
            untouched_unknown_relation_noops += noop ? 1U : 0U;
            print_untouched_result(
                "untouched_unknown_relation=" + name, memory, 0U, delta);
        };
    check_untouched_unknown_relation(
        "archive/Arcturus/density",
        "archive.lookup(entity=Arcturus, relation=density) ->",
        0U);
    check_untouched_unknown_relation(
        "retained-record/Draco/birthplace",
        "Fetch the birthplace datum in Draco's retained record.\nDatum:",
        3U);

    std::cout << "untouched_composition_summary=positives "
              << untouched_composition_positive_passes << '/'
              << untouched_composition_positive_count << " negative_noops "
              << untouched_composition_negative_noops << '/'
              << untouched_composition_negative_count << " missing_noops "
              << untouched_missing_noops << "/2 unknown_entity_noops "
              << untouched_unknown_entity_noops
              << "/2 unknown_relation_noops "
              << untouched_unknown_relation_noops << "/2\n";
    if (untouched_composition_positive_count != 12U ||
        untouched_composition_negative_count != 18U ||
        untouched_composition_positive_passes !=
            untouched_composition_positive_count ||
        untouched_composition_negative_noops !=
            untouched_composition_negative_count ||
        untouched_missing_noops != 2U ||
        untouched_unknown_entity_noops != 2U ||
        untouched_unknown_relation_noops != 2U) {
        throw std::runtime_error(
            "locked untouched composition evaluation failed");
    }

    std::cout << "stored_tuples=" << tuples.size()
              << " registered_action_views=" << action_view_specs.size()
              << " development_action_views=" << action_positives.size()
              << " wrong_intent_controls=" << wrong_intents.size()
              << " evaluation_routes=" << evaluation_routed << '/'
              << tuples.size() * 2U
              << " evaluation_recall=" << evaluation_recalled << '/'
              << tuples.size() * 2U
              << " two_stage_evaluation_routes="
              << two_stage_evaluation_routes << '/'
              << two_stage_evaluation_count
              << " residual_evaluation_recall="
              << residual_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_evaluation_recall="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " entity_address_evaluation_routes="
              << entity_address_association_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " entity_address_evaluation_recall="
              << entity_address_association_evaluation_recall << '/'
              << entity_address_evaluation_count
              << " missing_known_factor_tuples=2\n"
              << "factorized memory experiment passed\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gx1_llama_factorized_memory_experiment MODEL.gguf\n";
        return EXIT_FAILURE;
    }
    llama_log_set(&model_log, nullptr);
    llama_backend_init();
    try {
        const auto status = run(argv[1]);
        llama_backend_free();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "factorized memory experiment failed: " << error.what() << '\n';
        llama_backend_free();
        return EXIT_FAILURE;
    }
}
