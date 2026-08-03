#include "gx1/glamin_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gx1 {

struct GlaminRuntimeState {
    explicit GlaminRuntimeState(const glamin_runtime_t native_handle)
        : handle(native_handle) {}

    ~GlaminRuntimeState() {
        if (handle != 0) {
            static_cast<void>(glamin_runtime_destroy(handle));
        }
    }

    glamin_runtime_t handle{0};
};

struct GlaminGenerationResource {
    std::unique_ptr<GlaminFlatIndex> index;
    GlaminGenerationId generation{0};
    std::string label;
};

struct GlaminGenerationStoreState {
    explicit GlaminGenerationStoreState(std::shared_ptr<GlaminRuntimeState> runtime_state)
        : runtime(std::move(runtime_state)) {}

    ~GlaminGenerationStoreState() {
        if (!runtime || runtime->handle == 0) {
            return;
        }
        static_cast<void>(glamin_generation_deactivate(runtime->handle));
        active_generation = 0;
        for (const auto& entry : generations) {
            static_cast<void>(
                glamin_generation_retire(runtime->handle, entry.second->generation));
        }
        generations.clear();
    }

    std::shared_ptr<GlaminRuntimeState> runtime;
    std::unordered_map<GlaminGenerationId, std::shared_ptr<GlaminGenerationResource>>
        generations;
    GlaminGenerationId active_generation{0};
};

namespace {

std::string read_diagnostic(const glamin_runtime_t runtime) {
    std::uint64_t required = 0;
    const auto size_status = glamin_last_error(runtime, nullptr, 0, &required);
    if (size_status != GLAMIN_STATUS_BUFFER_TOO_SMALL || required == 0) {
        return "Glamin did not provide a diagnostic";
    }

    std::vector<char> buffer(static_cast<std::size_t>(required));
    const auto read_status = glamin_last_error(runtime, buffer.data(), required, &required);
    if (read_status != GLAMIN_STATUS_OK) {
        return "Glamin diagnostic retrieval failed";
    }
    return std::string(buffer.data());
}

glamin_metric native_metric(const GlaminMetric metric) {
    switch (metric) {
    case GlaminMetric::l2:
        return GLAMIN_METRIC_L2;
    case GlaminMetric::inner_product:
        return GLAMIN_METRIC_INNER_PRODUCT;
    }
    throw std::invalid_argument("unsupported Glamin metric");
}

std::uint64_t dense_vector_count(
    const std::vector<float>& values,
    const std::uint32_t dimension,
    const char* value_name) {
    const auto native_dimension = static_cast<std::size_t>(dimension);
    if (values.empty() || native_dimension == 0 || values.size() % native_dimension != 0) {
        throw std::invalid_argument(
            std::string(value_name) + " must contain complete, nonempty vectors");
    }
    return static_cast<std::uint64_t>(values.size() / native_dimension);
}

std::size_t result_element_count(
    const std::uint64_t query_count,
    const std::uint32_t k) {
    if (k == 0) {
        throw std::invalid_argument("search k must be positive");
    }
    const auto native_k = static_cast<std::size_t>(k);
    if (query_count > std::numeric_limits<std::size_t>::max() / native_k) {
        throw std::length_error("search result is too large for this process");
    }
    return static_cast<std::size_t>(query_count) * native_k;
}

void validate_generation_label(const std::string& label) {
    if (label.empty() || label.size() > 128 || label.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            "generation label must contain 1 to 128 non-null bytes");
    }
}

void validate_abi_text(
    const std::string& value,
    const std::size_t maximum_length,
    const char* value_name) {
    if (value.empty() || value.size() > maximum_length ||
        value.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            std::string(value_name) + " has an invalid length or contains a null byte");
    }
}

} // namespace

GlaminError::GlaminError(const glamin_status status, std::string message)
    : std::runtime_error(std::move(message)), status_(status) {}

glamin_status GlaminError::status() const noexcept {
    return status_;
}

GlaminRuntime::GlaminRuntime(const std::uint32_t worker_count) {
    glamin_runtime_t native_runtime = 0;
    const auto status = glamin_runtime_create(worker_count, &native_runtime);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(0));
    }
    state_ = std::make_shared<GlaminRuntimeState>(native_runtime);
}

GlaminRuntime::~GlaminRuntime() = default;

GlaminRuntime::GlaminRuntime(GlaminRuntime&& other) noexcept = default;

GlaminRuntime& GlaminRuntime::operator=(GlaminRuntime&& other) noexcept = default;

std::uint32_t GlaminRuntime::abi_version() noexcept {
    return glamin_abi_version();
}

std::uint64_t GlaminRuntime::native_handle() const noexcept {
    return state_ ? state_->handle : 0;
}

bool GlaminRuntime::is_open() const noexcept {
    return native_handle() != 0;
}

void GlaminRuntime::close() {
    if (!is_open()) {
        return;
    }

    const auto status = glamin_runtime_destroy(state_->handle);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(state_->handle));
    }
    state_->handle = 0;
}

GlaminFlatIndex::GlaminFlatIndex(
    GlaminRuntime& runtime,
    const std::uint32_t dimension,
    const GlaminMetric metric)
    : GlaminFlatIndex(runtime.state_, dimension, metric) {}

GlaminFlatIndex::GlaminFlatIndex(
    std::shared_ptr<GlaminRuntimeState> runtime_state,
    const std::uint32_t dimension,
    const GlaminMetric metric)
    : runtime_state_(std::move(runtime_state)), dimension_(dimension) {
    if (!runtime_state_ || runtime_state_->handle == 0) {
        throw std::invalid_argument("flat index requires an open Glamin runtime");
    }

    glamin_index_t native_index = 0;
    const auto status = glamin_flat_index_create(
        runtime_state_->handle,
        dimension,
        native_metric(metric),
        &native_index);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime_state_->handle));
    }
    handle_ = native_index;
}

GlaminFlatIndex::GlaminFlatIndex(
    std::shared_ptr<GlaminRuntimeState> runtime_state,
    const std::uint64_t native_handle,
    const std::uint32_t dimension,
    const std::uint64_t vector_count)
    : runtime_state_(std::move(runtime_state)),
      handle_(native_handle),
      vector_count_(vector_count),
      dimension_(dimension) {
    if (!runtime_state_ || runtime_state_->handle == 0 || native_handle == 0 ||
        dimension == 0 || vector_count == 0) {
        throw std::invalid_argument("loaded flat index metadata is invalid");
    }
}

GlaminFlatIndex::~GlaminFlatIndex() {
    close_without_throwing();
}

GlaminFlatIndex::GlaminFlatIndex(GlaminFlatIndex&& other) noexcept
    : runtime_state_(std::move(other.runtime_state_)),
      handle_(std::exchange(other.handle_, 0)),
      vector_count_(std::exchange(other.vector_count_, 0)),
      dimension_(std::exchange(other.dimension_, 0)) {}

GlaminFlatIndex& GlaminFlatIndex::operator=(GlaminFlatIndex&& other) noexcept {
    if (this != &other) {
        close_without_throwing();
        runtime_state_ = std::move(other.runtime_state_);
        handle_ = std::exchange(other.handle_, 0);
        vector_count_ = std::exchange(other.vector_count_, 0);
        dimension_ = std::exchange(other.dimension_, 0);
    }
    return *this;
}

std::uint64_t GlaminFlatIndex::native_handle() const noexcept {
    return handle_;
}

std::uint32_t GlaminFlatIndex::dimension() const noexcept {
    return dimension_;
}

std::uint64_t GlaminFlatIndex::vector_count() const noexcept {
    return vector_count_;
}

bool GlaminFlatIndex::is_open() const noexcept {
    return handle_ != 0;
}

void GlaminFlatIndex::add(const std::vector<float>& vectors) {
    if (!is_open()) {
        throw std::logic_error("cannot add to a closed Glamin index");
    }

    const auto incoming_count = dense_vector_count(vectors, dimension_, "vectors");
    const auto status = glamin_index_add_f32(
        runtime_state_->handle,
        handle_,
        vectors.data(),
        incoming_count,
        dimension_);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime_state_->handle));
    }
    vector_count_ += incoming_count;
}

GlaminSearchResult GlaminFlatIndex::search(
    const std::vector<float>& queries,
    const std::uint32_t k) const {
    if (!is_open()) {
        throw std::logic_error("cannot search a closed Glamin index");
    }

    const auto query_count = dense_vector_count(queries, dimension_, "queries");
    const auto result_count = result_element_count(query_count, k);
    GlaminSearchResult result;
    result.query_count = query_count;
    result.k = k;
    result.distances.resize(result_count);
    result.labels.resize(result_count);

    const auto status = glamin_index_search_f32(
        runtime_state_->handle,
        handle_,
        queries.data(),
        query_count,
        dimension_,
        k,
        result.distances.data(),
        result.labels.data());
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime_state_->handle));
    }
    return result;
}

void GlaminFlatIndex::close() {
    if (!is_open()) {
        return;
    }

    const auto status = glamin_index_destroy(runtime_state_->handle, handle_);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime_state_->handle));
    }
    handle_ = 0;
    vector_count_ = 0;
}

void GlaminFlatIndex::close_without_throwing() noexcept {
    if (!is_open()) {
        return;
    }
    static_cast<void>(glamin_index_destroy(runtime_state_->handle, handle_));
    handle_ = 0;
    vector_count_ = 0;
}

GlaminGenerationPin::GlaminGenerationPin(
    std::shared_ptr<GlaminGenerationStoreState> store_state,
    std::shared_ptr<GlaminGenerationResource> resource,
    const std::uint64_t native_pin)
    : store_state_(std::move(store_state)),
      resource_(std::move(resource)),
      native_pin_(native_pin) {}

GlaminGenerationPin::~GlaminGenerationPin() {
    release_without_throwing();
}

GlaminGenerationPin::GlaminGenerationPin(GlaminGenerationPin&& other) noexcept
    : store_state_(std::move(other.store_state_)),
      resource_(std::move(other.resource_)),
      native_pin_(std::exchange(other.native_pin_, 0)) {}

GlaminGenerationPin& GlaminGenerationPin::operator=(
    GlaminGenerationPin&& other) noexcept {
    if (this != &other) {
        release_without_throwing();
        store_state_ = std::move(other.store_state_);
        resource_ = std::move(other.resource_);
        native_pin_ = std::exchange(other.native_pin_, 0);
    }
    return *this;
}

GlaminGenerationId GlaminGenerationPin::id() const {
    if (!is_open()) {
        throw std::logic_error("generation pin is closed");
    }
    return resource_->generation;
}

const std::string& GlaminGenerationPin::label() const {
    if (!is_open()) {
        throw std::logic_error("generation pin is closed");
    }
    return resource_->label;
}

std::uint32_t GlaminGenerationPin::dimension() const {
    if (!is_open()) {
        throw std::logic_error("generation pin is closed");
    }
    return resource_->index->dimension();
}

std::uint64_t GlaminGenerationPin::vector_count() const {
    if (!is_open()) {
        throw std::logic_error("generation pin is closed");
    }
    return resource_->index->vector_count();
}

bool GlaminGenerationPin::is_open() const noexcept {
    return native_pin_ != 0;
}

GlaminSearchResult GlaminGenerationPin::search(
    const std::vector<float>& queries,
    const std::uint32_t k) const {
    if (!is_open()) {
        throw std::logic_error("cannot search with a closed generation pin");
    }

    const auto dimension = resource_->index->dimension();
    const auto query_count = dense_vector_count(queries, dimension, "queries");
    const auto result_count = result_element_count(query_count, k);
    GlaminSearchResult result;
    result.query_count = query_count;
    result.k = k;
    result.distances.resize(result_count);
    result.labels.resize(result_count);

    const auto runtime = store_state_->runtime->handle;
    const auto status = glamin_generation_search_f32(
        runtime,
        native_pin_,
        queries.data(),
        query_count,
        dimension,
        k,
        result.distances.data(),
        result.labels.data());
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    return result;
}

void GlaminGenerationPin::release() {
    if (!is_open()) {
        return;
    }

    const auto runtime = store_state_->runtime->handle;
    const auto status = glamin_generation_unpin(runtime, native_pin_);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    native_pin_ = 0;
    resource_.reset();
}

void GlaminGenerationPin::release_without_throwing() noexcept {
    if (!is_open()) {
        return;
    }
    static_cast<void>(
        glamin_generation_unpin(store_state_->runtime->handle, native_pin_));
    native_pin_ = 0;
    resource_.reset();
}

GlaminGenerationStore::GlaminGenerationStore(GlaminRuntime& runtime) {
    if (!runtime.is_open()) {
        throw std::invalid_argument("generation store requires an open Glamin runtime");
    }
    state_ = std::make_shared<GlaminGenerationStoreState>(runtime.state_);
}

GlaminGenerationStore::~GlaminGenerationStore() = default;

GlaminGenerationStore::GlaminGenerationStore(
    GlaminGenerationStore&& other) noexcept = default;

GlaminGenerationStore& GlaminGenerationStore::operator=(
    GlaminGenerationStore&& other) noexcept = default;

GlaminGenerationId GlaminGenerationStore::mount_flat(
    std::string label,
    const std::uint32_t dimension,
    const std::vector<float>& vectors,
    const GlaminMetric metric) {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    validate_generation_label(label);

    auto index = std::unique_ptr<GlaminFlatIndex>(
        new GlaminFlatIndex(state_->runtime, dimension, metric));
    index->add(vectors);
    return mount_index(std::move(label), std::move(index));
}

GlaminGenerationId GlaminGenerationStore::mount_flat_artifact(
    std::string label,
    const std::string& artifact_directory,
    const std::string& space_id,
    const GlaminMetric metric) {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    validate_generation_label(label);
    validate_abi_text(artifact_directory, 4000, "artifact_directory");
    validate_abi_text(space_id, 128, "space_id");

    glamin_index_t native_index = 0;
    std::uint32_t dimension = 0;
    std::uint64_t vector_count = 0;
    const auto runtime = state_->runtime->handle;
    const auto status = glamin_flat_index_load_artifact(
        runtime,
        artifact_directory.data(),
        static_cast<std::uint64_t>(artifact_directory.size()),
        space_id.data(),
        static_cast<std::uint64_t>(space_id.size()),
        native_metric(metric),
        &native_index,
        &dimension,
        &vector_count);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }

    std::unique_ptr<GlaminFlatIndex> index;
    try {
        index.reset(new GlaminFlatIndex(
            state_->runtime, native_index, dimension, vector_count));
    } catch (...) {
        static_cast<void>(glamin_index_destroy(runtime, native_index));
        throw;
    }
    return mount_index(std::move(label), std::move(index));
}

GlaminGenerationId GlaminGenerationStore::mount_index(
    std::string label,
    std::unique_ptr<GlaminFlatIndex> index) {
    auto resource = std::make_shared<GlaminGenerationResource>();
    resource->index = std::move(index);
    resource->label = std::move(label);

    glamin_generation_t native_generation = 0;
    const auto runtime = state_->runtime->handle;
    const auto status = glamin_generation_create(
        runtime,
        resource->index->native_handle(),
        resource->label.data(),
        static_cast<std::uint64_t>(resource->label.size()),
        &native_generation);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    resource->generation = native_generation;

    bool inserted = false;
    try {
        const auto [unused, insertion_succeeded] =
            state_->generations.emplace(native_generation, resource);
        static_cast<void>(unused);
        inserted = insertion_succeeded;
        if (!inserted) {
            throw std::logic_error("Glamin returned a duplicate generation handle");
        }
    } catch (...) {
        if (!inserted) {
            static_cast<void>(glamin_generation_retire(runtime, native_generation));
        }
        throw;
    }
    return native_generation;
}

void GlaminGenerationStore::activate(const GlaminGenerationId generation) {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    if (state_->generations.find(generation) == state_->generations.end()) {
        throw std::invalid_argument("cannot activate an unknown generation");
    }

    const auto runtime = state_->runtime->handle;
    const auto status = glamin_generation_activate(runtime, generation);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    state_->active_generation = generation;
}

void GlaminGenerationStore::deactivate() {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }

    const auto runtime = state_->runtime->handle;
    const auto status = glamin_generation_deactivate(runtime);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    state_->active_generation = 0;
}

GlaminGenerationPin GlaminGenerationStore::pin_active() {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }

    glamin_generation_pin_t native_pin = 0;
    glamin_generation_t native_generation = 0;
    const auto runtime = state_->runtime->handle;
    const auto status =
        glamin_generation_pin_active(runtime, &native_pin, &native_generation);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }

    const auto generation = state_->generations.find(native_generation);
    if (generation == state_->generations.end()) {
        static_cast<void>(glamin_generation_unpin(runtime, native_pin));
        throw std::logic_error("active generation is not owned by this store");
    }
    return GlaminGenerationPin(state_, generation->second, native_pin);
}

void GlaminGenerationStore::retire(const GlaminGenerationId generation) {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    const auto mounted = state_->generations.find(generation);
    if (mounted == state_->generations.end()) {
        throw std::invalid_argument("cannot retire an unknown generation");
    }

    const auto runtime = state_->runtime->handle;
    const auto status = glamin_generation_retire(runtime, generation);
    if (status != GLAMIN_STATUS_OK) {
        throw GlaminError(status, read_diagnostic(runtime));
    }
    state_->generations.erase(mounted);
}

GlaminGenerationId GlaminGenerationStore::active_generation() const noexcept {
    return state_ ? state_->active_generation : 0;
}

std::size_t GlaminGenerationStore::mounted_generation_count() const noexcept {
    return state_ ? state_->generations.size() : 0;
}

std::uint32_t GlaminGenerationStore::generation_dimension(
    const GlaminGenerationId generation) const {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    const auto found = state_->generations.find(generation);
    if (found == state_->generations.end()) {
        throw std::out_of_range("generation is not mounted");
    }
    return found->second->index->dimension();
}

std::uint64_t GlaminGenerationStore::generation_vector_count(
    const GlaminGenerationId generation) const {
    if (!state_) {
        throw std::logic_error("generation store is moved from");
    }
    const auto found = state_->generations.find(generation);
    if (found == state_->generations.end()) {
        throw std::out_of_range("generation is not mounted");
    }
    return found->second->index->vector_count();
}

} // namespace gx1
