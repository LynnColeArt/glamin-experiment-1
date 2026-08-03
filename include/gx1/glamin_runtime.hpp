#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "glamin_runtime.h"

namespace gx1 {

struct GlaminRuntimeState;
struct GlaminGenerationStoreState;
struct GlaminGenerationResource;

using GlaminGenerationId = std::uint64_t;

enum class GlaminMetric {
    l2,
    inner_product,
};

struct GlaminSearchResult {
    std::uint64_t query_count{0};
    std::uint32_t k{0};
    std::vector<float> distances;
    std::vector<std::uint64_t> labels;
};

class GlaminError final : public std::runtime_error {
public:
    GlaminError(glamin_status status, std::string message);

    [[nodiscard]] glamin_status status() const noexcept;

private:
    glamin_status status_;
};

class GlaminRuntime final {
public:
    explicit GlaminRuntime(std::uint32_t worker_count);
    ~GlaminRuntime();

    GlaminRuntime(const GlaminRuntime&) = delete;
    GlaminRuntime& operator=(const GlaminRuntime&) = delete;

    GlaminRuntime(GlaminRuntime&& other) noexcept;
    GlaminRuntime& operator=(GlaminRuntime&& other) noexcept;

    [[nodiscard]] static std::uint32_t abi_version() noexcept;
    [[nodiscard]] std::uint64_t native_handle() const noexcept;
    [[nodiscard]] bool is_open() const noexcept;

    void close();

private:
    friend class GlaminFlatIndex;
    friend class GlaminGenerationStore;

    std::shared_ptr<GlaminRuntimeState> state_;
};

class GlaminFlatIndex final {
public:
    GlaminFlatIndex(
        GlaminRuntime& runtime,
        std::uint32_t dimension,
        GlaminMetric metric = GlaminMetric::l2);
    ~GlaminFlatIndex();

    GlaminFlatIndex(const GlaminFlatIndex&) = delete;
    GlaminFlatIndex& operator=(const GlaminFlatIndex&) = delete;

    GlaminFlatIndex(GlaminFlatIndex&& other) noexcept;
    GlaminFlatIndex& operator=(GlaminFlatIndex&& other) noexcept;

    [[nodiscard]] std::uint64_t native_handle() const noexcept;
    [[nodiscard]] std::uint32_t dimension() const noexcept;
    [[nodiscard]] std::uint64_t vector_count() const noexcept;
    [[nodiscard]] bool is_open() const noexcept;

    void add(const std::vector<float>& vectors);
    [[nodiscard]] GlaminSearchResult search(
        const std::vector<float>& queries,
        std::uint32_t k) const;
    void close();

private:
    friend class GlaminGenerationStore;

    GlaminFlatIndex(
        std::shared_ptr<GlaminRuntimeState> runtime_state,
        std::uint32_t dimension,
        GlaminMetric metric);
    GlaminFlatIndex(
        std::shared_ptr<GlaminRuntimeState> runtime_state,
        std::uint64_t native_handle,
        std::uint32_t dimension,
        std::uint64_t vector_count);
    void close_without_throwing() noexcept;

    std::shared_ptr<GlaminRuntimeState> runtime_state_;
    std::uint64_t handle_{0};
    std::uint64_t vector_count_{0};
    std::uint32_t dimension_{0};
};

class GlaminGenerationPin final {
public:
    ~GlaminGenerationPin();

    GlaminGenerationPin(const GlaminGenerationPin&) = delete;
    GlaminGenerationPin& operator=(const GlaminGenerationPin&) = delete;

    GlaminGenerationPin(GlaminGenerationPin&& other) noexcept;
    GlaminGenerationPin& operator=(GlaminGenerationPin&& other) noexcept;

    [[nodiscard]] GlaminGenerationId id() const;
    [[nodiscard]] const std::string& label() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] std::uint64_t vector_count() const;
    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] GlaminSearchResult search(
        const std::vector<float>& queries,
        std::uint32_t k) const;
    void release();

private:
    friend class GlaminGenerationStore;

    GlaminGenerationPin(
        std::shared_ptr<GlaminGenerationStoreState> store_state,
        std::shared_ptr<GlaminGenerationResource> resource,
        std::uint64_t native_pin);
    void release_without_throwing() noexcept;

    std::shared_ptr<GlaminGenerationStoreState> store_state_;
    std::shared_ptr<GlaminGenerationResource> resource_;
    std::uint64_t native_pin_{0};
};

class GlaminGenerationStore final {
public:
    explicit GlaminGenerationStore(GlaminRuntime& runtime);
    ~GlaminGenerationStore();

    GlaminGenerationStore(const GlaminGenerationStore&) = delete;
    GlaminGenerationStore& operator=(const GlaminGenerationStore&) = delete;

    GlaminGenerationStore(GlaminGenerationStore&& other) noexcept;
    GlaminGenerationStore& operator=(GlaminGenerationStore&& other) noexcept;

    [[nodiscard]] GlaminGenerationId mount_flat(
        std::string label,
        std::uint32_t dimension,
        const std::vector<float>& vectors,
        GlaminMetric metric = GlaminMetric::l2);
    [[nodiscard]] GlaminGenerationId mount_flat_artifact(
        std::string label,
        const std::string& artifact_directory,
        const std::string& space_id,
        GlaminMetric metric = GlaminMetric::l2);
    void activate(GlaminGenerationId generation);
    void deactivate();
    [[nodiscard]] GlaminGenerationPin pin_active();
    void retire(GlaminGenerationId generation);

    [[nodiscard]] GlaminGenerationId active_generation() const noexcept;
    [[nodiscard]] std::size_t mounted_generation_count() const noexcept;
    [[nodiscard]] std::uint32_t generation_dimension(
        GlaminGenerationId generation) const;
    [[nodiscard]] std::uint64_t generation_vector_count(
        GlaminGenerationId generation) const;

private:
    [[nodiscard]] GlaminGenerationId mount_index(
        std::string label,
        std::unique_ptr<GlaminFlatIndex> index);

    std::shared_ptr<GlaminGenerationStoreState> state_;
};

} // namespace gx1
