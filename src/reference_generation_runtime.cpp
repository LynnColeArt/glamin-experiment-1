#include "gx1/reference_generation_runtime.hpp"

#include <stdexcept>
#include <utility>

namespace gx1::reference {

Generation::Generation(
    const GenerationId id,
    std::string label,
    std::vector<float> scale,
    std::vector<float> bias)
    : id_(id),
      label_(std::move(label)),
      scale_(std::move(scale)),
      bias_(std::move(bias)) {
    if (id_ == 0) {
        throw std::invalid_argument("generation id zero is reserved");
    }
    if (label_.empty()) {
        throw std::invalid_argument("generation label must not be empty");
    }
    if (scale_.empty()) {
        throw std::invalid_argument("generation dimension must be positive");
    }
    if (scale_.size() != bias_.size()) {
        throw std::invalid_argument("generation scale and bias dimensions differ");
    }
}

GenerationId Generation::id() const noexcept {
    return id_;
}

const std::string& Generation::label() const noexcept {
    return label_;
}

std::size_t Generation::dimension() const noexcept {
    return scale_.size();
}

std::vector<float> Generation::traverse(const std::vector<float>& input) const {
    if (input.size() != dimension()) {
        throw std::invalid_argument("input dimension does not match generation");
    }

    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * scale_[index] + bias_[index];
    }
    return output;
}

GenerationLease::GenerationLease(std::shared_ptr<const Generation> generation)
    : generation_(std::move(generation)) {}

GenerationId GenerationLease::id() const {
    if (!generation_) {
        throw std::logic_error("generation lease is empty");
    }
    return generation_->id();
}

const std::string& GenerationLease::label() const {
    if (!generation_) {
        throw std::logic_error("generation lease is empty");
    }
    return generation_->label();
}

std::vector<float> GenerationLease::traverse(
    const std::vector<float>& input) const {
    if (!generation_) {
        throw std::logic_error("generation lease is empty");
    }
    return generation_->traverse(input);
}

GenerationLease::operator bool() const noexcept {
    return static_cast<bool>(generation_);
}

void GenerationRuntime::install(std::shared_ptr<const Generation> generation) {
    if (!generation) {
        throw std::invalid_argument("cannot install a null generation");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto [unused, inserted] = generations_.emplace(generation->id(), generation);
    static_cast<void>(unused);
    if (!inserted) {
        throw std::invalid_argument("generation id is already mounted");
    }
}

void GenerationRuntime::activate(const GenerationId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generations_.find(id) == generations_.end()) {
        throw std::invalid_argument("cannot activate an unknown generation");
    }
    active_id_ = id;
}

GenerationLease GenerationRuntime::pin_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto generation = generations_.find(active_id_);
    if (active_id_ == 0 || generation == generations_.end()) {
        throw std::logic_error("no active generation is available to pin");
    }
    return GenerationLease(generation->second);
}

void GenerationRuntime::retire(const GenerationId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id == active_id_) {
        throw std::logic_error("cannot retire the active generation");
    }
    if (generations_.erase(id) == 0) {
        throw std::invalid_argument("cannot retire an unknown generation");
    }
}

GenerationId GenerationRuntime::active_generation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_id_ == 0) {
        throw std::logic_error("no generation is active");
    }
    return active_id_;
}

std::size_t GenerationRuntime::mounted_generation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generations_.size();
}

} // namespace gx1::reference
