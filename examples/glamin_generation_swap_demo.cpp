#include "gx1/glamin_runtime.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_result(
    const std::string& stage,
    const gx1::GlaminGenerationPin& pin,
    const gx1::GlaminSearchResult& result) {
    std::cout << stage << " generation=" << pin.id() << " label=" << pin.label()
              << " neighbor=" << result.labels.at(0) << " distance=" << std::fixed
              << std::setprecision(2) << result.distances.at(0) << '\n';
}

bool same_result(
    const gx1::GlaminSearchResult& left,
    const gx1::GlaminSearchResult& right) {
    return left.labels == right.labels && left.distances == right.distances;
}

} // namespace

int main() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation_a = generations.mount_flat(
        "behavior-a", 2, {0.0F, 0.0F, 10.0F, 10.0F});
    const auto generation_b = generations.mount_flat(
        "behavior-b", 2, {100.0F, 100.0F, 2.0F, 2.0F});
    const std::vector<float> query{1.9F, 2.1F};

    generations.activate(generation_a);
    auto pinned_a = generations.pin_active();
    const auto original_a = pinned_a.search(query, 1);
    print_result("initial", pinned_a, original_a);

    generations.activate(generation_b);
    auto pinned_b = generations.pin_active();
    const auto result_b = pinned_b.search(query, 1);
    print_result("after-swap", pinned_b, result_b);

    const auto still_a = pinned_a.search(query, 1);
    print_result("old-pin", pinned_a, still_a);

    generations.activate(generation_a);
    auto rollback_pin = generations.pin_active();
    const auto rollback_result = rollback_pin.search(query, 1);
    print_result("rollback", rollback_pin, rollback_result);

    const bool swap_changed_behavior = original_a.labels != result_b.labels;
    const bool old_pin_stayed_consistent = same_result(original_a, still_a);
    const bool rollback_was_deterministic = same_result(original_a, rollback_result);
    if (!swap_changed_behavior || !old_pin_stayed_consistent ||
        !rollback_was_deterministic) {
        std::cerr << "real Glamin generation contract failed\n";
        return EXIT_FAILURE;
    }

    generations.deactivate();
    generations.retire(generation_a);
    generations.retire(generation_b);
    rollback_pin.release();
    pinned_a.release();
    pinned_b.release();
    runtime.close();

    std::cout << "real Glamin generation contract passed\n";
    return EXIT_SUCCESS;
}
