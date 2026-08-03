#include "gx1/glamin_runtime.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: gx1_glamin_persistent_generation_demo "
                     "ARTIFACT_DIRECTORY [SPACE_ID]\n";
        return EXIT_FAILURE;
    }

    const std::string artifact_directory = argv[1];
    const std::string space_id = argc == 3 ? argv[2] : "geometry.auth";
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat_artifact(
        "persistent-generation", artifact_directory, space_id);
    generations.activate(generation);
    auto pin = generations.pin_active();

    std::vector<float> query(pin.dimension(), 0.0F);
    query.front() = 1.0F;
    const auto result = pin.search(query, 1);
    std::cout << "loaded generation=" << pin.id() << " label=" << pin.label()
              << " dimension=" << pin.dimension()
              << " vectors=" << pin.vector_count()
              << " neighbor=" << result.labels.at(0)
              << " distance=" << std::fixed << std::setprecision(4)
              << result.distances.at(0) << '\n';

    generations.deactivate();
    generations.retire(generation);
    pin.release();
    runtime.close();
    std::cout << "persistent Glamin generation load passed\n";
    return EXIT_SUCCESS;
}
