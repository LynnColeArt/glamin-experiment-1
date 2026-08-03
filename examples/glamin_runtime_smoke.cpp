#include "gx1/glamin_runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "Glamin ABI version=" << gx1::GlaminRuntime::abi_version() << '\n';

        gx1::GlaminRuntime runtime(2);
        std::cout << "Glamin runtime started handle=" << runtime.native_handle() << '\n';

        gx1::GlaminFlatIndex memory(runtime, 3);
        memory.add({
            1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 1.0F,
        });

        const auto recall = memory.search({0.05F, 0.90F, 0.05F}, 2);
        std::cout << "recall labels=[" << recall.labels[0] << ", " << recall.labels[1]
                  << "] distances=[" << recall.distances[0] << ", "
                  << recall.distances[1] << "]\n";

        memory.close();
        runtime.close();
        std::cout << "Glamin runtime stopped cleanly\n";
    } catch (const gx1::GlaminError& error) {
        std::cerr << "Glamin runtime smoke failed status="
                  << static_cast<int>(error.status())
                  << " diagnostic=" << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
