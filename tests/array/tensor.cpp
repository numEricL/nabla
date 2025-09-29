#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <vector>
#include <array>
#include <iostream>
#include "mdspan/mdspan.hpp"
#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"

template <typename TensorT>
    requires (TensorT::rank() == 2)
int test_assignment(TensorT& mat1, TensorT& mat2) {
    size_t counter = 0;
    for (auto iter = mat1.begin(); iter != mat1.end(); ++iter) {
        *iter = counter++;
    }
    for (auto iter = mat2.begin(); iter != mat2.end(); ++iter) {
        *iter = 0;
    }
    mat2 = mat1;
    for (size_t j = 0; j < mat1.extent(1); ++j) {
        for (size_t i = 0; i < mat1.extent(0); ++i) {
            if (mat1(i,j) != mat2(i,j)) {
                std::cerr << "Error in assignment test at (" << i << "," << j << "): expected "
                          << mat1(i,j) << ", got " << mat2(i,j) << "\n";
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    using Layout = nabla::LeftStride;
    using TensorArrayType = nabla::TensorArray<int, nabla::dims<2>, Layout>;

    Layout::mapping<nabla::dims<2>> map1({4,4}, {1,10});
    Layout::mapping<nabla::dims<2>> map2({4,4});
    TensorArrayType mat1(map1);
    TensorArrayType mat2(map2);

    int error_count = 0;
    error_count += test_assignment(mat1, mat2);

    std::cout << mat1 << std::endl;
    std::cout << mat2 << std::endl;

    if (error_count == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << error_count << " tests failed." << std::endl;
    }
    return error_count;
}
