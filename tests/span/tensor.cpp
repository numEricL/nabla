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
int test_access(TensorT& mat) {
    using coord_type = typename TensorT::coord_type;
    coord_type dims {mat.extent(0), mat.extent(1)};
    coord_type sub_dims = {dims[0]/2, dims[1]/2};
    coord_type offsets = {1, 1};
    TensorT submat = mat.subspan(sub_dims, offsets);

    for (auto iter = mat.begin(); iter != mat.end(); ++iter) {
        *iter = 0;
    }

    for (size_t j = 0; j < submat.extent(1); ++j) {
        for (size_t i = 0; i < submat.extent(0); ++i) {
            submat(i,j) = i + j*submat.extent(0);
        }
    }

    auto in_subrange = [&](size_t i, size_t j) {
        return (i >= offsets[0] && i < offsets[0] + sub_dims[0]) &&
               (j >= offsets[1] && j < offsets[1] + sub_dims[1]);
    };


    for (size_t j = 0; j < mat.extent(1); ++j) {
        for (size_t i = 0; i < mat.extent(0); ++i) {
            if (in_subrange(i,j)) {
                size_t local_i = i - offsets[0];
                size_t local_j = j - offsets[1];
                if (mat(i,j) != local_i + local_j*submat.extent(0)) {
                    std::cerr << "Error in assignment test at (" << i << "," << j << "): expected "
                              << (local_i + local_j*submat.extent(0)) << ", got " << mat(i,j) << "\n";
                    return 1;
                }
            } else {
                if (mat(i,j) != 0) {
                    std::cerr << "Error in assignment test at (" << i << "," << j << "): expected "
                              << 0 << ", got " << mat(i,j) << "\n";
                    return 1;
                }
            }
        }
    }

    return 0;
}

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
    using Layout = nabla::LeftStrided;
    using TensorType = nabla::TensorSpan<int, nabla::dims<2>, Layout>;

    Layout::mapping<nabla::dims<2>> map1({4,4}, {1,10});
    Layout::mapping<nabla::dims<2>> map2({4,4});
    std::vector<int> data1(map1.required_span_size());
    std::vector<int> data2(map2.required_span_size());
    TensorType mat1(data1.data(), map1);
    TensorType mat2(data2.data(), map2);

    int error_count = 0;
    error_count += test_access(mat1);
    error_count += test_assignment(mat1, mat2);

    if (error_count == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << error_count << " tests failed." << std::endl;
    }
    return error_count;
}
