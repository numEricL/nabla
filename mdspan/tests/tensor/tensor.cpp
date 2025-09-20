#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <vector>
#include <array>
#include <iostream>
#include "mdspan/mdspan.hpp"
#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"

template <size_t N>
using dims = Kokkos::dextents<size_t, N>;

template <typename TensorT>
    requires (TensorT::rank() == 2)
int test_assignment(TensorT& mat) {
    using coord_type = typename TensorT::coord_type;
    coord_type dims {mat.extent(0), mat.extent(1)};
    coord_type sub_dims = {dims[0]/2, dims[1]/2};
    coord_type offsets = {1, 1};
    TensorT submat = mat.subtensor(sub_dims, offsets);

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
                              << (i + j*mat.extent(0)) << ", got " << mat(i,j) << "\n";
                    return 1;
                }
            }
        }
    }

    return 0;
}

int main() {
    using Layout = nabla::LeftStrided;
    using TensorType = nabla::Tensor<int, dims<2>, Layout>;

    Layout::mapping<dims<2>> map({4,4}, {1,10});
    std::vector<int> data(map.required_span_size());
    TensorType mat(data.data(), map);

    int error_count = 0;
    error_count += test_assignment(mat);

    if (error_count == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << error_count << " tests failed." << std::endl;
    }
    return error_count;
}
