#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <vector>
#include <array>
#include <iostream>
#include "mdspan/mdspan.hpp"
#include "nabla/tensor.hpp"
#include "nabla/ostream.hpp"

template <size_t N>
using dims = Kokkos::dextents<size_t, N>;

int main() {
    using Layout = nabla::LeftStrided;
    using TensorType = nabla::Tensor<int, dims<2>, Layout>;

    Layout::mapping<dims<2>> map({3,4}, {1,10});
    std::vector<int> data(map.required_span_size());

    TensorType t1(data.data(), map);
    TensorType t2 = t1.subtensor({2,2}, {1,1});

    for (size_t j = 0; j < t2.extent(1); ++j) {
        for (size_t i = 0; i < t2.extent(0); ++i) {
            t2(i,j) = i + j*t2.extent(0);
        }
    }

    std::cout << "t1:\n" << t1 << "\n";

}
