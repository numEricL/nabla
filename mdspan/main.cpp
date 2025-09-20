#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <iostream>
#include <vector>
#include <array>
#include <span>
#include "mdspan/mdspan.hpp"
#include "nabla/tensor.hpp"
#include "nabla/ostream.hpp"

using nabla::dims;
using nabla::Tensor;

template<typename T>
void print2d(const T& span) {
    std::cout << "Printing array:\n";
    for (std::size_t i = 0; i < span.extents().extent(0); ++i) {
        for (std::size_t j = 0; j < span.extents().extent(1); ++j) {
            std::cout << span(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T>
void negate2d(T& span) {
    for (std::size_t i = 0; i < span.extent(0); ++i) {
        for (std::size_t j = 0; j < span.extent(1); ++j) {
            span(i, j) = -span(i, j);
        }
    }
}

template<typename T>
void print_flat(const T& span) {
    std::cout << "Printing flat array: " << nabla::temp::to_string(span.extents()) << "\n";
    for (auto it = span.begin(); it != span.end(); ++it) {
        std::cout << *it << "\t";
    }
    std::cout << "\n\n";
}

int main() {
    // using layout_t = mdspan_ns::layout_stride;
    using layout_t = nabla::LeftStrided;
    using mapping_t = layout_t::mapping<dims<2>>;
    constexpr dims<2> extents{3, 3};
    constexpr mapping_t::coord_type strides{10, 100};

    constexpr auto rows = extents.extent(0);
    constexpr auto cols = extents.extent(1);

    std::vector<int> data(100); // plenty of room
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i);
    }

    mapping_t layout(extents, strides);

    using mdspan_t = mdspan_ns::mdspan<int, dims<2>, layout_t>;
    using tensor_t = Tensor<int, dims<2>, layout_t>;
    using const_tensor_t = Tensor<const int, dims<2>, layout_t>;

    dims<2> dext{rows, cols};

    //mdspan_t arr(data.data(), layout);
    //tensor_t tns(data.data(), layout);

    mdspan_t arr(data.data(), layout);
    tensor_t tns(data.data(), {rows, cols});

    auto sub = tns.subtensor({2, 2}, {1, 1});

    const_tensor_t const_sub = sub;

    //mdspan_t arr(data.data(), {rows, cols});
    //tensor_t tns(data.data(), {rows, cols});
    //
    //auto sub = tns.subtensor({2, 2}, {1, 1});

    std::cout << "sizeof(mdspan): " << sizeof(arr) << std::endl;
    std::cout << "sizeof(size_t): " << sizeof(std::size_t) << std::endl;

    int val = 0;
    for (std::size_t i = 0; i < arr.extent(0); ++i) {
        for (std::size_t j = 0; j < arr.extent(1); ++j) {
            tns(i, j) = val++;
        }
        std::cout << std::endl;
    }

    const_sub(0, 0);

    print2d(arr);
    print2d(tns);
    print2d(sub);
    print2d(const_sub);
    negate2d(sub);
    print2d(tns);

    print_flat(tns);
    print_flat(sub);

    return 0;
}
