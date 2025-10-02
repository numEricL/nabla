#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include "mdspan/mdspan.hpp"
#include <memory>
#include <iostream>
#include "nabla/nabla.hpp"
#include "accessors.hpp"

namespace nb = nabla;

template<typename T>
void print2d(const T& span) {
    for (std::size_t i = 0; i < span.extent(0); ++i) {
        for (std::size_t j = 0; j < span.extent(1); ++j) {
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

// template instantiation
template class nb::TensorSpan<int, nb::extents<std::size_t, 2, 3>, nb::LeftStride, shared_ptr_accessor<int>>;
template class nb::TensorSpan<const int, nb::extents<std::size_t, 2, 3>, nb::LeftStride, shared_ptr_accessor<const int>>;
using ctensor_t = nb::TensorSpan<const int, nb::extents<std::size_t, 2, 3>, nb::LeftStride, shared_ptr_accessor<const int>>;

template class nb::mdspan_ns::mdspan<const int, nb::extents<std::size_t, 2, 3>, nb::LeftStride, shared_ptr_accessor<const int>>;
using cmdspan_t = nb::mdspan_ns::mdspan<const int, nb::extents<std::size_t, 2, 3>, nb::LeftStride, shared_ptr_accessor<const int>>;

void shared_ptr_example() {
    // allocate with shared_ptr
    auto data = std::shared_ptr<int[]>(new int[6], std::default_delete<int[]>());
    //std::shared_ptr<const int[]> cdata = data;

    // fill it
    for (int i = 0; i < 6; ++i) data[i] = i * 10;

    using extents_t = nb::extents<std::size_t, 2, 3>;
    using layout_t = nb::LeftStride;

    using mdspan_t = Kokkos::mdspan<int, extents_t, layout_t, shared_ptr_accessor<int>>;
    using cmdspan_t = Kokkos::mdspan<const int, extents_t, layout_t, shared_ptr_accessor<const int>>;

    using Tensor_t = nb::TensorSpan<int, extents_t, layout_t, shared_ptr_accessor<int>>;
    using cTensor_t = nb::TensorSpan<const int, extents_t, layout_t, shared_ptr_accessor<const int>>;

    extents_t exts;

    mdspan_t span(data, exts);
    Tensor_t tnsr(data, exts);

    cmdspan_t cspan(data, exts);
    cTensor_t ctnsr(data, exts);

    print2d(span);
    print2d(cspan);

    print2d(tnsr);
    negate2d(tnsr);
    print2d(tnsr);

}

void example() {
    int data[6] = {0, 10, 20, 30, 40, 50};

    using extents_t = nb::extents<std::size_t, 2, 3>;
    extents_t exts;

    Kokkos::mdspan<int, extents_t>
    span(data, exts);

    Kokkos::mdspan<const int, extents_t>
    const_span(data, exts);

    print2d(span);
    print2d(const_span);
}

int main() {
    shared_ptr_example();
    example();

    auto data = std::shared_ptr<int[]>(new int[6], std::default_delete<int[]>());
    cmdspan_t span(data);
    ctensor_t tnsr(data);
    auto& d1 = span.data_handle();
    auto d2 = tnsr.data_handle();

    return 0;
}
