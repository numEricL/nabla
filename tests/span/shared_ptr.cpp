#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include "mdspan/mdspan.hpp"
#include <memory>
#include <iostream>
#include "nabla/nabla.hpp"

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

template<class T>
class shared_ptr_accessor {
    public:
        using element_type = T;
        using reference = T&;
        using data_handle_type = std::shared_ptr<T[]>;
        using read_accessor_type = shared_ptr_accessor<const T>;
        using write_accessor_type = shared_ptr_accessor<std::remove_const_t<T>>;
        using write_handle_type = std::shared_ptr<std::remove_const_t<T>[]>;

        constexpr shared_ptr_accessor() noexcept = default;

        constexpr reference access(data_handle_type const& p, std::size_t i) const noexcept {
            return p.get()[i];
        }

        constexpr data_handle_type offset(data_handle_type const& p, std::size_t i) const noexcept {
            return data_handle_type(p, p.get() + i); // aliasing constructor
        }

        operator shared_ptr_accessor<const element_type>() const noexcept {
            return {};
        }

        static write_handle_type write_cast(std::shared_ptr<T[]> p) noexcept {
            return const_pointer_cast<std::remove_const_t<T>[]>(p);
        }
};

void shared_ptr_example() {
    // allocate with shared_ptr
    auto data = std::shared_ptr<int[]>(new int[6], std::default_delete<int[]>());
    //std::shared_ptr<const int[]> cdata = data;

    // fill it
    for (int i = 0; i < 6; ++i) data[i] = i * 10;

    using extents_t = Kokkos::extents<std::size_t, 2, 3>;
    using layout_t = nabla::LeftStride;

    using mdspan_t = Kokkos::mdspan<int, extents_t, layout_t, shared_ptr_accessor<int>>;
    using cmdspan_t = Kokkos::mdspan<const int, extents_t, layout_t, shared_ptr_accessor<const int>>;

    using Tensor_t = nabla::TensorSpan<int, extents_t, layout_t, shared_ptr_accessor<int>>;
    using cTensor_t = nabla::TensorSpan<const int, extents_t, layout_t, shared_ptr_accessor<const int>>;

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

    using extents_t = Kokkos::extents<std::size_t, 2, 3>;
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
    //example();
    return 0;
}
