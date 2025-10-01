#ifndef NABLA_TYPES_HPP
#define NABLA_TYPES_HPP

#include <vector>
#include "mdspan/mdspan.hpp" // for extents

namespace nabla {

    namespace mdspan_ns = Kokkos;
    using mdspan_ns::extents;
    using mdspan_ns::dextents;
    using mdspan_ns::Experimental::dims;
    using mdspan_ns::strided_slice;
    using mdspan_ns::full_extent;

    struct LeftStride;

    template <typename T>
    class default_accessor;

    template <
        typename ElementType,
        typename Extents,
        typename LayoutPolicy = LeftStride,
        typename AccessorPolicy = default_accessor<ElementType>
    > class TensorSpan;

    template <
        typename ElementType,
        typename Extents,
        typename LayoutPolicy = LeftStride,
        typename Container = std::vector<ElementType>
    > class TensorArray;

    struct ExprTag {};

    // TODO: remove
    namespace temp {
        template <typename T, std::size_t N>
        std::string to_string(const std::array<T, N>& arr, T offset = 0) {
            std::stringstream ss;
            ss << "[";
            for (std::size_t i = 0; i < N; ++i) {
                ss << arr[i] + offset;
                if (i < N - 1) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }

        template <typename Extents>
        std::string to_string(const Extents& exts, typename Extents::index_type offset = 0) {
            std::stringstream ss;
            ss << "[";
            for (std::size_t i = 0; i < exts.rank(); ++i) {
                ss << exts.extent(i) + offset;
                if (i < exts.rank() - 1) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }
    } // namespace temp

} // namespace nabla

#endif // NABLA_TYPES_HPP
