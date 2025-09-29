#ifndef NABLA_TYPES_HPP
#define NABLA_TYPES_HPP

#include <vector>
#include "mdspan/mdspan.hpp" // for extents and MDSPAN_IMPL_STANDARD_NAMESPACE

namespace nabla{}
namespace nb = nabla;
namespace mdspan_ns = Kokkos;

namespace nabla {

    template< std::size_t Rank, typename IndexT = std::size_t >
    using dims = typename mdspan_ns::dextents<IndexT, Rank>;

    struct LeftStride;

    template <typename T>
    class default_accessor;

    template <
        typename ElementType,
        typename Extents,
        typename LayoutPolicy = nabla::LeftStride,
        typename AccessorPolicy = default_accessor<ElementType>
    > class TensorSpan;

    template <
        typename ElementType,
        typename Extents,
        typename LayoutPolicy,
        typename Container = std::vector<ElementType>
    > class TensorArray;

    struct ExprTag {};

    namespace detail {
        template <class T> struct impl_is_extents : ::std::false_type {};

        template <class IndexType, size_t... ExtentsPack>
        struct impl_is_extents<::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>> : ::std::true_type {};

        template <class T>
        inline constexpr bool is_extents_v = impl_is_extents<std::remove_cvref_t<T>>::value;
    } // namespace detail

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
