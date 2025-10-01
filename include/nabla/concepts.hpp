#ifndef NABLA_CONCEPTS_HPP
#define NABLA_CONCEPTS_HPP

#include <type_traits>
#include "nabla/types.hpp"

namespace nabla {

//template <typename T, T::rank_type N>
//concept IsRankN = (T::rank() == N);
//
//template <typename T1, typename T2>
//concept IsSameRankTensor =
//IsTensorSpan<T1> && IsTensorSpan<T2> && (T1::rank() == T2::rank());

//
// Extents Concepts
//
namespace detail {
    template <typename T> struct impl_is_extents : std::false_type {};

    template <typename IndexType, size_t... ExtentsPack>
    struct impl_is_extents<extents<IndexType, ExtentsPack...>> : std::true_type {};

} // namespace detail

template <typename T>
concept IsExtents = detail::impl_is_extents<std::remove_cvref_t<T>>::value;

//
// Mapping Concepts
//
template <typename T>
concept IsMapping = requires { typename T::mapping_tag; };

//
// TensorSpan Concepts
//
namespace detail {
    template <typename T> struct impl_is_tensor_span : std::false_type {};

    template <typename T, typename Extents, typename LayoutT, typename AccessorT>
    struct impl_is_tensor_span<TensorSpan<T, Extents, LayoutT, AccessorT>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensorSpan = detail::impl_is_tensor_span<std::remove_cvref_t<T>>::value;

//
// TensorSpanIterator Concepts
//
template <typename TensorSpanT>
    requires (IsTensorSpan<TensorSpanT>)
class TensorSpanIterator;

namespace detail {
    template <typename T> struct impl_is_tensor_span_iterator : std::false_type {};

    template <typename TensorT>
        struct impl_is_tensor_span_iterator<TensorSpanIterator<TensorT>> : std::true_type {};
    } // namespace detail

template <typename T>
concept IsTensorSpanIterator = detail::impl_is_tensor_span_iterator<std::remove_cvref_t<T>>::value;

//
// TensorArray Concepts
//
namespace detail {
    template <typename T> struct impl_is_tensor_array : std::false_type {};

    template <typename T, typename Extents, typename LayoutT, typename ContainerT>
    struct impl_is_tensor_array<TensorArray<T, Extents, LayoutT, ContainerT>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensorArray = detail::impl_is_tensor_array<std::remove_cvref_t<T>>::value;

//
// Expression Concepts
//
template <typename T>
concept IsTensorExpr = std::is_base_of_v<ExprTag, std::remove_cvref_t<T>>;

template <typename T>
concept IsSpanOrExpr = IsTensorSpan<T> || IsTensorExpr<T>;

template <typename T, T::rank_type rank>
concept IsRankN = T::rank() == rank;

//
// Expression Iterator Concepts
//
struct ExprIteratorTag {};

template <typename T>
concept IsExprIterator = std::is_base_of_v<ExprIteratorTag, T>;

template <typename T>
concept IsExprIteratorCompatible = IsTensorSpanIterator<T> || IsExprIterator<T>;

//
// Tensor Concepts
//
template <typename T>
concept IsTensorLike = IsTensorSpan<T> || IsTensorArray<T> || IsTensorExpr<T>;


} // namespace nabla

#endif // NABLA_CONCEPTS_HPP
