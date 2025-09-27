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
// TensorSpan Concepts
//
namespace detail {
    template <class T> struct impl_is_tensor_span : std::false_type {};

    template <typename T, typename Extents, typename LayoutT, typename AccessorT>
    struct impl_is_tensor_span<TensorSpan<T, Extents, LayoutT, AccessorT>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensorSpan = detail::impl_is_tensor_span<std::remove_cvref_t<T>>::value;

//
// TensorSpanIterator Concepts

template <typename TensorSpanT>
    requires (IsTensorSpan<TensorSpanT>)
class TensorSpanIterator;

namespace detail {
    template <class T> struct impl_is_tensor_span_iterator : std::false_type {};

    template <typename TensorT>
        struct impl_is_tensor_span_iterator<TensorSpanIterator<TensorT>> : std::true_type {};
    } // namespace detail

template <typename T>
concept IsTensorSpanIterator = detail::impl_is_tensor_span_iterator<std::remove_cvref_t<T>>::value;

//
// TensorArray Concepts
//
namespace detail {
    template <class T> struct impl_is_tensor_array : std::false_type {};

    template <typename T, typename Extents, typename LayoutT, typename ContainerT>
    struct impl_is_tensor_array<TensorArray<T, Extents, LayoutT, ContainerT>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensorArray = detail::impl_is_tensor_array<std::remove_cvref_t<T>>::value;

//
// Tensor Concepts
//

template <typename T>
concept IsTensor = IsTensorSpan<T> || IsTensorArray<T>;

//
// Expression Concepts
//
template <typename T>
concept IsElementwiseExpr = std::is_base_of_v<ElementwiseExprTag, T>;

template <typename T>
concept IsElementwiseExprCompatible = IsTensorSpan<T> || IsElementwiseExpr<T>;

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

} // namespace nabla

#endif // NABLA_CONCEPTS_HPP
