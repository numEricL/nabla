#ifndef NABLA_CONCEPTS_HPP
#define NABLA_CONCEPTS_HPP

#include <type_traits>
#include "nabla/types.hpp"

namespace nabla {

//
// Tensor Concepts
//
namespace detail {
    template <class T> struct impl_is_tensor : std::false_type {};

    template <typename T, typename Extents, typename LayoutT, typename AccessorT>
    struct impl_is_tensor<Tensor<T, Extents, LayoutT, AccessorT>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensor = detail::impl_is_tensor<std::remove_cvref_t<T>>::value;

template <typename T, T::rank_type rank>
concept IsTensorRankN = IsTensor<T> && (T::rank() == rank);

template <typename T1, typename T2>
concept IsSameRankTensor =
IsTensor<T1> && IsTensor<T2> && (T1::rank() == T2::rank());

//
// Expression Concepts
//
template <typename T>
concept IsElementwiseExpr = std::is_base_of_v<ElementwiseExprTag, T>;

template <typename T>
concept IsElementwiseExprCompatible = IsTensor<T> ||
    std::is_base_of_v<ElementwiseExprTag, T>;

template <typename T, T::rank_type rank>
concept IsRankN = T::rank() == rank;

//
// Expression Iterator Concepts
//
namespace detail {
template <class T> struct impl_is_tensor_iterator : std::false_type {};

template <typename TensorType>
    struct impl_is_tensor_iterator<TensorIterator<TensorType>> : std::true_type {};
} // namespace detail

template <typename T>
concept IsTensorIterator = detail::impl_is_tensor_iterator<std::remove_cvref_t<T>>::value;

struct ExprIteratorTag {};

template <typename T>
concept IsExprIterator = std::is_base_of_v<ExprIteratorTag, T>;

template <typename T>
concept IsExprIteratorCompatible = IsTensorIterator<T> ||
std::is_base_of_v<ExprIteratorTag, T>;

} // namespace nabla

#endif // NABLA_CONCEPTS_HPP
