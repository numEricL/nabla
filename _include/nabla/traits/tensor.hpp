#ifndef NABLA_TRAITS_TENSOR_HPP
#define NABLA_TRAITS_TENSOR_HPP

#include <type_traits>
#include "nabla/traits/layout.hpp"

namespace nabla {

//
// Traits
//

template <typename T, Rank rank, typename LayoutT>
requires IsLayoutRankN<LayoutT, rank>
class Tensor;

template <typename T>
struct tensor_traits_impl {
    static constexpr bool value = false;
};

template <typename T, Rank rank_, typename LayoutT>
struct tensor_traits_impl<Tensor<T, rank_, LayoutT>> {
    static constexpr bool value = true;
    static constexpr Rank rank = rank_;
    using layout_type = LayoutT;
    using element_type = T;
    using value_type = std::remove_const_t<T>;
    using index_type = typename LayoutT::index_type;
    using subscript_type = typename LayoutT::subscript_type;
    using subscript_cref_type = typename LayoutT::subscript_cref_type;
};

template <typename T>
struct tensor_traits : tensor_traits_impl<std::remove_cvref_t<T>> {};

//
// Concepts
//

template <typename T>
concept IsTensor = tensor_traits<T>::value;

template <typename T, Rank rank>
concept IsTensorRankN = IsTensor<T> && (tensor_traits<T>::rank == rank);

template <typename T1, typename T2>
concept IsSameRankTensors =
IsTensor<T1> && IsTensor<T2> &&
(tensor_traits<T1>::tensor_rank == tensor_traits<T2>::tensor_rank);

} // namespace nabla

#endif // NABLA_TRAITS_TENSOR_HPP
