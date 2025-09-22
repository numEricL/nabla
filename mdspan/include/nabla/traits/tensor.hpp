#ifndef NABLA_TRAITS_TENSOR_HPP
#define NABLA_TRAITS_TENSOR_HPP

#include <type_traits>
#include "nabla/types.hpp"

namespace nabla {

//
// Traits
//

//template <typename T>
//struct tensor_traits_impl {
//    static constexpr bool value = false;
//};
//
//template <typename T, typename Extents, typename LayoutT, typename AccessorT>
//struct tensor_traits_impl<Tensor<T, Extents, LayoutT, AccessorT>> {
//    static constexpr bool value = true;
//    static constexpr Rank rank = rank_;
//    using layout_type = LayoutT;
//    using element_type = T;
//    using value_type = std::remove_const_t<T>;
//    using index_type = typename LayoutT::index_type;
//    using subscript_type = typename LayoutT::subscript_type;
//    using subscript_cref_type = typename LayoutT::subscript_cref_type;
//};
//
//template <typename T>
//struct tensor_traits : tensor_traits_impl<std::remove_cvref_t<T>> {};

//
// Concepts
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

} // namespace nabla

#endif // NABLA_TRAITS_TENSOR_HPP
