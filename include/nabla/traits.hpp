#ifndef NABLA_TRAITS_HPP
#define NABLA_TRAITS_HPP

namespace nabla {

using Rank = int;
struct LayoutTag {};

template <typename T>
concept IsLayout = std::is_base_of_v<LayoutTag, T>;

template <typename T, Rank rank>
concept IsLayoutRankN = 
IsLayout<T> && requires(T t) {
    { T::rank } -> std::convertible_to<Rank>;
};

template <typename T>
struct layout_traits {
    static constexpr bool value = false;
};

template <typename T>
requires IsLayout<T>
struct layout_traits<T> {
    static constexpr bool value = true;
    static constexpr Rank rank = T::rank;
    using index_type = typename T::index_type;
    using subscript_type = typename T::subscript_type;
    using subscript_cref_type = typename T::subscript_cref_type;
};

template <typename T, Rank rank, typename LayoutT>
requires IsLayoutRankN<LayoutT, rank>
class Tensor;

template <typename T>
struct tensor_traits {
    static constexpr bool value = false;
};

template <typename T, Rank rank_, typename LayoutT>
requires IsLayoutRankN<LayoutT, rank_>
struct tensor_traits<Tensor<T, rank_, LayoutT>> {
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
concept IsTensor = tensor_traits<T>::value;

template <typename T1, typename T2>
concept IsSameRankTensors =
    IsTensor<T1> && IsTensor<T2> &&
    (tensor_traits<T1>::tensor_rank == tensor_traits<T2>::tensor_rank);

} // namespace nabla

#endif // NABLA_TRAITS_HPP
