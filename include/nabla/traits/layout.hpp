#ifndef NABLA_TRAITS_LAYOUT_HPP
#define NABLA_TRAITS_LAYOUT_HPP

#include <type_traits>
#include "nabla/types.hpp"

namespace nabla {

//
// Traits
//

template <typename T>
struct layout_traits_impl {
    static constexpr bool value = std::is_base_of_v<LayoutTag, T>;
    static constexpr Rank rank = T::rank;
    using index_type = typename T::index_type;
    using subscript_type = typename T::subscript_type;
    using subscript_cref_type = typename T::subscript_cref_type;
};

template <typename T>
struct layout_traits : layout_traits_impl<std::remove_cvref_t<T>> {};

//
// Concepts
//

template <typename T>
concept IsLayout = std::is_base_of_v<LayoutTag, T>;

template <typename T, Rank rank>
concept IsLayoutRankN = 
IsLayout<T> && requires(T t) {
    { T::rank } -> std::convertible_to<Rank>;
};

} // namespace nabla

#endif // NABLA_TRAITS_LAYOUT_HPP
