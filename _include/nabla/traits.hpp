#ifndef NABLA_TRAITS_HPP
#define NABLA_TRAITS_HPP

#include <type_traits>
#include "nabla/traits/layout.hpp"
#include "nabla/traits/tensor.hpp"
#include "nabla/types.hpp"

namespace nabla {

// Expression traits
template <typename T>
struct expr_traits_impl {
    static constexpr bool value = std::is_base_of_v<ExprTag, T>;
    static constexpr bool is_elementwise = std::is_base_of_v<ElementwiseExprTag, T>;
    using op_type = T::op_type;
    using inputs_tuple = typename T::inputs_tuple;
    using inputs_type = typename T::inputs_type;
    using rank = typename T::rank;
    using index_type = typename T::index_type;
    using subscript_type = typename T::subscript_type;
    using subscript_cref_type = typename T::subscript_cref_type;
};

template <typename T>
struct expr_traits : expr_traits_impl<std::remove_cvref_t<T>> {};

// Expression Concepts

template <typename T>
concept IsExpr = std::is_base_of_v<ExprTag, T>;

template <typename T>
concept IsElementwiseExpr = std::is_base_of_v<ElementwiseExprTag, T>;

template <typename T>
concept IsExprCompatible =
    IsTensor<T> ||
    std::is_base_of_v<ExprTag, T>;

template <typename T>
struct expr_compatible_impl {
    static constexpr bool value = IsExprCompatible<T>;
};

template <typename T>
concept IsElementwiseExprCompatible =
    IsTensor<T> ||
    std::is_base_of_v<ElementwiseExprTag, T>;

template <typename T, Rank rank>
concept IsRankN = requires(T t) {
    { T::rank } -> std::convertible_to<Rank>;
} && (T::rank == rank);

} // namespace nabla

#endif // NABLA_TRAITS_HPP
