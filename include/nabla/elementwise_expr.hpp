#ifndef NABLA_ELEMENTWISE_EXPR_HPP
#define NABLA_ELEMENTWISE_EXPR_HPP

#include <functional> // for std::plus, etc.
#include <utility> // for std::index_sequence
#include <tuple>
#include "nabla/concepts.hpp"
#include "nabla/elementwise_expr_iterator.hpp"

// TODO: enforce invariants e.g. rank, dimensions, fp type

// optimization to consider: dimensions are a runtime invariant and duplicated
// unnecessarily in each node. we can drop them from the nodes and only
// propagate one through the expression dag.

namespace nabla {

template <typename Op, typename... Inputs>
    requires (IsSpanOrExpr<Inputs> && ...)
class ExprOp;

// base case: collect_leaf_ptrs for a leaf nodes
template <typename T>
    requires IsTensorSpan<T>
auto collect_leaf_ptrs(T& leaf) {
    return std::tuple<T*>{&leaf};
}

template <typename Op, typename... Inputs>
auto collect_leaf_ptrs(ExprOp<Op, Inputs...>& expr) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple_cat(collect_leaf_ptrs(std::get<Is>(expr._inputs))...);
    }(std::index_sequence_for<Inputs...>{});
}

// N-ary expression template
template <typename Op, typename... Inputs>
    requires (IsSpanOrExpr<Inputs> && ...)
class ExprOp : public ExprTag {
    Op _op;
    std::tuple<Inputs...> _inputs;
    using input1_t = std::tuple_element_t<0, std::tuple<Inputs...>>;

    public:

        //
        // Member types
        //
        using operation_type = Op;
        using inputs_type = std::tuple<Inputs...>;
        using extents_type = typename input1_t::extents_type;
        using index_type = typename input1_t::index_type;
        using coord_type = typename input1_t::coord_type;
        using rank_type = typename input1_t::rank_type;
        using value_type = typename input1_t::value_type;

        //
        // Member functions
        //

        //
        // Observers
        //
        static constexpr input1_t::rank_type rank() noexcept { return input1_t::rank(); }

        constexpr index_type extent(rank_type r) const noexcept { return std::get<0>(_inputs).extent(r); }
        constexpr extents_type extents() const noexcept { return std::get<0>(_inputs).extents(); }
        constexpr index_type size() const noexcept { return std::get<0>(_inputs).size(); }

        //
        // Constructors
        //

         ExprOp(const Op& op, const Inputs&... inputs)
            : _op(op), _inputs(inputs...) {}

         ExprOp(Op&& op, Inputs&&... inputs)
            : _op(std::move(op)), _inputs(std::move(inputs)...) {}

        template <typename... Args> requires (sizeof...(Args) == rank()) &&
            (std::conjunction_v<std::is_convertible<Args, index_type>...>)
        auto operator()(Args... args) const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(ins(args...)...);
                }, _inputs);
        }

        template <typename Op_, typename... Inputs_>
        friend auto collect_leaf_ptrs(ExprOp<Op_, Inputs_...>& expr);

        auto inputs() {
            return collect_leaf_ptrs(*this);
        }

        auto begin() const {
            return std::apply(
                [&](const auto&... inputs) {
                    return ExprIterator<operation_type, decltype(inputs.begin())...>{
                        _op, inputs.begin()...
                    };
                },
                _inputs
            );
        }

        auto end() const {
            // only first iterator is instantiated to end(), the rest are default-constructed
            // operator== and operator!= only compare the first iterator
            return std::apply(
                [&](const auto& first_input, const auto&... rest_inputs) {
                    return ExprIterator<operation_type, decltype(first_input.end()), decltype(rest_inputs.begin())...>{
                        _op,
                            first_input.end(),
                            decltype(rest_inputs.begin()){}... // default-constructed
                    };
                },
                _inputs
            );
        }
};

// decltype(auto) to preserve perfect forwarding
template <typename T>
requires IsTensorLike<T>
decltype(auto) span_or_forward(T&& input) {
    if constexpr (IsTensorArray<std::remove_cvref_t<T>>) {
        return input.to_span();
    } else {
        return std::forward<T>(input);
    }
}

//helper to decay input types (e.g. const T& -> T)
template <typename Op, typename... Inputs>
auto make_expr_op(Op&& op, Inputs&&... inputs) {
    return ExprOp<std::decay_t<Op>, std::decay_t<decltype(span_or_forward(std::forward<Inputs>(inputs)))>...>{
        std::forward<Op>(op), span_or_forward(std::forward<Inputs>(inputs))...
    };
}

// Operator overloads: Arithmetic operations (+ - * / % -)
template <typename In1, typename In2>
    requires (IsTensorLike<In1> && IsTensorLike<In2>)
auto operator+(In1&& input1, In2&& input2) {
    return make_expr_op( std::plus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In>
    requires IsTensorLike<In>
auto operator+(In&& input, typename std::remove_cvref_t<In>::value_type scalar) {
    return make_expr_op( [scalar](auto x) { return x + scalar; }, std::forward<In>(input));
}

template <typename In>
    requires IsTensorLike<In>
auto operator+(typename std::remove_cvref_t<In>::value_type scalar, In&& input) {
    return make_expr_op( [scalar](auto x) { return scalar + x; }, std::forward<In>(input));
}

template <typename In1, typename In2>
    requires (IsTensorLike<In1> && IsTensorLike<In2>)
auto operator-(In1&& input1, In2&& input2) {
    return make_expr_op( std::minus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In>
    requires IsTensorLike<In>
auto operator-(In&& input, typename std::remove_cvref_t<In>::value_type scalar) {
    return make_expr_op( [scalar](auto x) { return x - scalar; }, std::forward<In>(input));
}

template <typename In>
    requires IsTensorLike<In>
auto operator-(typename std::remove_cvref_t<In>::value_type scalar, In&& input) {
    return make_expr_op( [scalar](auto x) { return scalar - x; }, std::forward<In>(input));
}

template <typename In1, typename In2>
    requires (IsTensorLike<In1> && IsTensorLike<In2>)
auto operator*(In1&& input1, In2&& input2) {
    return make_expr_op( std::multiplies<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In>
    requires IsTensorLike<In>
auto operator*(In&& input, typename std::remove_cvref_t<In>::value_type scalar) {
    return make_expr_op( [scalar](auto x) { return x * scalar; }, std::forward<In>(input));
}

template <typename In>
    requires IsTensorLike<In>
auto operator*(typename std::remove_cvref_t<In>::value_type scalar, In&& input) {
    return make_expr_op( [scalar](auto x) { return scalar * x; }, std::forward<In>(input));
}

template <typename In1, typename In2>
    requires (IsTensorLike<In1> && IsTensorLike<In2>)
auto operator/(In1&& input1, In2&& input2) {
    return make_expr_op( std::divides<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In>
    requires IsTensorLike<In>
auto operator/(In&& input, typename std::remove_cvref_t<In>::value_type scalar) {
    return make_expr_op( [scalar](auto x) { return x / scalar; }, std::forward<In>(input));
}

template <typename In1, typename In2>
    requires (IsTensorLike<In1> && IsTensorLike<In2>)
auto operator%(In1&& input1, In2&& input2) {
    return make_expr_op( std::modulus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In>
    requires IsTensorLike<In>
auto operator%(In&& input, typename std::remove_cvref_t<In>::value_type scalar) {
    return make_expr_op( [scalar](auto x) { return x % scalar; }, std::forward<In>(input));
}

template <typename In>
    requires IsTensorLike<In>
auto operator-(In&& input) {
    return make_expr_op( std::negate<>(), std::forward<In>(input));
}

} // namespace nabla

#endif // NABLA_ELEMENTWISE_EXPR_HPP
