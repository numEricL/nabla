#include <functional>
#include <utility>
#include <tuple>
#include "nabla/concepts.hpp"

// TODO: enforce invariants e.g. rank, dimensions, fp type

// optimization to consider: dimensions are a runtime invariant and duplicated
// unnecessarily in each node. we can drop them from the nodes and only
// propagate one through the expression dag.

namespace nabla {

template <typename Op, typename... Inputs>
    requires (IsElementwiseExprCompatible<Inputs> && ...)
class ExprElementWiseOp;

// base case: collect_leaf_ptrs for a leaf nodes
template <typename T>
    requires IsTensor<T>
auto collect_leaf_ptrs(T& leaf) {
    return std::tuple<T*>{&leaf};
}

template <typename Op, typename... Inputs>
auto collect_leaf_ptrs(ExprElementWiseOp<Op, Inputs...>& expr) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple_cat(collect_leaf_ptrs(std::get<Is>(expr._inputs))...);
    }(std::index_sequence_for<Inputs...>{});
}

// N-ary expression template
template <typename Op, typename... Inputs>
    requires (IsElementwiseExprCompatible<Inputs> && ...)
class ExprElementWiseOp : public ElementwiseExprTag {
    Op _op;
    std::tuple<Inputs...> _inputs;
    using input1_t = std::tuple_element_t<0, std::tuple<Inputs...>>;

    public:

        //
        // Member types
        //
        using op_type = Op;
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
        ExprElementWiseOp(Op op, const Inputs&... inputs)
            : _op(op), _inputs(inputs...) {}

        ExprElementWiseOp(Op op, Inputs&&... inputs)
            : _op(std::move(op)), _inputs(std::move(inputs)...) {}

        // TODO: remove in favor of iterator-based access
        auto operator[](size_t i) const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(ins[i]...);
                }, _inputs);
        }

        template <typename... Args> requires (sizeof...(Args) == rank()) &&
            (std::conjunction_v<std::is_convertible<Args, index_type>...>)
        auto operator()(Args... args) const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(ins(args...)...);
                }, _inputs);
        }

        template <typename Op_, typename... Inputs_>
        friend auto collect_leaf_ptrs(ExprElementWiseOp<Op_, Inputs_...>& expr);

        auto inputs() {
            return collect_leaf_ptrs(*this);
        }
};

//helper to decay input types (e.g. const T& -> T)
template <typename Op, typename... Inputs>
auto make_expr_element_wise_op(Op&& op, Inputs&&... inputs) {
    return ExprElementWiseOp< std::decay_t<Op>, std::decay_t<Inputs>... >{
        std::forward<Op>(op), std::forward<Inputs>(inputs)...
    };
}

// Operator overloads: Arithmetic operations (+ - * / % -)
template <typename In1, typename In2>
auto operator+(In1&& input1, In2&& input2) {
    return make_expr_element_wise_op( std::plus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator-(In1&& input1, In2&& input2) {
    return make_expr_element_wise_op( std::minus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator*(In1&& input1, In2&& input2) {
    return make_expr_element_wise_op( std::multiplies<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator/(In1&& input1, In2&& input2) {
    return make_expr_element_wise_op( std::divides<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator%(In1&& input1, In2&& input2) {
    return make_expr_element_wise_op( std::modulus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1>
auto operator-(In1&& input1) {
    return make_expr_element_wise_op( std::negate<>(), std::forward<In1>(input1));
}

} // namespace nabla
