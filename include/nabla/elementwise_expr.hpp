#include <functional>
#include <utility>
#include <tuple>
#include "nabla/traits.hpp"

// TODO: enforce invariants e.g. rank, dimensions, fp type

// optimization to consider: dimensions are a runtime invariant and duplicated
// unnecessarily in each node. we can drop them from the nodes and only
// propagate one through the expression dag.

namespace nabla {

template <typename T>
    requires IsExprCompatible<T>
class ExprStorage {
    T _expr;
    public:
        ExprStorage(const T& expr) : _expr(expr) {}
        ExprStorage(T&& expr) : _expr(std::move(expr)) {}
        T::index_type size() const { return _expr.size(); }
        const T& get() const { return _expr; }
        T& get() { return _expr; }
};

template <typename Op, typename... Inputs>
class ExprElementWiseOp;

// base case: collect_leaf_ptrs for a leaf nodes
template <typename T>
    requires IsTensor<T>
auto collect_leaf_ptrs(T& leaf) {
    return std::tuple<T*>{&leaf};
}

// returns inputs of an expression in left-to-right depth-first order
//template <typename Op, typename... Inputs>
//auto collect_leaf_ptrs(ExprElementWiseOp<Op, Inputs...>& expr) {
//    return std::tuple_cat(collect_leaf_ptrs(std::get<ExprStorage<Inputs>>(expr._inputs).get())...);
//}

template <typename Op, typename... Inputs, std::size_t... Is>
auto collect_leaf_ptrs_impl(ExprElementWiseOp<Op, Inputs...>& expr, std::index_sequence<Is...>) {
    return std::tuple_cat(collect_leaf_ptrs(std::get<Is>(expr._inputs).get())...);
}

template <typename Op, typename... Inputs>
auto collect_leaf_ptrs(ExprElementWiseOp<Op, Inputs...>& expr) {
    return collect_leaf_ptrs_impl(expr, std::index_sequence_for<Inputs...>{});
}

// N-ary expression template
template <typename Op, typename... Inputs>
class ExprElementWiseOp : public ElementwiseExprTag {
    Op _op;
    std::tuple<ExprStorage<Inputs>...> _inputs;
    using input1_t = std::tuple_element_t<0, std::tuple<Inputs...>>;

    public:
        using op_type = Op;
        using inputs_type = std::tuple<ExprStorage<Inputs>...>;
        static constexpr Rank rank = input1_t::rank;
        using index_type = typename input1_t::index_type;
        using subscript_type = typename input1_t::subscript_type;
        using subscript_cref_type = typename input1_t::subscript_cref_type;

        ExprElementWiseOp(Op op, const Inputs&... inputs)
            : _op(op), _inputs(inputs...) {}

        ExprElementWiseOp(Op op, Inputs&&... inputs)
            : _op(std::move(op)), _inputs(std::move(inputs)...) {}

        index_type size() const {
            return std::get<0>(_inputs).get().size();
        }

        subscript_type dimensions() const {
            return std::get<0>(_inputs).get().dimensions();
        }

        auto operator[](size_t i) const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(ins.get()[i]...);
                }, _inputs);
        }

        template <typename... Args> requires (sizeof...(Args) == rank) &&
            (std::conjunction_v<std::is_convertible<Args, index_type>...>)
        auto operator()(Args... args) const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(ins.get()(args...)...);
                }, _inputs);
        }

        template <typename Op_, typename... Inputs_, std::size_t... Is>
        friend auto collect_leaf_ptrs_impl(ExprElementWiseOp<Op_, Inputs_...>& expr, std::index_sequence<Is...>);

        auto inputs() {
            return collect_leaf_ptrs(*this);
        }
};

//helper to decay input types (e.g. const T& -> T)
template <typename Op, typename... Inputs>
auto make_elementwise_op(Op&& op, Inputs&&... inputs) {
    return ExprElementWiseOp< std::decay_t<Op>, std::decay_t<Inputs>... >(
        std::forward<Op>(op), std::forward<Inputs>(inputs)...
        );
}

// Operator overloads: Arithmetic operations (+ - * / % -)
template <typename In1, typename In2>
auto operator+(In1&& input1, In2&& input2) {
    return make_elementwise_op( std::plus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator-(In1&& input1, In2&& input2) {
    return make_elementwise_op( std::minus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator*(In1&& input1, In2&& input2) {
    return make_elementwise_op( std::multiplies<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator/(In1&& input1, In2&& input2) {
    return make_elementwise_op( std::divides<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1, typename In2>
auto operator%(In1&& input1, In2&& input2) {
    return make_elementwise_op( std::modulus<>(), std::forward<In1>(input1), std::forward<In2>(input2));
}

template <typename In1>
auto operator-(In1&& input1) {
    return make_elementwise_op( std::negate<>(), std::forward<In1>(input1));
}

} // namespace nabla
