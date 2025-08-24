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
};

// N-ary expression template
template <typename Op, typename... Inputs>
class ExprElementWiseOp : public ElementwiseExprTag {
    Op _op;
    std::tuple<ExprStorage<Inputs>...> _inputs;

    using input1_t = std::tuple_element_t<0, std::tuple<Inputs...>>;
public:
    using op_type = Op;
    using inputs_tuple = std::tuple<ExprStorage<Inputs>...>;
    using inputs_type = std::tuple<Inputs...>;
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
};

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
