#ifndef NABLA_ELEMENTWISE_EXPR_ITERATOR_HPP
#define NABLA_ELEMENTWISE_EXPR_ITERATOR_HPP

#include <functional>
#include <utility>
#include <tuple>
#include "nabla/concepts.hpp"
#include "nabla/types.hpp"

namespace nabla {

template <typename Op, typename... Inputs>
    requires (IsExprIteratorCompatible<Inputs> && ...)
class ExprIterator : public ExprIteratorTag {
    Op _op;
    std::tuple<Inputs...> _inputs;
    using input1_t = std::tuple_element_t<0, std::tuple<Inputs...>>;

    public:

        //
        // Member types
        //
        using op_type = Op;
        using inputs_type = std::tuple<Inputs...>;
        using element_type = typename input1_t::element_type;
        using value_type = typename input1_t::value_type;
        using difference_type = typename input1_t::difference_type;
        using pointer = typename input1_t::pointer;

        //
        // Constructors
        //
        ExprIterator() = default;
        ExprIterator(const ExprIterator&) = default;
        ExprIterator(ExprIterator&&) = default;

        ExprIterator(const Op& op, const Inputs&... inputs)
            : _op(op), _inputs(inputs...) {}

        ExprIterator(Op&& op, Inputs&&... inputs)
            : _op(std::move(op)), _inputs(std::move(inputs)...) {}

        element_type operator*() const {
            return std::apply(
                [&](auto const&... ins) {
                    return _op(*ins...);
                }, _inputs);
        }

        ExprIterator& operator++() {
            std::apply([](auto&... ins) { ((++ins), ...); }, _inputs);
            return *this;
        }

        ExprIterator operator++(int) {
            ExprIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const ExprIterator& other) const {
            return std::get<0>(_inputs) == std::get<0>(other._inputs);
        }

        bool operator!=(const ExprIterator& other) const {
            return !(*this == other);
        }
};

} // namespace nabla

#endif // NABLA_ELEMENTWISE_EXPR_ITERATOR_HPP
