#ifndef NABLA_TENSOR_SPAN_ITERATOR_HPP
#define NABLA_TENSOR_SPAN_ITERATOR_HPP

#include "nabla/concepts.hpp"

namespace nabla {

template <typename TensorSpanT>
    requires (IsTensorSpan<TensorSpanT>)
class TensorSpanIterator {
    public:
        using mapping_iterator_type = typename TensorSpanT::mapping_type::iterator_type;

    private:
        const TensorSpanT* _tensor;
        mapping_iterator_type _flat_iterator;

    public:

        using element_type = typename TensorSpanT::element_type;
        using value_type = typename TensorSpanT::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = element_type*;
        using reference = element_type&;
        //using iterator_category = std::forward_iterator_tag;
        //using iterator_concept = std::forward_iterator<TensorSpanIterator>;

        TensorSpanIterator() = default;
        TensorSpanIterator(const TensorSpanIterator&) = default;
        TensorSpanIterator(TensorSpanIterator&&) = default;

        TensorSpanIterator(const TensorSpanT* tensor, mapping_iterator_type flat_iter) 
            : _tensor(tensor), _flat_iterator(flat_iter) {}

        reference operator*() const {
            return _tensor->access(*_flat_iterator);
        }

        TensorSpanIterator& operator++() {
            ++_flat_iterator;
            return *this;
        }

        TensorSpanIterator operator++(int) {
            TensorSpanIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const TensorSpanIterator& other) const {
            return _flat_iterator == other._flat_iterator;
        }

        bool operator!=(const TensorSpanIterator& other) const {
            return !(*this == other);
        }

}; // class TensorSpanIterator

} // namespace nabla

#endif // NABLA_TENSOR_SPAN_ITERATOR_HPP
