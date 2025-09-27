#ifndef NABLA_TENSOR_ARRAY_ITERATOR_HPP
#define NABLA_TENSOR_ARRAY_ITERATOR_HPP

#include "nabla/concepts.hpp"

namespace nabla {

template <typename TensorArrayT>
    requires (IsTensorArray<TensorArrayT>)
class ConstTensorArrayIterator {
    public:
        using mapping_iterator_type = typename TensorArrayT::mapping_type::iterator_type;

    private:
        const TensorArrayT* _tensor;
        mapping_iterator_type _flat_iterator;

    public:

        using element_type = typename TensorArrayT::element_type;
        using value_type = typename TensorArrayT::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const element_type*;
        using reference = const element_type&;
        //using iterator_category = std::forward_iterator_tag;
        //using iterator_concept = std::forward_iterator<ConstTensorArrayIterator>;

        ConstTensorArrayIterator() = default;
        ConstTensorArrayIterator(const ConstTensorArrayIterator&) = default;
        ConstTensorArrayIterator(ConstTensorArrayIterator&&) = default;

        ConstTensorArrayIterator(const TensorArrayT* tensor, mapping_iterator_type flat_iter) 
            : _tensor(tensor), _flat_iterator(flat_iter) {}

        reference operator*() const {
            return _tensor->access(*_flat_iterator);
        }

        ConstTensorArrayIterator& operator++() {
            ++_flat_iterator;
            return *this;
        }

        ConstTensorArrayIterator operator++(int) {
            ConstTensorArrayIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const ConstTensorArrayIterator& other) const {
            return _flat_iterator == other._flat_iterator;
        }

        bool operator!=(const ConstTensorArrayIterator& other) const {
            return !(*this == other);
        }

}; // class ConstTensorArrayIterator

template <typename TensorArrayT>
    requires (IsTensorArray<TensorArrayT>)
class TensorArrayIterator {
    public:
        using mapping_iterator_type = typename TensorArrayT::mapping_type::iterator_type;

    private:
        TensorArrayT* _tensor;
        mapping_iterator_type _flat_iterator;

    public:

        using element_type = typename TensorArrayT::element_type;
        using value_type = typename TensorArrayT::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = element_type*;
        using reference = element_type&;
        //using iterator_category = std::forward_iterator_tag;
        //using iterator_concept = std::forward_iterator<TensorArrayIterator>;

        TensorArrayIterator() = default;
        TensorArrayIterator(const TensorArrayIterator&) = default;
        TensorArrayIterator(TensorArrayIterator&&) = default;

        TensorArrayIterator(TensorArrayT* tensor, mapping_iterator_type flat_iter) 
            : _tensor(tensor), _flat_iterator(flat_iter) {}

        reference operator*() const {
            return _tensor->access(*_flat_iterator);
        }

        TensorArrayIterator& operator++() {
            ++_flat_iterator;
            return *this;
        }

        TensorArrayIterator operator++(int) {
            TensorArrayIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const TensorArrayIterator& other) const {
            return _flat_iterator == other._flat_iterator;
        }

        bool operator!=(const TensorArrayIterator& other) const {
            return !(*this == other);
        }

}; // class TensorArrayIterator

} // namespace nabla

#endif // NABLA_TENSOR_ARRAY_ITERATOR_HPP
