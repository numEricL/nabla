#ifndef NABLA_TENSOR_ITERATOR_HPP
#define NABLA_TENSOR_ITERATOR_HPP

namespace nabla {

template <typename TensorType>
    requires (IsTensor<TensorType>)
class TensorIterator {
    public:
        using mapping_iterator_type = typename TensorType::mapping_type::iterator_type;

    private:
        const TensorType* _tensor;
        mapping_iterator_type _flat_iter;

    public:

        using element_type = typename TensorType::element_type;
        using value_type = typename TensorType::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = element_type*;
        using reference = element_type&;
        //using iterator_category = std::forward_iterator_tag;
        //using iterator_concept = std::forward_iterator<TensorIterator>;

        TensorIterator() = default;
        TensorIterator(const TensorIterator&) = default;
        TensorIterator(TensorIterator&&) = default;

        TensorIterator(const TensorType* tensor, mapping_iterator_type flat_iter) 
            : _tensor(tensor), _flat_iter(flat_iter) {}

        reference operator*() const {
            return _tensor->accessor().access(_tensor->data_handle(), *_flat_iter);
        }

        TensorIterator& operator++() {
            ++_flat_iter;
            return *this;
        }

        TensorIterator operator++(int) {
            TensorIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const TensorIterator& other) const {
            return _flat_iter == other._flat_iter;
        }

        bool operator!=(const TensorIterator& other) const {
            return !(*this == other);
        }

}; // class TensorIterator

} // namespace nabla

#endif // NABLA_TENSOR_ITERATOR_HPP
