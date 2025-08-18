#ifndef NABLA_TENSOR_ITERATOR_HPP
#define NABLA_TENSOR_ITERATOR_HPP

namespace nabla {

template <typename TensorType, bool IsConst>
class TensorIterator {
    using index_type = typename tensor_traits<TensorType>::index_type;
    using value_type = typename tensor_traits<TensorType>::value_type;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    const TensorType* _tensor;
    index_type _index;

public:
    TensorIterator(const TensorType& tensor, index_type index)
        : _tensor(&tensor), _index(index) {}

    reference operator*() const { return (*_tensor)[_index]; }
    pointer operator->() const { return &(*_tensor)[_index]; }
    reference operator[](difference_type n) const { return (*_tensor)[_index + n]; }

    TensorIterator& operator++() { ++_index; return *this; }
    TensorIterator operator++(int) { TensorIterator tmp = *this; ++(*this); return tmp; }
    TensorIterator& operator--() { --_index; return *this; }
    TensorIterator operator--(int) { TensorIterator tmp = *this; --(*this); return tmp; }

    TensorIterator operator+(difference_type n) const { return TensorIterator(*_tensor, _index + n); }
    TensorIterator operator-(difference_type n) const { return TensorIterator(*_tensor, _index - n); }
    TensorIterator& operator+=(difference_type n) { _index += n; return *this; }
    TensorIterator& operator-=(difference_type n) { _index -= n; return *this; }
    difference_type operator-(const TensorIterator& other) const { return _index - other._index; }

    auto operator<=>(const TensorIterator& other) const { return _index <=> other._index; }
    bool operator==(const TensorIterator& other) const { return _index == other._index; }
};

} // namespace nabla

#endif // NABLA_TENSOR_ITERATOR_HPP
