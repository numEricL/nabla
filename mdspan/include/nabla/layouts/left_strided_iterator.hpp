#ifndef NABLA_LAYOUTS_LEFT_STRIDED_ITERATOR_HPP
#define NABLA_LAYOUTS_LEFT_STRIDED_ITERATOR_HPP

namespace nabla {

// forward iterator that avoids integer multiplication in
// incrementer. Constructors still use multiplication.
template <typename Extents>
class LeftStrided::FlatIndexIterator {
    using mapping_type = LeftStrided::mapping<Extents>;
    using extents_type = typename mapping_type::extents_type;
    using index_type = typename mapping_type::index_type;
    using rank_type = typename mapping_type::rank_type;
    using coord_type = std::array<index_type, mapping_type::extents_type::rank()>;

private:
    const mapping_type* _mapping = nullptr;
    coord_type _indices{};
    coord_type _deltas{};
    index_type _flat_index = 0;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = index_type;
    using reference = value_type;
    using pointer = void;
    using iterator_category = std::forward_iterator_tag;

    FlatIndexIterator() = default;

    // begin iterator constructor
    FlatIndexIterator(const mapping_type* mapping)
        : _mapping(mapping) {}

    // end iterator constructor
    FlatIndexIterator(const mapping_type* mapping, bool)
        : _mapping(mapping), _flat_index(mapping->required_span_size()) {}

    // middle iterator constructor
    FlatIndexIterator(const mapping_type* mapping, coord_type indices)
        : _mapping(mapping), _indices(indices), _flat_index(mapping->operator()(indices)) {}

    reference operator*() const {
        return _flat_index;
    }

    FlatIndexIterator& operator++() {
        increment();
        return *this;
    }

    FlatIndexIterator operator++(int) {
        FlatIndexIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const FlatIndexIterator& other) const {
        return _flat_index == other._flat_index;
    }

    bool operator!=(const FlatIndexIterator& other) const {
        return !(*this == other);
    }

private:
    void increment() {
        index_type prev_index = _flat_index;
        for (rank_type r = 0; r < mapping_type::extents_type::rank(); ++r) {
            ++_indices[r];
            _flat_index += _mapping->stride(r);
            _deltas[r] += _mapping->stride(r);
            if (_indices[r] < _mapping->extents().extent(r)) {
                return;
            }
            // Wrap around this dimension
            _flat_index -= _deltas[r];
            _deltas[r] = 0;
            _indices[r] = 0;
        }
        _flat_index = prev_index + 1; // one past the end
    }
};

} // namespace nabla

#endif // NABLA_LAYOUTS_LEFT_STRIDED_ITERATOR_HPP
