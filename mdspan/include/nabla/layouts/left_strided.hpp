#ifndef NABLA_LAYOUTS_LEFT_STRIDED_HPP
#define NABLA_LAYOUTS_LEFT_STRIDED_HPP

#include <array>
#include <sstream>
#include <stacktrace>
#include <stdexcept>
#include "nabla/types.hpp" // for is_extents_v
//#include "nabla/traits.hpp"


// TODO: remove _to_array and replace with fold expression in operator()

namespace nabla {

struct LeftStrided {
    template <typename Extents>
    class mapping {
        public:
            using extents_type = Extents;
            using index_type = typename extents_type::index_type;
            using rank_type = typename extents_type::rank_type;
            using coord_type = std::array<index_type, extents_type::rank()>;

        private:
            static constexpr rank_type _rank = extents_type::rank();
            static constexpr rank_type _stride_rank_dynamic = extents_type::rank();

            extents_type _extents;
            coord_type _strides;

        public:
            static constexpr rank_type stride_rank_dynamic() noexcept { return _stride_rank_dynamic; }

            static constexpr bool is_always_unique = true;
            static constexpr bool is_always_contiguous = false;
            static constexpr bool is_always_strided = true;
            constexpr bool is_unique() const noexcept { return true; }
            constexpr bool is_contiguous() const noexcept { return false; }
            constexpr bool is_strided() const noexcept { return true; }

            constexpr index_type stride(rank_type r) const { return _strides[r]; }
            constexpr const extents_type& extents() const noexcept { return _extents; }

            // not part of the standard mapping interface
            constexpr const coord_type& strides() const { return _strides; }

        private:

            static constexpr coord_type _default_strides(const extents_type& exts) {
                coord_type strides;
                index_type stride = 1;
                for (rank_type i = 0; i < _stride_rank_dynamic; ++i) {
                    strides[i] = stride;
                    stride *= exts.extent(i);
                }
                return strides;
            }

            void _assert_index(const coord_type& indices) const {
                for (rank_type i = 0; i < _rank; ++i) {
                    if (indices[i] < 0 || indices[i] >= _extents.extent(i)) {
                        std::stringstream ss;
                        ss << "LeftStrided::mapping error: index " << i << " is out of bounds\n"
                            << "\tindices:     " << temp::to_string(indices) << "\n"
                            << "\tupper bound: " << temp::to_string(_extents, -1) << "\n"
                            << "\n\n"
                            << std::stacktrace::current() << std::endl;
                        throw std::out_of_range(ss.str());
                    }
                }
            }

        public:
            mapping() = default;

            template<typename OtherExtents>
                requires detail::is_extents_v<OtherExtents>
            constexpr mapping(const OtherExtents& exts, const coord_type& strides)
                : _extents(exts), _strides(strides) {
                    for (rank_type i = 0; i < _rank; ++i) {
                        if (_extents.extent(i) < 0) {
                            std::stringstream ss;
                            ss << "LeftStrided::mapping error: extents[" << i << "] = " << _extents.extent(i) << " must be >= 0."
                                << "\n\n"
                                << std::stacktrace::current() << std::endl;
                            throw std::out_of_range(ss.str());
                        }
                    }
                    for (rank_type i = 0; i < _rank; ++i) {
                        if (stride(i) <= 0) {
                            std::stringstream ss;
                            ss << "LeftStrided::mapping error: strides[" << i << "] = " << stride(i) << " must be > 0."
                                << "\n\n"
                                << std::stacktrace::current() << std::endl;
                            throw std::out_of_range(ss.str());
                        }
                    }
                    index_type min_stride = 1;
                    for (rank_type i = 0; i < _rank; ++i) {
                        if (stride(i) < min_stride) {
                            std::stringstream ss;
                            ss << "LeftStrided::mapping error: strides[" << i << "] = " << stride(i) << " < " << min_stride
                                << " at extents[" << i << "] = " << _extents.extent(i) << ". Stride must be at least 1 for the first extents."
                                << "\n\n"
                                << std::stacktrace::current() << std::endl;
                            throw std::out_of_range(ss.str());
                        }
                        min_stride *= _extents.extent(i);
                    }
                }

            template<typename OtherExtents>
                requires detail::is_extents_v<OtherExtents>
            constexpr mapping(const OtherExtents& exts)
                : mapping(exts, _default_strides(exts)) {}

            constexpr mapping(const coord_type& exts, const coord_type& strides)
                : mapping(extents_type(exts), strides) {}

            constexpr mapping(const coord_type& exts)
                : mapping(extents_type(exts)) {}

            constexpr mapping(const mapping&) = default;
            constexpr mapping(mapping&&) = default;

        protected:
            explicit constexpr mapping(const mapping& parent, const extents_type& exts, const coord_type& offsets = {})
                : _extents(exts), _strides(parent._strides) {
                    for (rank_type i = 0; i < _rank; ++i) {
                        if (exts.extent(i) < 0 || offsets[i] < 0 || offsets[i] + exts.extent(i) > parent._extents.extent(i)) {
                            std::stringstream ss;
                            ss << "LeftStrided::mapping error: submap : index " << i << " is out of bounds\n"
                                << "\trequested extent: " << exts.extent(i) << " at offset " << offsets[i] << "\n"
                                << "\tparent upper bound: " << parent._extents.extent(i) - 1 << "\n"
                                << "\n\n"
                                << std::stacktrace::current() << std::endl;
                            throw std::out_of_range(ss.str());
                        }
                    }
                }

        public:
            template<typename OtherExtents>
                requires detail::is_extents_v<OtherExtents>
            mapping submap(const OtherExtents& exts, const coord_type& offsets = {}) const {
                return mapping(*this, exts, offsets);
            }

            mapping submap(const coord_type& exts, const coord_type& offsets = {}) const {
                return mapping(*this, exts, offsets);
            }

            mapping& operator=(const mapping&) = default;
            mapping& operator=(mapping&&) = default;

        private:
            template <class... Indices>
                requires(sizeof...(Indices) == _rank)
            static constexpr coord_type _to_array(Indices&&... indices) {
                return coord_type{static_cast<index_type>(std::forward<Indices>(indices))...};
            }

        public:
            template <class... Indices>
                requires(sizeof...(Indices) == _rank)
            constexpr index_type operator()(Indices&&... indices) const {
                auto idx_arr = _to_array(std::forward<Indices>(indices)...);
                _assert_index(idx_arr);
                index_type idx = 0;
                for (rank_type i = 0; i < _rank; ++i) {
                    idx += idx_arr[i] * stride(i);
                }
                return idx;
            }

            //index_type flat_index(index_type idx) const {
            //    std::array<index_type, rank> indices;
            //    for (index_type i = rank - 1; i >= 0; --i) {
            //        indices[i] = idx / stride(i);
            //        idx -= indices[i] * stride(i);
            //    }
            //    return this->operator()(indices);
            //}

            constexpr std::size_t required_span_size() const {
                std::size_t size = 0;
                for (rank_type i = 0; i < _rank; ++i) {
                    size += (_extents.extent(i) - 1) * stride(i);
                }
                return size + 1;
            }

            // forward iterator that avoids integer multiplication in
            // incrementer. Constructors may still use multiplication.
            class FlatIndexIterator {
                private:
                    const mapping* _mapping = nullptr;
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
                    FlatIndexIterator(const mapping* mapping)
                        : _mapping(mapping) {}

                    // end iterator constructor
                    FlatIndexIterator(const mapping* mapping, bool)
                        : _mapping(mapping), _flat_index(mapping->required_span_size()) {}

                    // middle iterator constructor
                    FlatIndexIterator(const mapping* mapping, coord_type indices)
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

                    // no integer multiplication
                    void increment() {
                        index_type prev_index = _flat_index;
                        for (rank_type r = 0; r < _rank; ++r) {
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

            }; // class FlatIndexIterator

            FlatIndexIterator begin() const {
                return FlatIndexIterator(this);
            }

            FlatIndexIterator end() const {
                return FlatIndexIterator(this, true);
            }

    }; // class mapping
}; // struct LeftStrided

} // namespace nabla

#endif // NABLA_LAYOUTS_LEFT_STRIDED_HPP
