#ifndef NABLA_LAYOUT_LEFT_STRIDE_HPP
#define NABLA_LAYOUT_LEFT_STRIDE_HPP

#include <array>
#include <sstream>
#include <stacktrace>
#include <stdexcept>
#include "nabla/types.hpp" // for is_extents_v
#include "nabla/layout/left_iterator.hpp"

namespace nabla {

struct LeftStride {
    template <typename Extents>
    class mapping : public mdspan_ns::layout_stride::mapping<Extents> {
        //
        // Member types
        //
        public:
            using base_t = mdspan_ns::layout_stride::mapping<Extents>;
            using layout_type = LeftStride;
            using extents_type = typename base_t::extents_type;
            using index_type = typename base_t::index_type;
            using rank_type = typename base_t::rank_type;
            using coord_type = std::array<index_type, extents_type::rank()>;
            using iterator_type = LeftIterator<mapping>;

        //
        // Helpers
        //
        private:
            static constexpr coord_type _default_strides(const extents_type& exts);
            constexpr void _assert_constructor() const;

            template <typename... IndexTypes>
                requires(sizeof...(IndexTypes) == Extents::rank() && (std::is_convertible_v<IndexTypes, typename LeftStride::mapping<Extents>::index_type> && ...))
            constexpr void _assert_index(IndexTypes... idxs) const;

        //
        // Constructors
        //
        public:

            // TODO: add move constructors (base_t doesn't have move extents constructor)
            // NOTE: base_t constructors are ambiguous with mapping{ {...} } brace-init constructor so we redefine them all here

            constexpr mapping() = default;
            constexpr mapping(const mapping&) = default;
            constexpr mapping(mapping&&) = default;
            constexpr mapping& operator=(const mapping&) = default;
            constexpr mapping& operator=(mapping&&) = default;

            template<typename OtherExtents>
                requires detail::is_extents_v<OtherExtents>
            constexpr mapping(const OtherExtents& exts, const coord_type& strides)
                : base_t::mapping(exts, strides) {
                    _assert_constructor();
                }

            template<typename OtherExtents>
                requires detail::is_extents_v<OtherExtents>
            constexpr mapping(const OtherExtents& exts)
                : mapping(exts, _default_strides(exts)) {}

            constexpr mapping(const coord_type& exts, const coord_type& strides)
                : mapping(extents_type(exts), strides) {}

            constexpr mapping(const coord_type& exts)
                : mapping(extents_type(exts)) {}

            constexpr mapping(const base_t& other)
                : base_t::mapping(other) {
                    _assert_constructor();
                }

        //
        // Submap
        //
        public:
            // For ADL use by submdspan
            template<class... SliceSpecifiers>
            friend constexpr auto submdspan_mapping(const mapping& src, SliceSpecifiers&&... slices) {
                auto mdspan_mapping_result = submdspan_mapping(static_cast<const base_t&>(src), std::forward<SliceSpecifiers>(slices)...);
                return mdspan_ns::submdspan_mapping_result<mapping>{mdspan_mapping_result.mapping, mdspan_mapping_result.offset};
            }

        //
        // Iterators
        //
        public:
            iterator_type begin() const {
                return iterator_type(this);
            }
            iterator_type end() const {
                return iterator_type(this, true);
            }

    }; // class mapping
}; // struct LeftStride

        //// Submap constructor
        //protected:
        //    explicit constexpr mapping(const mapping& parent, const extents_type& exts, const coord_type& offsets = {})
        //        : _extents(exts), _strides(parent._strides) {
        //            for (rank_type i = 0; i < extents_type::rank(); ++i) {
        //                if (exts.extent(i) < 0 || offsets[i] < 0 || offsets[i] + exts.extent(i) > parent._extents.extent(i)) {
        //                    std::stringstream ss;
        //                    ss << "LeftStride::mapping error: submap : index " << i << " is out of bounds\n"
        //                        << "\trequested extent: " << exts.extent(i) << " at offset " << offsets[i] << "\n"
        //                        << "\tparent upper bound: " << parent._extents.extent(i) - 1 << "\n"
        //                        << "\n\n"
        //                        << std::stacktrace::current() << std::endl;
        //                    throw std::out_of_range(ss.str());
        //                }
        //            }
        //        }
        //
        //// Submap
        //public:
        //    template<typename OtherExtents>
        //        requires detail::is_extents_v<OtherExtents>
        //    mapping submap(const OtherExtents& exts, const coord_type& offsets = {}) const {
        //        return mapping(*this, exts, offsets);
        //    }
        //
        //    mapping submap(const coord_type& exts, const coord_type& offsets = {}) const {
        //        return mapping(*this, exts, offsets);
        //    }


template <typename Extents>
typename LeftStride::mapping<Extents>::coord_type
constexpr LeftStride::mapping<Extents>::_default_strides(const extents_type& exts) {
    coord_type strides;
    index_type stride = 1;
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
        strides[i] = stride;
        stride *= exts.extent(i);
    }
    return strides;
}

template <typename Extents>
constexpr void LeftStride::mapping<Extents>::_assert_constructor() const {
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
        if (this->extents().extent(i) < 0) {
            std::stringstream ss;
            ss << "LeftStride::mapping error: extents[" << i << "] = " << this->extents().extent(i) << " must be >= 0."
                << "\n\n"
                << std::stacktrace::current() << std::endl;
            throw std::out_of_range(ss.str());
        }
    }
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
        if (this->stride(i) <= 0) {
            std::stringstream ss;
            ss << "LeftStride::mapping error: strides[" << i << "] = " << this->stride(i) << " must be > 0."
                << "\n\n"
                << std::stacktrace::current() << std::endl;
            throw std::out_of_range(ss.str());
        }
    }
    index_type min_stride = 1;
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
        if (this->stride(i) < min_stride) {
            std::stringstream ss;
            ss << "LeftStride::mapping error: strides[" << i << "] = " << this->stride(i) << " < " << min_stride
                << " at extents[" << i << "] = " << this->extents().extent(i) << ". Stride must be at least 1 for the first extents."
                << "\n\n"
                << std::stacktrace::current() << std::endl;
            throw std::out_of_range(ss.str());
        }
        min_stride *= this->extents().extent(i);
    }
}

//template <typename Extents>
//constexpr void LeftStride::mapping<Extents>::_assert_submap() const {
//    for (rank_type i = 0; i < extents_type::rank(); ++i) {
//        if (exts.extent(i) < 0 || offsets[i] < 0 || offsets[i] + exts.extent(i) > parent._extents.extent(i)) {
//            std::stringstream ss;
//            ss << "LeftStride::mapping error: submap : index " << i << " is out of bounds\n"
//                << "\trequested extent: " << exts.extent(i) << " at offset " << offsets[i] << "\n"
//                << "\tparent upper bound: " << parent._extents.extent(i) - 1 << "\n"
//                << "\n\n"
//                << std::stacktrace::current() << std::endl;
//            throw std::out_of_range(ss.str());
//        }
//    }
//}


template <typename Extents>
template <typename... IndexTypes>
    requires(sizeof...(IndexTypes) == Extents::rank() && (std::is_convertible_v<IndexTypes, typename LeftStride::mapping<Extents>::index_type> && ...))
constexpr void LeftStride::mapping<Extents>::_assert_index(IndexTypes... idxs) const {
    coord_type idx_arr{static_cast<index_type>(idxs)...};
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
        if (idx_arr[i] < 0 || idx_arr[i] >= this->extents().extent(i)) {
            std::stringstream ss;
            ss << "LeftStride::mapping error: index " << i << " is out of bounds\n"
                << "\tindices:     " << nabla::temp::to_string(idx_arr) << "\n"
                << "\tupper bound: " << nabla::temp::to_string(this->extents(), -1) << "\n"
                << "\n\n"
                << std::stacktrace::current() << std::endl;
            throw std::out_of_range(ss.str());
        }
    }
}

} // namespace nabla

#endif // NABLA_LAYOUT_LEFT_STRIDE_HPP
