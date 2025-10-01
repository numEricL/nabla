#ifndef NABLA_TENSOR_TENSOR_ARRAY_HPP
#define NABLA_TENSOR_TENSOR_ARRAY_HPP

#include "mdspan/mdarray.hpp"
#include "nabla/types.hpp"
#include "nabla/tensor_span.hpp"
#include "nabla/tensor_array_iterator.hpp"
#include "nabla/default_accessor.hpp"
#include "nabla/nested_initializer_list.hpp"
#include "nabla/concepts.hpp"

namespace nabla {

template <
    typename ElementType,
    typename Extents,
    typename LayoutPolicy,
    typename Container
>
class TensorArray {

    //
    // Data members
    //
    protected:
        using mdarray_type = mdspan_ns::Experimental::mdarray<ElementType, Extents, LayoutPolicy, Container>;
        mdarray_type _mdarray;

    //
    // Member types
    //
    public:
        using element_type = ElementType;
        using value_type   = std::remove_cv_t<element_type>;

        using extents_type = Extents;
        using index_type   = typename extents_type::index_type;
        using size_type    = typename extents_type::size_type;
        using rank_type    = typename extents_type::rank_type;

        using layout_type  = LayoutPolicy;
        using mapping_type = typename layout_type::template mapping<extents_type>;

        using container_type  = Container;
        using pointer         = typename container_type::pointer;
        using reference       = typename container_type::reference;
        using const_pointer   = typename container_type::const_pointer;
        using const_reference = typename container_type::const_reference;

        using coord_type    = std::array<index_type, mdarray_type::rank()>;
        using iterator = TensorArrayIterator<TensorArray>;
        using const_iterator = ConstTensorArrayIterator<TensorArray>;

    //
    // Member functions
    //

    //
    // Observers
    //
    public:
        constexpr const_pointer data() const noexcept { return _mdarray.data(); }
        constexpr pointer data() noexcept { return _mdarray.data(); }
        constexpr const container_type& container() const noexcept { return _mdarray.container(); }
        constexpr container_type& container() noexcept { return _mdarray.container(); }

        static constexpr rank_type rank() noexcept { return mdarray_type::rank(); }
        static constexpr rank_type rank_dynamic() noexcept { return mdarray_type::rank_dynamic(); }
        static constexpr std::size_t static_extent(rank_type r) noexcept { return mdarray_type::static_extent(r); }

        constexpr const extents_type& extents() const noexcept { return _mdarray.extents(); }
        constexpr index_type extent(rank_type r) const noexcept { return _mdarray.extent(r); }
        constexpr index_type size() const noexcept { return _mdarray.size(); }

        constexpr const mapping_type& mapping() const noexcept { return _mdarray.mapping(); }
        constexpr index_type stride(rank_type r) const noexcept { return _mdarray.stride(r); }

        static constexpr bool is_always_unique = mdarray_type::is_always_unique;
        static constexpr bool is_always_exhaustive = mdarray_type::is_always_exhaustive;
        static constexpr bool is_always_strided = mdarray_type::is_always_strided;

        constexpr bool is_unique() const noexcept { return _mdarray.is_unique(); }
        constexpr bool is_exhaustive() const noexcept { return _mdarray.is_exhaustive(); }
        constexpr bool is_strided() const noexcept { return _mdarray.is_strided(); }

    //
    // Constructors
    //
    public:
        constexpr TensorArray() requires(extents_type::rank_dynamic() != 0) = default;
        constexpr TensorArray(const TensorArray&) = default;
        constexpr TensorArray(TensorArray&&) = default;

        // initializer list constructors
        TensorArray(NestedInitializerList<element_type, extents_type::rank()> list)
            : _mdarray(extents_type(detail::get_extents_from_initializer_list<element_type, extents_type::rank()>(list))) {
            detail::fill_array_from_initializer_list<extents_type::rank()>(list, _mdarray);
        }

        TensorArray(NestedInitializerList<element_type, extents_type::rank()> list, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(detail::get_extents_from_initializer_list<element_type, extents_type::rank()>(list)), strides)) {
            detail::fill_array_from_initializer_list<extents_type::rank()>(list, _mdarray);
        }

        TensorArray(std::initializer_list<element_type> list, const coord_type& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides)) {
            std::copy(list.begin(), list.end(), this->begin());
        }

        // non-initializing constructors
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        explicit constexpr TensorArray(IndexTypes... exts)
            : _mdarray(extents_type(exts...)) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(const OtherExtents& exts)
            : _mdarray(extents_type(exts)) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(const OtherExtents& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides)) {}

        constexpr TensorArray(const coord_type& exts)
            : _mdarray(mapping_type(extents_type(exts))) {}

        constexpr TensorArray(const coord_type& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides)) {}

        constexpr TensorArray(const mapping_type& mapping)
            : _mdarray(mapping) {}

        // container copy constructors
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        explicit constexpr TensorArray(const container_type& ctr, IndexTypes... exts)
            : _mdarray(extents_type(exts...), ctr) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(const container_type& ctr, const OtherExtents& exts)
            : _mdarray(extents_type(exts), ctr) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(const container_type& ctr, const OtherExtents& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides), ctr) {}

        constexpr TensorArray(const container_type& ctr, const coord_type& exts)
            : _mdarray(mapping_type(extents_type(exts)), ctr) {}

        constexpr TensorArray(const container_type& ctr, const coord_type& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides), ctr) {}

        constexpr TensorArray(const container_type& ctr, const mapping_type& mapping)
            : _mdarray(mapping, ctr) {}

        // container move constructors
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        explicit constexpr TensorArray(container_type&& ctr, IndexTypes... exts)
            : _mdarray(extents_type(exts...), std::move(ctr)) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(container_type&& ctr, const OtherExtents& exts)
            : _mdarray(extents_type(exts), std::move(ctr)) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorArray(container_type&& ctr, const OtherExtents& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides), std::move(ctr)) {}

        constexpr TensorArray(container_type&& ctr, const coord_type& exts)
            : _mdarray(mapping_type(extents_type(exts)), std::move(ctr)) {}

        constexpr TensorArray(container_type&& ctr, const coord_type& exts, const coord_type& strides)
            : _mdarray(mapping_type(extents_type(exts), strides), std::move(ctr)) {}

        constexpr TensorArray(container_type&& ctr, const mapping_type& mapping)
            : _mdarray(mapping, std::move(ctr)) {}

        template <typename U>
            requires IsTensorArray<U> || IsTensorSpan<U>
        constexpr TensorArray(const U& other)
            : _mdarray(mapping_type(other.extents(), other.mapping().strides())) {
                *this = other;
            }

        template <typename U>
            requires IsTensorExpr<U>
        constexpr TensorArray(const U& other)
            : _mdarray(mapping_type(other.extents(), other.mapping().strides())) {
                *this = other;
            }

    //
    // Operator =
    //
    public:
        // TODO: add runtime debug assert on extents match
        template <typename U>
            requires IsTensorLike<U>
        TensorArray& operator=(const U& other) {
            auto it = begin();
            auto end_it = end();
            auto other_it = other.begin();
            for (; it != end_it; ++it, ++other_it) {
                *it = *other_it;
            }
            return *this;
        }

        // TODO: optimize
        // TODO: add runtime debug assert on extents match
        TensorArray& operator=(const TensorArray& other) {
            if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
                auto it = begin();
                auto end_it = end();
                auto other_it = other.begin();
                for (; it != end_it; ++it, ++other_it) {
                    *it = *other_it;
                }
            }
            return *this;
        }

        TensorArray& operator=(TensorArray&& other) = default;

    //
    // Modifiers
    // 
    public:
        void swap(TensorArray& t) { std::swap(*this, t); }
        void swap(TensorArray&& t) { std::swap(*this, t); }

    //
    // Element access
    //
    public:
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        constexpr const_reference operator()(IndexTypes... idxs) const {
#if MDSPAN_USE_BRACKET_OPERATOR
            return _mdarray[idxs...];
#else
            return _mdarray(idxs...);
#endif
        }

        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        constexpr reference operator()(IndexTypes... idxs) {
#if MDSPAN_USE_BRACKET_OPERATOR
            return _mdarray[idxs...];
#else
            return _mdarray(idxs...);
#endif
        }

        const_reference access(size_t i) const {
            return container()[i];
        }

        reference access(size_t i) {
            return container()[i];
        }

    //
    // Conversion to TensorSpan
    //
    public:

        // operator to const TensorSpan
        template <typename AccessorType = default_accessor<std::add_const_t<element_type>>>
        constexpr operator TensorSpan<std::add_const_t<element_type>, extents_type, layout_type, AccessorType>() const {
            return TensorSpan<std::add_const_t<element_type>, extents_type, layout_type, AccessorType>(data(), mapping());
        }

        // to const TensorSpan
        template <typename AccessorType = default_accessor<std::add_const_t<element_type>>>
        constexpr TensorSpan<std::add_const_t<element_type>, extents_type, layout_type, AccessorType> to_span(const AccessorType& accessor = AccessorType()) const {
            return TensorSpan<std::add_const_t<element_type>, extents_type, layout_type, AccessorType>(data(), mapping(), accessor);
        }

        // operator to TensorSpan
        template <typename AccessorType = default_accessor<element_type>>
        constexpr operator TensorSpan<element_type, extents_type, layout_type, AccessorType>() {
            return TensorSpan<element_type, extents_type, layout_type, AccessorType>(data(), mapping());
        }

        // to TensorSpan
        template <typename AccessorType = default_accessor<element_type>>
        constexpr TensorSpan<element_type, extents_type, layout_type, AccessorType> to_span(const AccessorType& accessor = AccessorType()) {
            return TensorSpan<element_type, extents_type, layout_type, AccessorType>(data(), mapping(), accessor);
        }

    //
    // Iterators
    //
    public:

        iterator begin() {
            return iterator(this, this->mapping().begin());
        }

        const_iterator begin() const {
            return const_iterator(this, this->mapping().begin());
        }

        const_iterator cbegin() const {
            return const_iterator(this, this->mapping().begin());
        }

        iterator end() {
            return iterator(this, this->mapping().end());
        }

        const_iterator end() const {
            return const_iterator(this, this->mapping().end());
        }

        const_iterator cend() const {
            return const_iterator(this, this->mapping().end());
        }
};

} // namespace nabla

#endif // NABLA_TENSOR_TENSOR_ARRAY_HPP
