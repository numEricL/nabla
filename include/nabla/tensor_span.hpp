#ifndef NABLA_TENSOR_SPAN_HPP
#define NABLA_TENSOR_SPAN_HPP

#include "mdspan/mdspan.hpp"
#include "nabla/types.hpp"
#include "nabla/tensor_span_iterator.hpp"
#include "nabla/default_accessor.hpp"
#include "nabla/nested_initializer_list.hpp"

namespace nabla {

template <
    typename T,
    typename Extents,
    typename LayoutPolicy,
    typename AccessorPolicy
>
class TensorSpan<const T, Extents, LayoutPolicy, AccessorPolicy> {
    // TensorSpan class must enforce const-correctness through the inherited
    // const pattern. Non-const TensorSpan<T> inherits from TensorSpan<const T>.
    // TensorSpan<const T> owns data and provides a read-only interface, TensorSpan<T>
    // only provides a read-write interface. The "is-a" relationship allows
    // slicing off write access leaving behind a read-only interface. This
    // allows for implicit polymorphism that wouldn't ordinarily be possible,
    // such as is the case of template deduction or returning references to
    // low-const types. The (usually light-weight) copy used in conversion
    // operators is avoided as well.

    //
    // Data members
    //
    protected:
        // non-const data view
        using write_accessor_type = AccessorPolicy::write_accessor_type;
        using mdspan_type = mdspan_ns::mdspan<T, Extents, LayoutPolicy, write_accessor_type>;
        mdspan_type _mdspan;

    //
    // Member types
    //
    public:
        using element_type = const T;
        using value_type   = std::remove_cv_t<element_type>;

        using extents_type = Extents;
        using index_type   = typename extents_type::index_type;
        using size_type    = typename extents_type::size_type;
        using rank_type    = typename extents_type::rank_type;

        using layout_type  = LayoutPolicy;
        using mapping_type = typename layout_type::template mapping<extents_type>;

        using accessor_type    = typename AccessorPolicy::read_accessor_type;
        using reference        = typename accessor_type::reference;
        using data_handle_type = typename accessor_type::data_handle_type;

        using coord_type    = std::array<index_type, mdspan_type::rank()>;
        using iterator_type = TensorSpanIterator<TensorSpan>;

    //
    // Member functions
    //

    //
    // Observers
    //
    public:
        static constexpr rank_type rank() noexcept { return mdspan_type::rank(); }
        static constexpr rank_type rank_dynamic() noexcept { return mdspan_type::rank_dynamic(); }
        static constexpr std::size_t static_extent(rank_type r) noexcept { return mdspan_type::static_extent(r); }
        constexpr index_type extent(rank_type r) const noexcept { return _mdspan.extent(r); }

        // in general, data handle and accessor cannot be cast to const references
        // Q: should accessor_type::data_handle_ref_type and accessor_type::accessor_ref_type be introduced?
        constexpr data_handle_type data_handle() const noexcept { return _mdspan.data_handle(); }
        constexpr accessor_type accessor() const noexcept { return _mdspan.accessor(); }

        constexpr const extents_type& extents() const noexcept { return _mdspan.extents(); }
        constexpr const mapping_type& mapping() const noexcept { return _mdspan.mapping(); }


        constexpr index_type stride(rank_type r) const noexcept { return _mdspan.stride(r); }
        constexpr index_type size() const noexcept { return _mdspan.size(); }
        constexpr bool empty() const noexcept { return _mdspan.empty(); }

        static constexpr bool is_always_unique = mdspan_type::is_always_unique;
        static constexpr bool is_always_exhaustive = mdspan_type::is_always_exhaustive;
        static constexpr bool is_always_strided = mdspan_type::is_always_strided;

        constexpr bool is_unique() const noexcept { return _mdspan.is_unique(); }
        constexpr bool is_exhaustive() const noexcept { return _mdspan.is_exhaustive(); }
        constexpr bool is_strided() const noexcept { return _mdspan.is_strided(); }

    //
    // Constructors
    //
    public:
        constexpr TensorSpan() = default;
        constexpr TensorSpan(const TensorSpan&) = default;
        constexpr TensorSpan(TensorSpan&&) = default;

        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        explicit constexpr TensorSpan(data_handle_type p, IndexTypes... exts)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(exts...))) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorSpan(data_handle_type p, const OtherExtents& exts)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(exts))) {}

        template <typename OtherExtents>
            requires std::is_convertible_v<OtherExtents, extents_type>
        constexpr TensorSpan(data_handle_type p, const OtherExtents& exts, const coord_type& strides)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(exts), strides)) {}

        constexpr TensorSpan(data_handle_type p, const coord_type& exts)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(exts))) {}

        constexpr TensorSpan(data_handle_type p, const coord_type& exts, const coord_type& strides)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(exts), strides)) {}

        constexpr TensorSpan(data_handle_type p, const mapping_type& mapping)
            : _mdspan(accessor_type::write_cast(p), mapping) {}

        constexpr TensorSpan(data_handle_type p, const mapping_type& mapping, const accessor_type& accessor)
            : _mdspan(accessor_type::write_cast(p), mapping, accessor.to_write()) {}

    //
    // Modifiers
    // 
    public:
        void swap(TensorSpan& t) { std::swap(_mdspan, t._mdspan); }

    //
    // Element access
    //
    public:
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        constexpr reference operator()(IndexTypes... idxs) const {
#if MDSPAN_USE_BRACKET_OPERATOR
            return _mdspan[idxs...];
#else
            return _mdspan(idxs...);
#endif

        }

        reference access(size_t i) const {
            return accessor().access(data_handle(), i);
        }

    //
    // Iterators
    //
    public:
        iterator_type begin() const {
            return iterator_type(this, this->mapping().begin());
        }

        iterator_type end() const {
            return iterator_type(this, this->mapping().end());
        }
};

template <
    typename T,
    typename Extents,
    typename LayoutPolicy,
    typename AccessorPolicy
>
class TensorSpan : public TensorSpan<const T, Extents, LayoutPolicy, typename AccessorPolicy::read_accessor_type> {

    public:
        using base_type = TensorSpan<std::add_const_t<T>, Extents, LayoutPolicy, typename AccessorPolicy::read_accessor_type>;

    //
    // Data members
    //
    protected:
        using base_type::_mdspan;

    //
    // Member types
    //
    public:
        using element_type = T;
        using value_type   = std::remove_cv_t<element_type>;
        using index_type   = typename base_type::index_type;
        using accessor_type    = typename AccessorPolicy::write_accessor_type;
        using reference        = typename accessor_type::reference;
        using data_handle_type = typename accessor_type::data_handle_type;
        using iterator_type = TensorSpanIterator<TensorSpan>;

    //
    // Member functions
    //

    //
    // Observers
    //
    public:
        constexpr const accessor_type& accessor() const noexcept { return _mdspan.accessor(); }
        constexpr const data_handle_type& data_handle() const noexcept { return _mdspan.data_handle(); }

    //
    // Constructors
    //
    public:
        using base_type::base_type; // inherit constructors
        constexpr TensorSpan(const TensorSpan&) = default;
        constexpr TensorSpan(TensorSpan&&) = default;

    //
    // Modifiers
    // 
    public:
        using base_type::swap;

    //
    // Operator =
    //
    public:
        // TODO: add runtime debug assert on extents match
        template <typename U>
            requires IsTensorLike<U>
        TensorSpan& operator=(const U& other) {
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
        TensorSpan& operator=(const TensorSpan& other) {
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

        TensorSpan& operator=(TensorSpan&& other) {
            this->_mdspan = std::move(other._mdspan);
            return *this;
        }

    //
    // Element access
    //
    public:
        template <typename... IndexTypes>
            requires((std::is_convertible_v<IndexTypes, index_type> && ...))
        constexpr reference operator()(IndexTypes... idxs) const {
#if MDSPAN_USE_BRACKET_OPERATOR
            return this->_mdspan[idxs...];
#else
            return this->_mdspan(idxs...);
#endif
        }

        reference access(size_t i) const {
            return accessor().access(data_handle(), i);
        }

    //
    // Iterators
    //
    public:
        iterator_type begin() const {
            return iterator_type(this, this->mapping().begin());
        }

        iterator_type end() const {
            return iterator_type(this, this->mapping().end());
        }
};

} // namespace nabla

#endif // NABLA_TENSOR_SPAN_HPP
