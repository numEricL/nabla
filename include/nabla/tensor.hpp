#ifndef NABLA_TENSOR_TENSOR_HPP
#define NABLA_TENSOR_TENSOR_HPP

#include "mdspan/mdspan.hpp"
#include "nabla/types.hpp"
#include "nabla/tensor_iterator.hpp"

namespace nabla {

template <typename T>
class default_accessor<const T> {
    public:
        //using offset_policy = default_accessor;
        using element_type = const T;
        using reference = const T&;
        using data_handle_type = const T*;
        using read_accessor_type = default_accessor<const T>;
        using write_accessor_type = default_accessor<T>;

        constexpr default_accessor() noexcept = default;

        template <typename OtherElementType>
            requires std::is_convertible_v<OtherElementType(*)[], element_type(*)[]>
        constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p[i];
        }

        constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
            return p + i;
        }

    private:
        template <typename U, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
        friend class Tensor;

        using write_handle_type = T*;
        static write_handle_type write_cast(data_handle_type p) noexcept {
            return const_cast<write_handle_type>(p);
        }
};

template <typename T>
class default_accessor : default_accessor<const T> {
    public:
        //using offset_policy = default_accessor;
        using element_type = T;
        using reference = T&;
        using data_handle_type = T*;
        using read_accessor_type = default_accessor<const T>;
        using write_accessor_type = default_accessor<T>;

        constexpr default_accessor() noexcept = default;

        template <typename OtherElementType>
            requires std::is_convertible_v<OtherElementType(*)[], element_type(*)[]>
            constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p[i];
        }

        constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
            return p + i;
        }
};

template <
    typename T,
    typename Extents,
    typename LayoutPolicy,
    typename AccessorPolicy
>
class Tensor<const T, Extents, LayoutPolicy, AccessorPolicy> {
    // Tensor class must enforce const-correctness through the inherited
    // const pattern. Non-const Tensor<T> inherits from Tensor<const T>.
    // Tensor<const T> owns data and provides a read-only interface, Tensor<T>
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
        using index_type   = extents_type::index_type;
        using size_type    = extents_type::size_type;
        using rank_type    = extents_type::rank_type;

        using layout_type  = LayoutPolicy;
        using mapping_type = layout_type::template mapping<extents_type>;

        using accessor_type    = AccessorPolicy::read_accessor_type;
        using reference        = typename accessor_type::reference;
        using data_handle_type = typename accessor_type::data_handle_type;

        using coord_type    = std::array<index_type, mdspan_type::rank()>;
        using iterator_type = TensorIterator<Tensor>;

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
        static constexpr bool is_always_unique = mdspan_type::is_always_unique;
        static constexpr bool is_always_exhaustive = mdspan_type::is_always_exhaustive;
        static constexpr bool is_always_strided = mdspan_type::is_always_strided;

        constexpr index_type extent(rank_type r) const noexcept { return _mdspan.extent(r); }
        constexpr index_type size() const noexcept { return _mdspan.size(); }
        constexpr bool empty() const noexcept { return _mdspan.empty(); }
        constexpr index_type stride(rank_type r) const noexcept { return _mdspan.stride(r); }
        constexpr const extents_type& extents() const noexcept { return _mdspan.extents(); }
        constexpr const data_handle_type& data_handle() const noexcept { return _mdspan.data_handle(); }
        constexpr const mapping_type& mapping() const noexcept { return _mdspan.mapping(); }
        constexpr const accessor_type& accessor() const noexcept { return _mdspan.accessor(); }
        constexpr bool is_unique() const noexcept { return _mdspan.is_unique(); }
        constexpr bool is_exhaustive() const noexcept { return _mdspan.is_exhaustive(); }
        constexpr bool is_strided() const noexcept { return _mdspan.is_strided(); }

    //
    // Constructors
    //
    public:
        constexpr Tensor() = default;

        template <typename... IndexTypes>
        explicit constexpr Tensor(data_handle_type p, IndexTypes... extents)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(extents...))) {}

        template <typename OtherExtents>
            requires detail::is_extents_v<OtherExtents>
        constexpr Tensor(data_handle_type p, const OtherExtents& extents)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents)) {}

        template <typename OtherExtents>
            requires detail::is_extents_v<OtherExtents>
        constexpr Tensor(data_handle_type p, const OtherExtents& extents, const coord_type& strides)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents, strides)) {}

        constexpr Tensor(data_handle_type p, const coord_type& extents)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(extents))) {}

        constexpr Tensor(data_handle_type p, const coord_type& extents, const coord_type& strides)
            : _mdspan(accessor_type::write_cast(p), mapping_type(extents_type(extents), strides)) {}

        constexpr Tensor(data_handle_type p, const mapping_type& mapping)
            : _mdspan(accessor_type::write_cast(p), mapping) {}

        constexpr Tensor(const Tensor&) = default;
        constexpr Tensor(Tensor&&) = default;

    // Subtensor constructors
    protected:
        explicit constexpr Tensor(const Tensor& parent, const extents_type& extents)
            : _mdspan(parent._mdspan.data_handle(), mapping_type(parent.mapping().submap(extents)), parent._mdspan.accessor()) {}

        explicit constexpr Tensor(const Tensor& parent, const extents_type& extents, const coord_type& offsets)
            : _mdspan(parent._mdspan.accessor().offset(parent._mdspan.data_handle(), std::apply(parent.mapping(), offsets)), mapping_type(parent.mapping().submap(extents, offsets)), parent._mdspan.accessor()) {}

    public:
        template<typename E>
            requires std::is_same_v<E, extents_type>
        constexpr Tensor subtensor(const extents_type& extents) const {
            return Tensor(*this, extents);
        }

        template<typename E>
            requires std::is_same_v<E, extents_type>
        constexpr Tensor subtensor(const extents_type& extents, const coord_type& offsets) const {
            return Tensor(*this, extents, offsets);
        }

        constexpr Tensor subtensor(const coord_type& extents) const {
            return Tensor(*this, extents_type(extents));
        }

        constexpr Tensor subtensor(const coord_type& extents, const coord_type& offsets) const {
            return Tensor(*this, extents_type(extents), offsets);
        }

    //
    // Modifiers
    // 
    public:
        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }

    //
    // Element access
    //
    public:
        template <typename... Args>
        constexpr reference operator()(Args&&... args) const {
#if MDSPAN_USE_BRACKET_OPERATOR
            return _mdspan[std::forward<Args>(args)...];
#else
            return _mdspan(std::forward<Args>(args)...);
#endif

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
class Tensor : public Tensor<const T, Extents, LayoutPolicy, typename AccessorPolicy::read_accessor_type> {

    public:
        using base_type = Tensor<std::add_const_t<T>, Extents, LayoutPolicy, typename AccessorPolicy::read_accessor_type>;

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
        using value_type   = std::remove_cv_t<T>;

        using extents_type = typename base_type::extents_type;
        using index_type   = typename base_type::index_type;
        using size_type    = typename base_type::size_type;
        using rank_type    = typename base_type::rank_type;

        using layout_type  = typename base_type::layout_type;
        using mapping_type = typename base_type::mapping_type;

        using accessor_type    = AccessorPolicy::write_accessor_type;
        using reference        = typename accessor_type::reference;
        using data_handle_type = typename accessor_type::data_handle_type;

        using coord_type    = typename base_type::coord_type;
        using iterator_type = TensorIterator<Tensor>;

    //
    // Member functions
    //

    //
    // Observers
    //
    public:
        using base_type::rank;
        using base_type::rank_dynamic;
        using base_type::static_extent;
        using base_type::is_always_unique;
        using base_type::is_always_exhaustive;
        using base_type::is_always_strided;

        using base_type::extent;
        using base_type::size;
        using base_type::empty;
        using base_type::stride;
        using base_type::extents;
        using base_type::mapping;
        using base_type::is_unique;
        using base_type::is_exhaustive;
        using base_type::is_strided;

        constexpr const accessor_type& accessor() const noexcept { return _mdspan.accessor(); }
        constexpr const data_handle_type& data_handle() const noexcept { return _mdspan.data_handle(); }

    //
    // Constructors
    //
    public:
        constexpr Tensor() = default;

        template <typename... IndexTypes>
            requires( (std::is_convertible_v<IndexTypes, index_type> && ...) )
        explicit constexpr Tensor(data_handle_type p, IndexTypes... extents)
            : base_type(p, extents...) {}

        template <typename OtherExtents>
            requires detail::is_extents_v<OtherExtents>
        constexpr Tensor(data_handle_type p, const OtherExtents& extents)
            : base_type(p, extents) {}

        template <typename OtherExtents>
            requires detail::is_extents_v<OtherExtents>
        constexpr Tensor(data_handle_type p, const OtherExtents& extents, const coord_type& strides)
            : base_type(p, extents, strides) {}

        constexpr Tensor(data_handle_type p, const coord_type& extents)
            : base_type(p, extents) {}

        constexpr Tensor(data_handle_type p, const coord_type& extents, const coord_type& strides)
            : base_type(p, extents, strides) {}

        constexpr Tensor(data_handle_type p, const mapping_type& mapping)
            : base_type(p, mapping) {}

        constexpr Tensor(const Tensor&) = default;
        constexpr Tensor(Tensor&&) = default;

    // Subtensor constructors
    protected:
        explicit constexpr Tensor(const Tensor& parent, const extents_type& extents)
            : base_type(parent, extents) {}

        explicit constexpr Tensor(const Tensor& parent, const extents_type& extents, const coord_type& offsets)
            : base_type(parent, extents, offsets) {}

    public:
        template<typename E>
            requires std::is_same_v<E, extents_type>
        constexpr Tensor subtensor(const extents_type& extents) const {
           return Tensor(*this, extents);
        }

        template<typename E>
            requires std::is_same_v<E, extents_type>
        constexpr Tensor subtensor(const extents_type& extents, const coord_type& offsets) const {
           return Tensor(*this, extents, offsets);
        }

        constexpr Tensor subtensor(const coord_type& extents) const {
           return Tensor(*this, extents_type(extents));
        }

        constexpr Tensor subtensor(const coord_type& extents, const coord_type& offsets) const {
           return Tensor(*this, extents_type(extents), offsets);
        }

    //
    // Modifiers
    // 
    public:
        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }

    //
    // Operator =
    //
    public:
        // TODO: add runtime debug assert on extents match
        template <typename U>
            requires IsElementwiseExprCompatible<U>
        Tensor& operator=(const U& other) {
            // for (size_type i = 0; i < this->size(); ++i) {
            //    operator[](i) = other[i];
            // }
            // return *this;

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
        Tensor& operator=(const Tensor& other) {
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

        Tensor& operator=(Tensor&& other) {
            this->_mdspan = std::move(other._mdspan);
        }

        Tensor& operator=(base_type&& other) {
            this->_mdspan = std::move(other._mdspan);
        }

    //
    // Element access
    //
    public:
        template <typename... Args>
        constexpr reference operator()(Args&&... args) const {
#if MDSPAN_USE_BRACKET_OPERATOR
            return this->_mdspan[std::forward<Args>(args)...];
#else
            return this->_mdspan(std::forward<Args>(args)...);
#endif
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

#endif // NABLA_TENSOR_TENSOR_HPP
