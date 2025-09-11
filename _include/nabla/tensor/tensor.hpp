#ifndef NABLA_TENSOR_TENSOR_HPP
#define NABLA_TENSOR_TENSOR_HPP

#include <vector>
#include "nabla/layouts.hpp"
#include "nabla/traits.hpp"
#include "nabla/tensor/iterator.hpp"

namespace nabla {

template <typename T, Rank rank, typename LayoutT = layout::LeftStrided<rank>>
    requires IsLayoutRankN<LayoutT, rank>
class Tensor;

template <typename T, Rank rank_, typename LayoutT>
    requires IsLayoutRankN<LayoutT, rank_>
class Tensor<const T, rank_, LayoutT> {
    public:
        static constexpr Rank rank = rank_;
        using index_type = typename tensor_traits<Tensor>::index_type;
        using subscript_type = typename tensor_traits<Tensor>::subscript_type;
        using subscript_cref_type = typename tensor_traits<Tensor>::subscript_cref_type;

    protected:
        T* _ptr = nullptr; // class must ensure const-like access
        index_type _offset = 0;
        LayoutT _layout;

    public:

        const LayoutT& layout() const {
            return _layout;
        }

        index_type size() const {
            return _layout.size();
        }

        subscript_type dimensions() const {
            return _layout.dimensions();
        }

        index_type offset() const {
            return _offset;
        }

        template <Rank r>
        index_type dimension() const {
            return _layout.template dimension<r>();
        }

        // constructors
        Tensor() = default;
        Tensor(std::nullptr_t) : Tensor() {}
        Tensor(const T* ptr, LayoutT layout, index_type offset = 0)
            : _ptr(const_cast<T*>(ptr)), _offset(offset), _layout(layout) {}
        Tensor(const T* ptr, subscript_cref_type dimensions, index_type offset = 0)
            : _ptr(const_cast<T*>(ptr)), _offset(offset), _layout(dimensions) {}
        Tensor(const std::vector<T>& vec, LayoutT layout, index_type offset = 0)
            : Tensor(vec.data(), layout, offset) {
                layout.assert_container(vec.size(), _offset);
            }

        // subtensor constructors
        Tensor(Tensor t, subscript_cref_type dimensions, subscript_cref_type offset = {})
            : _ptr(t._ptr), _offset(t._offset + t._layout(offset)), _layout(t._layout, dimensions, offset) {}

        Tensor subtensor(subscript_cref_type dimensions, subscript_cref_type offset = {}) const {
            return Tensor(*this, dimensions, offset);
        }

        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }
        void shrink(subscript_cref_type dimensions, subscript_cref_type offset = {}) {
            swap(subtensor(dimensions, offset));
        }

        const T* pointer() const {
            return (_ptr ? _ptr + _offset : nullptr);
        }

        const T* data_pointer() const {
            return _ptr;
        }

        template <typename... Args>
            requires (sizeof...(Args) == rank_) &&
            (std::conjunction_v<std::is_convertible<Args, index_type>...>)
        const T& operator()(Args... args) const {
            subscript_type indices{static_cast<index_type>(std::forward<Args>(args))...};
            return pointer()[_layout(indices)];
        }

        const T& operator[](index_type idx) const {
            return pointer()[_layout.flat_index(idx)];
        }

        using Iterator = TensorIterator<Tensor, true>;
        Iterator begin() const {
            return Iterator(*this, 0);
        }
        Iterator end() const {
            return Iterator(*this, size());
        }
};

template <typename T, Rank rank_, typename LayoutT>
    requires IsLayoutRankN<LayoutT, rank_>
class Tensor : public Tensor<const T, rank_, LayoutT> {
    public:
        static constexpr Rank rank = rank_;
        using index_type = typename tensor_traits<Tensor>::index_type;
        using subscript_type = typename tensor_traits<Tensor>::subscript_type;
        using subscript_cref_type = typename tensor_traits<Tensor>::subscript_cref_type;

    protected:
        using base_t = Tensor<const T, rank_, LayoutT>;
        using base_t::_ptr;
        using base_t::_offset;
        using base_t::_layout;

    public:
        // Constructors
        Tensor() = default;
        Tensor(std::nullptr_t) : Tensor() {}
        Tensor(T* ptr, LayoutT layout, index_type offset = 0)
            : base_t(ptr, layout, offset) {}
        Tensor(T* ptr, subscript_cref_type dimensions, index_type offset = 0)
            : base_t(ptr, dimensions, offset) {}
        Tensor(std::vector<T>& vec, LayoutT layout, index_type offset = 0)
            : base_t(vec, layout, offset) {}

        // subtensor constructors
        Tensor(Tensor t, subscript_cref_type dimensions, subscript_cref_type offset = {})
            : base_t(t, dimensions, offset) {}

        Tensor subtensor(subscript_cref_type dimensions, subscript_cref_type offset = {}) const {
            return Tensor(*this, dimensions, offset);
        }

        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }
        using base_t::shrink;

        T* pointer() const {
            return (_ptr ? _ptr + _offset : nullptr);
        }

        T* data_pointer() const {
            return _ptr;
        }

        template <typename... Args>
            requires (sizeof...(Args) == rank_) &&
            (std::conjunction_v<std::is_convertible<Args, index_type>...>)
        T& operator()(Args... args) const {
            subscript_type indices{static_cast<index_type>(std::forward<Args>(args))...};
            return pointer()[_layout(indices)];
        }

        T& operator[](index_type idx) const {
            return pointer()[_layout.flat_index(idx)];
        }

        using Iterator = TensorIterator<Tensor, false>;
        Iterator begin() const {
            return Iterator(*this, 0);
        }
        Iterator end() const {
            return Iterator(*this, this->size());
        }

        // unoptimized assignement operator
        template <typename U>
            requires IsExprCompatible<U>
        Tensor& operator=(const U& other) {
            if (reinterpret_cast<const void*>(this) == reinterpret_cast<const void*>(&other)) {
                return *this;
            }
            if (this->size() != other.size()) {
                throw std::runtime_error("Tensor assignment: size mismatch");
            }
            for (index_type i = 0; i < this->size(); ++i) {
                operator[](i) = other[i];
            }
            return *this;
        }

};

} // namespace nabla

#endif // NABLA_TENSOR_TENSOR_HPP
