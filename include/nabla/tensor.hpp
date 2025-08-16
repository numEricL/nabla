#ifndef TENSOR_TENSOR_HPP
#define TENSOR_TENSOR_HPP

#include <vector>
#include "nabla/forward_declarations.hpp"
#include "nabla/concepts.hpp"
#include "nabla/layout.hpp"

namespace nabla {

template <typename T, Rank rank_, typename LayoutT = layout::LeftStrided<rank_>>
class Tensor;

template <typename T, Rank rank, typename LayoutT>
class Tensor<const T, rank, LayoutT> {
    public:
        using value_type = const T;
        using Index = LayoutT::Index;
        using Subscript = LayoutT::Subscript;
        using SubscriptConstRef = LayoutT::SubscriptConstRef;

    protected:
        T* _ptr = nullptr;
        Index _offset = 0;
        LayoutT _layout;

    public:

        Index size() {
            return _layout.size();
        }

        Subscript dimensions() const {
            return _layout.dimensions();
        }

        Index offset() const {
            return _offset;
        }

        template <Rank r>
        Index dimension() const {
            static_assert(r < rank, "Dimension out of bounds");
            return _layout.dimensions()[r];
        }

        // constructors
        Tensor() = default;
        Tensor(std::nullptr_t) : Tensor() {}
        Tensor(const T* ptr, LayoutT layout, Index offset = 0)
            : _ptr(const_cast<T*>(ptr)), _offset(offset), _layout(layout) {}

        Tensor(const std::vector<T>& vec, LayoutT layout, Index offset = 0)
            : Tensor(vec.data(), layout, offset) {
                layout.assert_container(vec.size(), _offset);
            }

        // subtensor constructors
        Tensor(Tensor t, SubscriptConstRef dimensions, SubscriptConstRef offset = {})
            : _ptr(t._ptr), _offset(t._offset + t._layout(offset)), _layout(t._layout, dimensions, offset) {}

        Tensor subtensor(SubscriptConstRef dimensions, SubscriptConstRef offset = {}) const {
            return Tensor(*this, dimensions, offset);
        }

        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }
        void shrink(SubscriptConstRef dimensions, SubscriptConstRef offset = {}) {
            swap(subtensor(dimensions, offset));
        }

        const T* pointer() const {
            return (_ptr ? _ptr + _offset : nullptr);
        }

        const T* data_pointer() const {
            return _ptr;
        }

        const T& operator()(SubscriptConstRef indices) const {
            return pointer()[_layout(indices)];
        }

        const T& operator[](Index idx) const {
            return pointer()[_layout.flat_index(idx)];
        }
};

template <typename T, Rank rank, typename LayoutT>
class Tensor : public Tensor<const T, rank, LayoutT> {
    public:
        using value_type = T;
        using Index = LayoutT::Index;
        using Subscript = LayoutT::Subscript;
        using SubscriptConstRef = LayoutT::SubscriptConstRef;

    protected:
        using base_t = Tensor<const T, rank, LayoutT>;
        using base_t::_ptr;
        using base_t::_offset;
        using base_t::_layout;

    public:
        // Constructors
        Tensor() = default;
        Tensor(std::nullptr_t) : Tensor() {}
        Tensor(T* ptr, LayoutT layout, Index offset = 0)
            : base_t(ptr, layout, offset) {}
        Tensor(std::vector<T>& vec, LayoutT layout, Index offset = 0)
            : base_t(vec, layout, offset) {}

        // subtensor constructors
        Tensor(Tensor t, SubscriptConstRef dimensions, SubscriptConstRef offset = {})
            : base_t(t, dimensions, offset) {}

        Tensor subtensor(SubscriptConstRef dimensions, SubscriptConstRef offset = {}) const {
            return Tensor(*this, dimensions, offset);
        }

        void swap(Tensor& t) { std::swap(*this, t); }
        void swap(Tensor&& t) { std::swap(*this, t); }
        void shrink(SubscriptConstRef dimensions, SubscriptConstRef offset = {}) {
            swap(subtensor(dimensions, offset));
        }

        T* pointer() const {
            return (_ptr ? _ptr + _offset : nullptr);
        }

        T* data_pointer() const {
            return _ptr;
        }

        T& operator()(SubscriptConstRef indices) {
            return pointer()[_layout(indices)];
        }

        T& operator[](Index idx) {
            return pointer()[_layout.flat_index(idx)];
        }
};

} // namespace nabla

#include "./ostream.hpp"

#endif // TENSOR_TENSOR_HPP
