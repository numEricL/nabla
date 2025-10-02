#pragma once

#include "nabla/nabla.hpp"

namespace nb = nabla;

template <typename T>
class read_only_accessor {
    public:
        using element_type = T;
        using reference = std::add_const_t<T>&;
        using data_handle_type = std::add_const_t<T>*;
        using offset_policy = read_only_accessor;
        using read_accessor_type = read_only_accessor;
        using write_accessor_type = read_accessor_type;

        constexpr read_only_accessor() noexcept = default;

        template <typename OtherElementType>
            requires std::is_convertible_v<OtherElementType(*)[], element_type(*)[]>
            constexpr read_only_accessor(read_only_accessor<OtherElementType>) noexcept {}

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p[i];
        }

        constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
            return p + i;
        }

    private:
        template <typename U, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
        friend class nb::TensorSpan;

        write_accessor_type to_write() const noexcept {
            return *this;
        }
};

template<class T>
class shared_ptr_accessor;

template<class T>
class shared_ptr_accessor<const T> {
    public:
        using element_type = const T;
        using reference = const T&;
        using data_handle_type = std::shared_ptr<T[]>;
        using offset_policy = shared_ptr_accessor;
        using read_accessor_type = shared_ptr_accessor;
        using write_accessor_type = shared_ptr_accessor<T>;

        constexpr shared_ptr_accessor() noexcept = default;

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p.get()[i];
        }

        constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
            return data_handle_type(p, p.get() + i); // aliasing constructor
        }

        write_accessor_type to_write() const noexcept {
            return {};
        }
};

template <typename T>
class shared_ptr_accessor : public shared_ptr_accessor<const T> {
    public:
        using element_type = T;
        using reference = T&;
        using data_handle_type = std::shared_ptr<T[]>;
        using offset_policy = shared_ptr_accessor;
        using read_accessor_type = shared_ptr_accessor<const T>;
        using write_accessor_type = shared_ptr_accessor;

        constexpr shared_ptr_accessor() noexcept = default;

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p.get()[i];
        }

        constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
            return data_handle_type(p, p.get() + i); // aliasing constructor
        }

        write_accessor_type to_write() const noexcept {
            return {};
        }
};
