#pragma once

#include "nabla/nabla.hpp"

template <typename T>
class read_only_accessor {
    public:
        using offset_policy = read_only_accessor;
        using element_type = T;
        using reference = std::add_const_t<T>&;
        using data_handle_type = std::add_const_t<T>*;
        using read_accessor_type = read_only_accessor;

        using write_handle_type = data_handle_type;
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
            friend class TensorSpan;

        static write_handle_type write_cast(data_handle_type p) noexcept {
            return p;
        }

        write_accessor_type to_write() const noexcept {
            return *this;
        }
};
