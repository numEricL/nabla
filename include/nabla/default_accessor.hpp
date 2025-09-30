#ifndef NABLA_DEFAULT_ACCESSOR_HPP
#define NABLA_DEFAULT_ACCESSOR_HPP

namespace nabla {

template <typename T>
class default_accessor<const T> {
    public:
        using offset_policy = default_accessor;
        using element_type = const T;
        using reference = const T&;
        using data_handle_type = const T*;
        using read_accessor_type = default_accessor<const T>;

        using write_handle_type = T*;
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
        friend class TensorSpan;

        static write_handle_type write_cast(data_handle_type p) noexcept {
            return const_cast<write_handle_type>(p);
        }

        write_accessor_type to_write() const noexcept;
};

template <typename T>
class default_accessor : public default_accessor<const T> {
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

template <typename T>
typename default_accessor<const T>::write_accessor_type default_accessor<const T>::to_write() const noexcept {
    return write_accessor_type();
}

} // namespace nabla

#endif // NABLA_DEFAULT_ACCESSOR_HPP
