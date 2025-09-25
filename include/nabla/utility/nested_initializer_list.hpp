#ifndef NABLA_UTILITY_NESTED_INITIALIZER_LIST_HPP
#define NABLA_UTILITY_NESTED_INITIALIZER_LIST_HPP

#include <initializer_list>

namespace nabla::utility {

namespace detail {

template<typename T, std::size_t N>
struct NestedInitializerListImpl {
    using type = std::initializer_list<typename NestedInitializerListImpl<T, N-1>::type>;
};

template <typename T>
struct NestedInitializerListImpl<T, 0> {
    using type = T;
};

} // namespace detail

template<typename T, std::size_t N>
using NestedInitializerList = typename detail::NestedInitializerListImpl<T, N>::type;

} // namespace nabla::utility

#endif // NABLA_UTILITY_NESTED_INITIALIZER_LIST_HPP
