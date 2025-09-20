#ifndef UTILITY_HELPERS_HPP
#define UTILITY_HELPERS_HPP

#include <string_view>
#include <tuple>

namespace nabla {
namespace utility {

template <typename T>
constexpr std::string_view type_name_func()
{
    using namespace std;
#ifdef __clang__
    std::string_view p = __PRETTY_FUNCTION__;
    return std::string_view(p.data() + 39, p.size() - 39 - 1);
#elif defined(__GNUC__)
    std::string_view p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return std::string_view(p.data() + 41, p.size() - 41 - 1);
#  else
    return std::string_view(p.data() + 54, p.find(';', 54) - 54);
#  endif
#elif defined(_MSC_VER)
    std::string_view p = __FUNCSIG__;
    return std::string_view(p.data() + 89, p.size() - 89 - 7);
#endif
}

template <typename T>
constexpr std::string_view type_name = type_name_func<T>();

template <typename Tuple, typename F, std::size_t... I>
void for_each_impl(Tuple&& t, F&& f, std::index_sequence<I...>) {
    (f(std::get<I>(t)), ...); // fold over comma
}

template <typename Tuple, typename F>
void for_each_in_tuple(Tuple&& t, F&& f) {
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_impl(std::forward<Tuple>(t), std::forward<F>(f), std::make_index_sequence<N>{});
}

} // namespace utility
} // namespace nabla

#endif // UTILITY_HELPERS_HPP
