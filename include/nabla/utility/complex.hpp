#ifndef UTILITY_COMPLEX_HPP
#define UTILITY_COMPLEX_HPP

#include <complex>
#include <type_traits>

namespace nabla {
namespace utility {

template <typename fp>
concept IsRealFP = std::is_floating_point_v<fp>;

template<typename T>
    concept IsComplexFP = requires { typename T::value_type; } &&
    std::is_same_v<T, std::complex<typename T::value_type>> &&
    std::is_floating_point_v<typename T::value_type> &&
    !std::is_const_v<typename T::value_type> &&
    !std::is_volatile_v<typename T::value_type>;

template <typename T>
concept IsFP = IsRealFP<T> || IsComplexFP<T>;

namespace detail {

    template <typename T>
    struct real_type;

    template <IsRealFP T>
    struct real_type<T> { using type = T; };

    template <IsComplexFP T>
    struct real_type<T> { using type = typename T::value_type; };

} // namespace detail

template <typename T>
using real_t = typename detail::real_type<T>::type;

template <IsRealFP fp>
fp& real(fp& x) {
    return x;
}

template <IsComplexFP fp>
typename fp::value_type& real(fp& x) {
    return reinterpret_cast<typename fp::value_type(&)[2]>(x)[0];
}

template <IsRealFP fp>
constexpr fp imag(fp& x) {
    return 0.0;
}

template <IsComplexFP fp>
typename fp::value_type& imag(fp& x) {
    return reinterpret_cast<typename fp::value_type(&)[2]>(x)[1];
}

template <IsRealFP fp>
fp conj(fp x) {
    return x;
}

template <IsComplexFP fp>
fp conj(fp x) {
    return std::conj(x);
}

} // namespace utility
} // namespace nabla

#endif // UTILITY_COMPLEX_HPP
