#ifndef NABLA_LAYOUTS_HPP
#define NABLA_LAYOUTS_HPP

namespace nabla {

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, std::array<T, N> arr) {
    std::string sep = "";
    os << "(";
    for (auto elem : arr) {
        os << sep << elem;
        sep = ", ";
    }
    os << ")";
    return os;
}

} // namespace nabla

#include "nabla/layouts/left_strided.hpp"

#endif // NABLA_LAYOUTS_HPP
