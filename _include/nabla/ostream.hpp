#ifndef NABLA_OSTREAM_HPP
#define NABLA_OSTREAM_HPP

#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include "nabla/tensor/tensor.hpp"
#include "utility/complex.hpp"

namespace nabla {

namespace detail {

struct Format {
    int precision;
    int width;
    int max_rows;
    int max_cols;
};

template <typename T>
Format get_format() {
    if constexpr (std::is_integral_v<T>) {
        return Format(0, 6, 16, 12);
    } else if constexpr (utility::IsComplexFP<T>) {
        return Format(1, 22, 16, 7);
    } else {
        return Format(1, 9, 16, 12);
    }
}

template <typename intT>
requires std::is_integral_v<intT>
intT shrink(intT x, int64_t = 0) {
    return x;
}

template <typename rfp>
requires std::is_floating_point_v<rfp>
rfp shrink(rfp x, int64_t eps_factor = 100) {
    auto eps = std::numeric_limits<utility::real_t<rfp> >::epsilon();
    return ( std::abs(x) < eps_factor*eps )? 0.0f : x;
}

template <typename cfp>
requires utility::IsComplexFP<cfp>
cfp shrink(cfp x, int64_t eps_factor = 100) {
    return {shrink(utility::real(x), eps_factor), shrink(utility::imag(x), eps_factor)};
}

} // namespace detail

template <typename LayoutT>
requires IsLayoutRankN<LayoutT,2>
std::ostream& operator<<(std::ostream& out, const LayoutT& layout) {
    using Index = layout_traits<LayoutT>::index_type;
    auto fmt = detail::get_format<Index>();

    auto m = layout.dimensions()[0];
    auto n = layout.dimensions()[1];

    for (Index row = 0; row < std::min<int>(m, fmt.max_rows); ++row) {
        out << "[";
        for (Index col = 0; col < std::min<int>(n, fmt.max_cols); ++col) {
            out << std::scientific << std::setprecision(fmt.precision)
               << std::setw(fmt.width) << layout({row, col});
            if (col < n - 1 && col < fmt.max_cols - 1) {
                out << ", ";
            }
        }
        if (n > fmt.max_cols) {
            out << "  ..]";
        } else {
            out << "  ]";
        }
        out << "\n";
    }
    if (m > fmt.max_rows) {
        out << "  ..\n";
    }
    return out;
}

template <typename T>
    requires IsExprCompatible<T> && IsRankN<T, 2>
std::ostream& operator<<(std::ostream& out, const T& t) {
    using Index = T::index_type;
    auto fmt = detail::get_format<T>();

    auto m = t.dimensions()[0];
    auto n = t.dimensions()[1];

    for (Index row = 0; row < std::min<int>(m, fmt.max_rows); ++row) {
        out << "[";
        for (Index col = 0; col < std::min<int>(n, fmt.max_cols); ++col) {
            out << std::scientific << std::setprecision(fmt.precision)
               << std::setw(fmt.width) << t(row, col);
            if (col < n - 1 && col < fmt.max_cols - 1) {
                out << ", ";
            }
        }
        if (n > fmt.max_cols) {
            out << "  ..]";
        } else {
            out << "  ]";
        }
        out << "\n";
    }
    if (m > fmt.max_rows) {
        out << "  ..\n";
    }
    return out;
}

} // namespace nabla

#endif // NABLA_OSTREAM_HPP
