#ifndef NABLA_OSTREAM_HPP
#define NABLA_OSTREAM_HPP

#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "nabla/layout/left_stride.hpp" // remove once we support all layouts

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
    } else {
        return Format(1, 9, 16, 12);
    }
}

template <typename T, typename MatT>
    requires (MatT::extents_type::rank() == 2)
void print2d(std::ostream& os, const MatT& mat) {
    auto fmt = get_format<T>();

    auto m = mat.extents().extent(0);
    auto n = mat.extents().extent(1);

    std::cout << "shape: (" << m << ", " << n << ")  stride: (" << mat.stride(0) << ", " << mat.stride(1) << ")\n";
    for (size_t row = 0; row < std::min<int>(m, fmt.max_rows); ++row) {
        os << "[";
        for (size_t col = 0; col < std::min<int>(n, fmt.max_cols); ++col) {
            os << std::scientific << std::setprecision(fmt.precision)
                      << std::setw(fmt.width) << mat(row, col);
            if (col < n - 1 && col < fmt.max_cols - 1) {
                os << ", ";
            }
        }
        if (n > fmt.max_cols) {
            os << "  ..]";
        } else {
            os << "  ]";
        }
        os << "\n";
    }
    if (m > fmt.max_rows) {
        os << "  ..\n";
    }
}

} // namespace detail


// TODO: enable for all layouts
template <typename Extents>
std::ostream& operator<<(std::ostream& os, const typename LeftStride::mapping<Extents>& mapping) {
    using MappingT = typename LeftStride::mapping<Extents>;
    using Index = MappingT::index_type;

    detail::print2d<Index>(os, mapping);
    return os;
}

template <typename TensorT>
    requires IsTensorLike<TensorT> && (TensorT::extents_type::rank() == 2)
std::ostream& operator<<(std::ostream& os, const TensorT& mat) {
    using T = typename TensorT::value_type;

    detail::print2d<T>(os, mat);
    return os;
}

} // namespace nabla

#endif // NABLA_OSTREAM_HPP
