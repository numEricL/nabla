#ifndef NABLA_REFERENCE_GEMM_HPP
#define NABLE_REFERENCE_GEMM_HPP

#include "utility/complex.hpp"

namespace nabla {
namespace reference {

enum class Transpose {
    nontrans,
    trans,
    conjtrans
};

template <typename fp, typename LayoutT = layout::LeftStrided<2>>
using Mat = Tensor<fp, 2, LayoutT>;

//template <typename fp, IsLayout LayoutT>
//void gemm( Mat<const fp, LayoutT> a,
//           Mat<const fp, LayoutT> b,
//           Mat<fp, LayoutT> c,
//           fp alpha = fp(1), fp beta = fp(0),
//           Transpose transa = Transpose::nontrans,
//           Transpose transb = Transpose::nontrans );
//{
//    using index_t = LayoutT::index_type;
//    index_t m = c.dimension<0>();
//    index_t n = c.dimension<1>();
//    index_t k = (transa == Transpose::nontrans)? a.dimension<1> : a.dimension<0>;
//    assert_dimensions(c, {m,n});
//    if (transa == Transpose::nontrans) {
//        assert_equal(a.dimensions, {m,k});
//    } else {
//        assert_equal(a.dimensions, {k,m});
//    }
//    if (transb == Transpose::nontrans) {
//        assert_equal(b.dimensions, {k,n});
//    } else {
//        assert_equal(b.dimensions, {n,k});
//    }
//    assert_equal(a.layout().template strides[0], 1);
//    assert_equal(b.layout().template strides[0], 1);
//    assert_equal(c.layout().template strides[0], 1);
//
//}
//
//
//{
//    using fp = typename tensor_traits<MatA>::element_type;
//    for (index_t row = 0; row < m; ++row) {
//        for (index_t col = 0; col < n; ++col) {
//            fp sum = fp(0);
//            for (index_t p = 0; p < k; ++p) {
//                sum += A[row * lda + p] * B[p * ldb + col];
//            }
//            C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
//        }
//    }
//}

} // namespace reference
} // namespace nabla

#endif // NABLA_REFERENCE_GEMM_HPP
