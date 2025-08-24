#include <cassert>
#include <iostream>
#include "nabla/nabla.hpp"

int main() {
    using nabla::Tensor;
    using fp = float;

    // Create two 2D tensors (2x3)
    std::vector<fp> _a(2*3);
    std::vector<fp> _b(2*3);
    std::vector<fp> _c(2*3);
    std::vector<fp> _d(2*3);
    std::vector<fp> _e(2*3);

    Tensor<fp, 2> a(_a.data(), {2, 3});
    Tensor<fp, 2> b(_b.data(), {2, 3});
    Tensor<fp, 2> c(_c.data(), {2, 3});
    Tensor<fp, 2> d(_d.data(), {2, 3});
    Tensor<fp, 2> e(_e.data(), {2, 3});

    // Fill tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a(i, j) = static_cast<float>(i + j);
            b(i, j) = static_cast<float>(i * j);
        }
    }

    // Elementwise addition
    c = a + b;
    std::cout << c << "\n";
    std::cout << a + b << "\n";

    //// Elementwise multiplication
    //d = a * b;
    //
    //// Negation
    //e = -a;
    //
    //std::cout << c << "\n";
    ////std::cout << a + b << "\n";
    //
    //// Check results
    //for (size_t i = 0; i < 2; ++i) {
    //    for (size_t j = 0; j < 3; ++j) {
    //        assert(c(i, j) == a(i, j) + b(i, j));
    //        assert(d(i, j) == a(i, j) * b(i, j));
    //        assert(e(i, j) == -a(i, j));
    //    }
    //}
    //
    //std::cout << "2D elementwise tests passed.\n";
    return 0;
}
