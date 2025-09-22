#include <cassert>
#include <iostream>
#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"
#include "nabla/utility.hpp"

int check(bool condition) {
    if (!condition) {
        std::cerr << "Check failed!\n";
    }
    return !condition;
}

int main() {
    using fp = float;
    using Tensor = nabla::Tensor<fp, nabla::dims<2>, nabla::LeftStrided>;

    // Create two 2D tensors (2x3)
    std::vector<fp> _a(2*3);
    std::vector<fp> _b(2*3);
    std::vector<fp> _c(2*3);
    std::vector<fp> _d(2*3);
    std::vector<fp> _e(2*3);

    Tensor a(_a.data(), {2, 3});
    Tensor b(_b.data(), {2, 3});
    Tensor c(_c.data(), {2, 3});
    Tensor d(_d.data(), {2, 3});
    Tensor e(_e.data(), {2, 3});

    // Fill tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a(i, j) = static_cast<float>(i + j);
            b(i, j) = static_cast<float>(i * j);
        }
    }

    // Elementwise addition
    c = a + b;
    auto expr = a + b + c;
    auto inputs = expr.inputs();

    std::cout << "a b c" << "\n";
    nabla::utility::for_each_in_tuple(inputs, [](const auto& t) {
        std::cout << *t << "\n";
    });

    std::cout << c << "\n";
    std::cout << expr << "\n";


    // Elementwise multiplication
    d = a * b;

    // Negation
    e = -a;

    std::cout << c << "\n";
    std::cout << a + b << "\n";

    int error_count = 0;
    // Check results
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            error_count += check(c(i, j) == a(i, j) + b(i, j));
            error_count += check(d(i, j) == a(i, j) * b(i, j));
            error_count += check(e(i, j) == -a(i, j));
        }
    }

    if (error_count == 0) {
        std::cout << "2D elementwise tests passed.\n";
    } else {
        std::cout << error_count << " checks failed in 2D elementwise tests.\n";
    }
    return 0;
}
