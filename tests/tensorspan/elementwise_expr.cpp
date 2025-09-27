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
    using Map = nabla::LeftStrided::mapping<nabla::dims<2>>;
    using TensorSpan = nabla::TensorSpan<fp, nabla::dims<2>, nabla::LeftStrided>;
    using tt = typename TensorSpan::value_type;

    Map map({2,3}, {2, 10});

    // Create two 2D tensors (2x3)
    std::vector<fp> _a(map.required_span_size());
    std::vector<fp> _b(map.required_span_size());
    std::vector<fp> _c(map.required_span_size());
    std::vector<fp> _d(map.required_span_size());
    std::vector<fp> _e(map.required_span_size());

    TensorSpan a(_a.data(), {2, 3});
    TensorSpan b(_b.data(), map);
    TensorSpan c(_c.data(), {2, 3});
    TensorSpan d(_d.data(), {2, 3});
    TensorSpan e(_e.data(), {2, 3});

    // Fill tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a(i, j) = static_cast<float>(i + j);
            b(i, j) = static_cast<float>(i + j);
        }
    }


    std::cout << "TensorSpan a:\n" << a << "\n";
    std::cout << "TensorSpan b:\n" << b << "\n";

    std::cout << "a*2:\n" << a*2 << "\n";
    // Elementwise addition
    c = a + b;
    auto expr = a + b + c;
    auto inputs = expr.inputs();

    //std::cout << expr << std::endl;
    //std::cout << "2*expr:\n" << fp(2) * expr << "\n";

    auto iter = expr.begin();
    for (size_t i = 0; i < 6; ++i) {
        std::cout << *iter << " ";
        ++iter;
    }
    std::cout << std::endl;

    for (auto it = expr.begin(); it != expr.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

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
