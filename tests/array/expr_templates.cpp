#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"
#include "nabla/utility.hpp"
#include <iostream>

namespace nb = nabla;

int main() {
    using Ext1 = nb::dims<2>;
    using Ext2 = nb::extents<int, 3, 4>;
    using TArr1 = nb::TensorArray<float, Ext1, nb::LeftStride>;
    using TArr2 = nb::TensorArray<float, Ext2, nb::LeftStride>;

    TArr1 t1{3, 4};
    TArr2 t2{3, 4};

    int i = 0;
    for (auto it = t1.begin(); it != t1.end(); ++i, ++it) {
        *it = i;
    }
    auto expr = t1.to_span()*2 + t1;
    t2 = (t1.to_span()*2 + t1*t1)/t1.to_span() - 2;

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
}
