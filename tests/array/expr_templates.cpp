#include "nabla/nabla.hpp"
#include "nabla/tensor_array.hpp"
#include "nabla/ostream.hpp"
#include "nabla/utility.hpp"
#include <iostream>

int main() {
    using Ext1 = nabla::dims<2>;
    using Ext2 = Kokkos::extents<int, 3, 4>;
    using TArr1 = nabla::TensorArray<float, Ext1, nabla::LeftStrided>;
    using TArr2 = nabla::TensorArray<float, Ext2, nabla::LeftStrided>;

    TArr1 t1{3, 4};
    TArr2 t2{3, 4};

    int i = 0;
    for (auto it = t1.begin(); it != t1.end(); ++i, ++it) {
        *it = i;
    }
    auto expr = t1.to_span()*2 + t1;
    t2 = (t1.to_span()*2 + t1*t1)/t1.to_span() - 2;

    std::cout << nabla::utility::type_name<decltype(expr)>() << std::endl;

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;

}
