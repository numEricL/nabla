#include "nabla/nabla.hpp"
#include "nabla/tensor_array.hpp"
#include "nabla/ostream.hpp"
#include <iostream>

int main() {
    using Ext = nabla::dims<2>;
    // using Ext = Kokkos::extents<int, 3, 4>;
    using TArray = nabla::TensorArray<float, Ext, nabla::LeftStrided>;

    TArray t1{3, 4};
    int i = 0;
    for (auto it = t1.begin(); it != t1.end(); ++i, ++it) {
        *it = i;
    }
    auto expr = t1.to_span()*2.0f;

    std::cout << t1 << std::endl;
    std::cout << expr << std::endl;

    std::cout << t1*t1 << std::endl;

}
