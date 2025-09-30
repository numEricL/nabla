#include "nabla/nabla.hpp"
#include "nabla/tensor_array.hpp"
#include "nabla/ostream.hpp"
#include "nabla/utility.hpp"
#include <iostream>


int main() {
    using Ext1 = Kokkos::extents<int, 3, 4>;
    // using Ext1 = Kokkos::dextents<int, 2>;
    using TArr1 = nabla::TensorArray<float, Ext1, nabla::LeftStride>;

    //using mdarray_type = mdspan_ns::Experimental::mdarray<float, Ext1, nabla::LeftStride>;

    TArr1 t1 = { {1, 2, 3, 4},
                 {5, 6, 7, 8},
                 {9,10,11,12} };

    auto s1 = t1.to_span();
    auto s2 = nabla::subspan(s1, std::pair(1,3), std::pair(0,2));
    auto s3 = nabla::subspan(t1, std::pair(1,3), std::pair(0,2));
    //auto s2 = nabla::subspan(s1, {1,3}, {0,2});

    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
    std::cout << s3 << std::endl;

}
