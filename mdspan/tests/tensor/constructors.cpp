#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <iostream>
#include <vector>
#include <array>
#include <span>
#include "mdspan/mdspan.hpp"
#include "nabla/tensor.hpp"
#include "nabla/ostream.hpp"

int main() {
    using Ext1 = Kokkos::dextents<size_t, 2>;
    using Ext2 = Kokkos::dextents<int, 2>;
    using Arr1 = std::array<size_t, 2>;
    using Tensor = nabla::Tensor<float, Ext1, nabla::LeftStrided>;

    std::vector<float> vec(1000);

    // constructor 1
    {
        Tensor t1{};
    }

    // constructor 2
    {
        Tensor t1(vec.data(), 3, 4);
        Tensor t2(vec.data(), 3lu, 4lu);
    }

    // constructor 3
    {
        Tensor t1(vec.data(), Ext1{3, 4});
        Tensor t2(vec.data(), Ext2{3, 4});
    }

    // constructor 4
    {
        Tensor t1(vec.data(), Ext1{3, 4}, Arr1{1, 10});
        Tensor t3(vec.data(), Ext1{3, 4}, {1, 10});
        Tensor t4(vec.data(), Ext2{3, 4}, {1, 10});
    }

    // construct 5
    {
        Tensor t1(vec.data(), Arr1{3, 4});
        Tensor t2(vec.data(), {3, 4});
    }

    // construct 6
    {
        Tensor t1(vec.data(), Arr1{3, 4}, Arr1{1, 10});
        Tensor t2(vec.data(), {3, 4}, {1, 10});
    }

    // construct 7
    {
        Tensor t1(vec.data(), typename Tensor::mapping_type(Ext1{3, 4}));
    }

    // copy/move constructors
    {
        Tensor t1(vec.data(), 3, 4);
        Tensor t2(t1);
        Tensor t3(std::move(t1));
    }

    return 0;
}
