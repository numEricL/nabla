#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <iostream>
#include <vector>
#include <array>
#include "mdspan/mdspan.hpp"
#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"

int main() {
    using Ext1 = Kokkos::dextents<size_t, 2>;
    using Ext2 = Kokkos::dextents<int, 2>;
    using Arr1 = std::array<size_t, 2>;
    using TensorSpan = nabla::TensorSpan<float, Ext1, nabla::LeftStrided>;

    std::vector<float> vec(1000);

    // constructor 1
    {
        TensorSpan t1{};
    }

    // constructor 2
    {
        TensorSpan t1(vec.data(), 3, 4);
        TensorSpan t2(vec.data(), 3lu, 4lu);
    }

    // constructor 3
    {
        TensorSpan t1(vec.data(), Ext1{3, 4});
        TensorSpan t2(vec.data(), Ext2{3, 4});
    }

    // constructor 4
    {
        TensorSpan t1(vec.data(), Ext1{3, 4}, Arr1{1, 10});
        TensorSpan t3(vec.data(), Ext1{3, 4}, {1, 10});
        TensorSpan t4(vec.data(), Ext2{3, 4}, {1, 10});
    }

    // construct 5
    {
        TensorSpan t1(vec.data(), Arr1{3, 4});
        TensorSpan t2(vec.data(), {3, 4});
    }

    // construct 6
    {
        TensorSpan t1(vec.data(), Arr1{3, 4}, Arr1{1, 10});
        TensorSpan t2(vec.data(), {3, 4}, {1, 10});
    }

    // construct 7
    {
        TensorSpan t1(vec.data(), typename TensorSpan::mapping_type(Ext1{3, 4}));
    }

    // copy/move constructors
    {
        TensorSpan t1(vec.data(), 3, 4);
        TensorSpan t2(t1);
        TensorSpan t3(std::move(t1));
    }

    return 0;
}
