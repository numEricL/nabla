#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <iostream>
#include <vector>
#include <array>
#include "mdspan/mdspan.hpp"
#include "nabla/nabla.hpp"
#include "nabla/ostream.hpp"
#include "read_only.hpp"

namespace nb = nabla;

using Ext1 = nb::dextents<size_t, 2>;
using Ext2 = nb::dextents<int, 2>;
using Arr1 = std::array<size_t, 2>;
using TSpan = nb::TensorSpan<float, Ext1, nb::LeftStride>;
//using TSpan = nb::TensorSpan<float, Ext1, nb::LeftStride, read_only_accessor<float>>;
// using TSpan = nb::TensorSpan<float, Ext1, nb::layout_stride>;

template class nb::TensorSpan<float, Ext1, nb::LeftStride>;
template class nb::TensorSpan<const float, nb::extents<int, 2, 3, 4>, nb::LeftStride, read_only_accessor<float>>;

int main() {


    std::vector<float> vec(1000);

    // constructor 1
    {
        TSpan t1{};
    }

    // constructor 2
    {
        TSpan t1(vec.data(), 3, 4);
        TSpan t2(vec.data(), 3lu, 4lu);
    }

    // constructor 3
    {
        TSpan t1(vec.data(), Ext1{3, 4});
        TSpan t2(vec.data(), Ext2{3, 4});
    }

    // constructor 4
    {
        TSpan t1(vec.data(), Ext1{3, 4}, Arr1{1, 10});
        TSpan t3(vec.data(), Ext1{3, 4}, {1, 10});
        TSpan t4(vec.data(), Ext2{3, 4}, {1, 10});
    }

    // construct 5
    {
        TSpan t1(vec.data(), Arr1{3, 4});
        TSpan t2(vec.data(), {3, 4});
    }

    // construct 6
    {
        TSpan t1(vec.data(), Arr1{3, 4}, Arr1{1, 10});
        TSpan t2(vec.data(), {3, 4}, {1, 10});
    }

    // construct 7
    {
        TSpan t1(vec.data(), typename TSpan::mapping_type(Ext1{3, 4}));
    }

    // copy/move constructors
    {
        TSpan t1(vec.data(), 3, 4);
        TSpan t2(t1);
        TSpan t3(std::move(t1));
    }

    return 0;
}
