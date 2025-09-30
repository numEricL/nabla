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
    using TensorArray = nabla::TensorArray<float, Ext1, nabla::LeftStride>;

    std::vector<float> vec(1000);

    // constructor 1
    {
        TensorArray t1{};
    }

    // constructor 2
    {
        TensorArray t1(vec, 3, 4);
        TensorArray t2(vec, 3lu, 4lu);

        TensorArray t3(3, 4);
        TensorArray t4(3lu, 4lu);
    }

    // constructor 3
    {
        TensorArray t1(vec, Ext1{3, 4});
        TensorArray t2(vec, Ext2{3, 4});

        TensorArray t3(Ext1{3, 4});
        TensorArray t4(Ext2{3, 4});
    }

    // constructor 4
    {
        TensorArray t1(vec, Ext1{3, 4}, Arr1{1, 10});
        TensorArray t2(vec, Ext1{3, 4}, {1, 10});
        TensorArray t3(vec, Ext2{3, 4}, {1, 10});

        TensorArray t4(Ext1{3, 4}, Arr1{1, 10});
        TensorArray t5(Ext1{3, 4}, {1, 10});
        TensorArray t6(Ext2{3, 4}, {1, 10});
    }

    // construct 5
    {
        TensorArray t1(vec, Arr1{3, 4});
        TensorArray t2(vec, {3, 4});

        TensorArray t3(Arr1{3, 4});
        TensorArray t4({3, 4});
    }

    // construct 6
    {
        TensorArray t1(vec, Arr1{3, 4}, Arr1{1, 10});
        TensorArray t2(vec, {3, 4}, {1, 10});

        TensorArray t3(Arr1{3, 4}, Arr1{1, 10});
        //TensorArray t4({3, 4}, {1, 10}); // ambiguous with container constructors
    }

    // construct 7
    {
        TensorArray t1(vec, typename TensorArray::mapping_type(Ext1{3, 4}));

        TensorArray t2(typename TensorArray::mapping_type(Ext1{3, 4}));
    }

    // copy/move constructors
    {
        TensorArray t1(vec, 3, 4);
        TensorArray t2(t1);
        TensorArray t3(std::move(t1));
    }

    {
        using Ext1 = Kokkos::dextents<size_t, 2>;
        using Arr1 = std::array<size_t, 2>;
        using TensorArray = nabla::TensorArray<float, Ext1, nabla::LeftStride>;
        TensorArray t1({{ 1, 2, 3 }, { 4, 5, 6 }});
        TensorArray t2 = {{ 1, 2, 3 }, { 4, 5, 6 }};

        TensorArray t3({{ 1, 2, 3 }, { 4, 5, 6 }}, {1, 10});
        TensorArray t4 = {{{ 1, 2, 3 }, { 4, 5, 6 }}, {2, 10}};

        TensorArray t5( {1, 2, 3, 4, 5, 6}, {2, 3});
        TensorArray t6( {1, 2, 3, 4, 5, 6}, {2, 3}, {1, 10});
    }

    return 0;
}
