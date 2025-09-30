#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <array>
#include "mdspan/mdspan.hpp"
#include "nabla/layout/left_stride.hpp"

namespace nb = nabla;

int main() {

    //using Ext1 = nb::dextents<size_t, 2>;
    //using Ext2 = nb::dextents<int, 2>;

    using Ext1 = nb::extents<size_t, 3, 4>;
    using Ext2 = nb::extents<int, 3, 4>;
    using Arr1 = std::array<size_t, 2>;
    using Arr2 = std::array<int, 2>;
    using Mapping = nb::LeftStride::mapping<Ext1>;
    //using Mapping = nb::layout_stride::mapping<Ext1>;

    // default constructor
    {
        Mapping map{};
    }

    // constructor 1
    {
        Ext1 extents1{3, 4};
        Arr1 strides1{1, 10};

        Ext2 extents2{3, 4};
        Arr2 strides2{1, 10};

        Mapping map1(extents1, strides1);
        Mapping map2(extents2, strides1);
        //Mapping map3(extents1, strides2); // compiler error
        //Mapping map4(extents2, strides2); // compiler error

        Mapping map5(Ext1{3,4}, Arr1{1, 10});
        Mapping map6(Ext2{3,4}, Arr1{1, 10});
        //Mapping map7(Ext1{3,4}, Arr2{1, 10}); // compiler error
        //Mapping map8(Ext2{3,4}, Arr2{1, 10}); // compiler error

         Mapping map9(Ext1{3,4}, {1, 10});
         Mapping map10(Ext2{3,4}, {1, 10});
    }

    // constructor 2
    {
        Ext1 extents1{3, 4};
        Ext2 extents2{3, 4};

        Mapping map1(extents1);
        Mapping map2(extents2);
        Mapping map3(Ext1{3,4});
        Mapping map4(Ext2{3,4});
    }

    // constructor 3
    {
        Arr1 extents1{3, 4};
        Arr1 strides1{1, 10};

        Arr2 extents2{3, 4};
        Arr2 strides2{1, 10};


        Mapping map1(extents1, strides1);
        Mapping map2(extents1, Arr1{1, 10});
        Mapping map3(extents1, {1, 10});

        Mapping map4(Ext1{3,4}, strides1);
        Mapping map5(Ext1{3,4}, Arr1{1, 10});
        Mapping map6(Ext1{3,4}, {1, 10});

        Mapping map7({3,4}, strides1);
        Mapping map8({3,4}, Arr1{1, 10});
        Mapping map9({3,4}, {1, 10});

        //Mapping map10(extents2, strides1); // compiler error
        //Mapping map11(extents1, strides2); // compiler error
        //Mapping map12(extents2, strides2); // compiler error

        //Mapping map13(Arr2{3,4}, Arr1{1, 10}); // compiler error
        //Mapping map14(Arr1{3,4}, Arr2{1, 10}); // compiler error
        //Mapping map15(Arr2{3,4}, Arr2{1, 10}); // compiler error
    }

    // constructor 4
    {
        Arr1 extents1{3, 4};
        Arr2 extents2{3, 4};

        Mapping map1(extents1);
        //Mapping map2(extents2); // compiler error 
        Mapping map3(Arr1{3,4});
        //Mapping map4(Arr2{3,4}); // compiler error
        Mapping map5({3,4});
    }

    // copy/move constructors
    {
        Mapping map1(Ext1{3,4}, Arr1{1, 10});
        Mapping map2(map1);
        Mapping map3(std::move(map1));
    }
}
