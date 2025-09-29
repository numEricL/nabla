#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <vector>
#include <array>
#include <iostream>
#include "mdspan/mdspan.hpp"
#include "mdspan/mdarray.hpp"
#include "nabla/layout/left_stride.hpp"

template <typename MapT>
    requires (MapT::extents_type::rank() == 2)
int access_test(const MapT& map_) {
    MapT map;
    map = map_;

    auto extents = map.extents();
    auto strides = map.strides();


    for (size_t j = 0; j < extents.extent(1); ++j) {
        for (size_t i = 0; i < extents.extent(0); ++i) {
            if ( map(i, j) != i*strides[0] + j*strides[1] ) {
                std::cerr << "Error in operator() at (" << i << ", " << j << "): "
                    << map(i, j) << " != " << (i*strides[0] + j*strides[1]) << std::endl;
                return 1;
            }
        }
    }
    return 0;
}

template <typename MapT>
    requires (MapT::extents_type::rank() == 2)
int iterator_test(const MapT& map_) {
    MapT map;
    map = map_;

    auto extents = map.extents();
    auto strides = map.strides();

    auto it = map.begin();
    for (size_t j = 0; j < extents.extent(1); ++j) {
        for (size_t i = 0; i < extents.extent(0); ++i) {
            if ( *it != i*strides[0] + j*strides[1] ) {
                std::cerr << "Error in iterator at (" << i << ", " << j << "): "
                          << *it << " != " << (i*strides[0] + j*strides[1]) << std::endl;
                return 1;
            }
            ++it;
        }
    }
    return 0;
}

template <typename MapT>
    requires (MapT::extents_type::rank() == 3)
int access_test(const MapT& map) {
    auto extents = map.extents();
    auto strides = map.strides();

    for (size_t k = 0; k < extents.extent(2); ++k) {
        for (size_t j = 0; j < extents.extent(1); ++j) {
            for (size_t i = 0; i < extents.extent(0); ++i) {
                if ( map(i, j, k) != i*strides[0] + j*strides[1] + k*strides[2] ) {
                    std::cerr << "Error in operator() at (" << i << ", " << j << ", " << k << "): "
                              << map(i, j, k) << " != " << (i*strides[0] + j*strides[1] + k*strides[2]) << std::endl;
                    return 1;
                }
            }
        }
    }
    return 0;
}

template <typename MapT>
    requires (MapT::extents_type::rank() == 3)
int iterator_test(const MapT& map) {
    auto extents = map.extents();
    auto strides = map.strides();
    auto it = map.begin();
    for (size_t k = 0; k < extents.extent(2); ++k) {
        for (size_t j = 0; j < extents.extent(1); ++j) {
            for (size_t i = 0; i < extents.extent(0); ++i) {
                if ( *it != i*strides[0] + j*strides[1] + k*strides[2] ) {
                    std::cerr << "Error in iterator at (" << i << ", " << j << ", " << k << "): "
                              << *it << " != " << (i*strides[0] + j*strides[1] + k*strides[2]) << std::endl;
                    return 1;
                }
                ++it;
            }
        }
    }
    return 0;
}

template <size_t Rank>
using Ext = Kokkos::dextents<size_t, Rank>;

template <size_t Rank>
using Arr = std::array<size_t, Rank>;

using Layout = nabla::LeftStride;
// using Layout = Kokkos::layout_stride;

template <size_t Rank>
using Mapping = Layout::mapping<Ext<Rank>>;

int main() {
    int error_count = 0;
    {
        Mapping<2> map(Ext<2>{3, 4}, Arr<2>{1, 10});
        error_count += access_test(map);
        error_count += iterator_test(map);

        //map = map.submap({2, 2}, {1, 1});
        //error_count += access_test(map);
        error_count += iterator_test(map);
    }
    {
        Mapping<3> map(Ext<3>{3, 4, 5}, Arr<3>{1, 10, 100});
        error_count += access_test(map);
        error_count += iterator_test(map);

        //map = map.submap({2, 2, 2}, {1, 1, 1});
        //error_count += access_test(map);
        //error_count += iterator_test(map);
    }

    {
        Mapping<3> map(Ext<3>{2, 3, 4}, Arr<3>{1, 10, 100});
        auto submap = submdspan_mapping(map, Kokkos::strided_slice{0,2,1}, Kokkos::strided_slice{1,3,1}, Kokkos::strided_slice{0,4,1});

        Kokkos::Experimental::mdarray<int, Ext<3>, Layout> arr{map};
        auto span = arr.to_mdspan();
        auto subspan = Kokkos::submdspan(span, Kokkos::strided_slice{0,2,1}, Kokkos::strided_slice{1,3,1}, Kokkos::strided_slice{0,4,1});

    }

    if (error_count == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << error_count << " tests failed." << std::endl;
    }
    return error_count;
}
