#include <vector>
#include <array>
#include <iostream>
#include "mdspan/mdspan.hpp"
#include "nabla/left_strided.hpp"
#include "nabla/ostream.hpp"

template< std::size_t Rank, typename IndexType = std::size_t >
using dims = typename Kokkos::dextents<IndexType, Rank>;

//template <typename T, class Extents, class Layout>
//void print_mdspan(const Kokkos::mdspan<T, Extents, Layout>& arr) {
//    for (std::size_t i = 0; i < arr.extent(0); ++i) {
//        for (std::size_t j = 0; j < arr.extent(1); ++j) {
//            std::cout << arr[i, j] << " ";
//        }
//        std::cout << std::endl;
//    }
//}

void rank_2_example() {
    std::cout << "Rank 2 example" << std::endl;

    constexpr size_t rows = 3;
    constexpr size_t cols = 4;

    dims<2> extents{rows, cols};
    std::array<size_t, 2> strides{1, 10};

    std::vector<int> data(100); // plenty of room
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = -1; // initialize to -1
    }

    // using layout_t = Kokkos::layout_stride;
    using layout_t = nabla::LeftStrided;


    layout_t::mapping layout(extents, strides);
    Kokkos::mdspan<int, dims<2>, layout_t> mat(data.data(), layout);


    // Fill the matrix in column-major logical order
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            mat[i, j] = static_cast<int>(i + j * 100);
        }
    }

    std::cout << "layout: " << std::endl;
    std::cout << layout << std::endl << std::endl;


    for (auto it = layout.begin(); it != layout.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

void rank_3_example() {
    std::cout << "Rank 3 example" << std::endl;

    constexpr size_t dim1 = 2;
    constexpr size_t dim2 = 2;
    constexpr size_t dim3 = 2;

    dims<3> extents{dim1, dim2, dim3};
    std::array<size_t, 3> strides{1, 10, 100};

    std::vector<int> data(1000); // plenty of room
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = -1; // initialize to -1
    }

    using layout_t = layout::LeftStrided;

    layout_t::mapping layout(extents, strides);
    Kokkos::mdspan<int, dims<3>, layout_t> mat(data.data(), layout);

    // Fill the matrix in column-major logical order
    for (size_t k = 0; k < dim3; ++k) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t i = 0; i < dim1; ++i) {
                mat[i, j, k] = static_cast<int>(i + j * 100 + k * 1000);
            }
        }
    }

    std::cout << "layout: " << std::endl;

    std::cout << "Flat indices: " << std::endl;
    std::cout << *layout.begin() << " ... " << *(layout.end()) - 1 << std::endl;
    std::cout << *layout.begin() << " " << *(++layout.begin()) << " " << *(++(++layout.begin())) << std::endl;

    for (auto it = layout.begin(); it != layout.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

int main() {

    rank_2_example();
    rank_3_example();

    //print_mdspan(mat);
}
