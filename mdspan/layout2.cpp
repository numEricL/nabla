#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0
#include <iostream>
#include <vector>
#include <numeric>
#include "tensor.hpp"
#include "ostream.hpp"

template< std::size_t Rank, typename IndexType = std::size_t >
using DimT = typename mdspan_ns::dextents<IndexType, Rank>;

int main() {
    constexpr int rank = 2;
    using LayoutT = layout::LeftStrided;
    using MappingT = LayoutT::mapping<DimT<rank>>;
    using TensorT = Tensor<int, DimT<2>, LayoutT>;

    MappingT mapping({2,3});
    std::cout << "mapping: " << std::endl;
    std::cout << mapping << std::endl;

    std::cout << MappingT({2,3}, {1, 4}) << std::endl;
    std::cout << MappingT({2,3}, {2, 4}) << std::endl;

    std::vector<int> data(mapping.required_span_size());
    TensorT t(data.data(), mapping);
    std::iota(t.begin(), t.end(), 1); // fill with 1, 2, ..., 6

    std::cout << t(0, 0) << std::endl;

    std::cout << "tensor: " << std::endl;
    std::cout << t << std::endl;
    //
    // if (t.size() != 6) {
    //    std::cerr << "size() failed\n";
    // }
    // if (t.dimensions()[0] != 2 || t.dimensions()[1] != 3) {
    //    std::cerr << "dimensions() failed\n";
    // }
    // if (t(0, 0) != 1 || t(1, 0) != 2 || t(1, 2) != 6) {
    //    std::cerr << "operator() failed\n";
    // }
    // if (t[0] != 1 || t[5] != 6) {
    //     std::cerr << "operator[] failed\n";
    // }
    // if (t.pointer() != data.data()) {
    //    std::cerr << "pointer() failed\n";
    // }
    //
    // LayoutT::subscript_type sub_dims = {1, 2};
    // LayoutT::subscript_type sub_offset = {1, 0};
    // auto sub = t.subtensor(sub_dims, sub_offset);
    // if (sub.size() != 2 || sub(0, 0) != 2 || sub(0, 1) != 4) {
    //     std::cerr << "subtensor failed\n";
    //    return 1;
    // }
    //
    // std::cout << "All tensor tests passed.\n";
    return 0;
}
