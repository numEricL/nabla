#include <iostream>
#include <vector>
#include "nabla/tensor.hpp"

using namespace nabla;

int main() {
    constexpr int rank = 2;
    using LayoutT = layout::LeftStrided<rank>;
    using TensorT = Tensor<int, rank, LayoutT>;

    std::cout << LayoutT({2,3}) << std::endl;
    std::cout << LayoutT({2,3}, {1, 4}) << std::endl;
    std::cout << LayoutT({2,3}, {2, 4}) << std::endl;

    LayoutT layout({2, 3});
    std::vector<int> data(layout.mem_size());

    for (auto i = 0; i < layout.mem_size(); ++i) {
       data[i] = i + 1;
    }
    TensorT t(data, layout);
    std::cout << t({0, 0}) << std::endl;

    std::cout << "tensor: " << std::endl;
    std::cout << t << std::endl;

     if (t.size() != 6) {
        std::cerr << "size() failed\n";
     }
     if (t.dimensions()[0] != 2 || t.dimensions()[1] != 3) {
        std::cerr << "dimensions() failed\n";
     }
     if (t({0, 0}) != 1 || t({1, 0}) != 2 || t({1, 2}) != 6) {
        std::cerr << "operator() failed\n";
     }
     if (t[0] != 1 || t[5] != 6) {
         std::cerr << "operator[] failed\n";
     }
     if (t.pointer() != data.data()) {
        std::cerr << "pointer() failed\n";
     }

     LayoutT::Subscript sub_dims = {1, 2};
     LayoutT::Subscript sub_offset = {1, 0};
     auto sub = t.subtensor(sub_dims, sub_offset);
     if (sub.size() != 2 || sub({0, 0}) != 2 || sub({0, 1}) != 4) {
         std::cerr << "subtensor failed\n";
        return 1;
     }

     std::cout << "All tensor tests passed.\n";
    return 0;
}

