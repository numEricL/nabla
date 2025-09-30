#define MDSPAN_DEBUG
#define MDSPAN_USE_BRACKET_OPERATOR 0

#include <iostream>
#include "mdspan/mdspan.hpp"
#include "mdspan/mdarray.hpp"
#include "nabla/nabla.hpp"

namespace nb = nabla;

template <int Rank>
using Ext = nb::dextents<size_t, Rank>;
using Layout = nb::LeftStride;
template <int Rank>
using Mapping = Layout::mapping<Ext<Rank>>;

template <typename fp, int Rank>
using Arr = nb::TensorArray<fp, Ext<Rank>, Layout>;

using ss = nb::strided_slice<int, int, int>;
int main() {
    using fp = float;

    Arr<fp, 3> arr(std::array<int, 3>{4, 5, 6}, {1, 10, 100});

    fp counter = 0;
    for (auto i = arr.begin(); i != arr.end(); ++i) {
        *i = counter++;
    }
    auto subspan = nb::subspan(arr, ss{1}, ss{1,5,1}, ss{1,6,1});
    //std::cout << subspan << std::endl;
    return 0;
}
