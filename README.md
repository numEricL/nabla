# NABLA
Envisioned to be a suite of C++ tools for numerical algorithms. The short term goals are to extend C++23/26 mdspan and mdarray for PyTorch-esque usability while staying compatible with mdspan APIs.

## What's included now:
A Tensor class built on mdspan and mdarray.

* Element-wise expression templates
* Layout iterators
* Less verbose constructors via brace init
* Template friendly non-const to const argument passing

## template const correct example

```cpp
#include "nabla/nabla.hpp"

namespace nb = nabla;

template <typename T, typename Extents>
void foo(nb::TensorSpan<const T, Extents> span) {}

template <typename T, typename Extents>
void bar(Kokkos::mdspan<const T, Extents> span) {}

int main() {
    std::vector<float> vec(4);
    using Ext = nb::extents<size_t, 2, 2>;
    nb::TensorSpan<float, Ext> t1{vec.data(), 2, 2};
    Kokkos::mdspan<float, Ext> t2{vec.data(), 2, 2};
    foo(t1); // ok
    bar(t2); // error: no matching function for call to `bar(Kokkos::mdspan<float, Kokkos::extents<long unsigned int, 2, 2> >&)`
    return 0;
}
```
