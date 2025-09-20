#include <array>
#include <cstdint>
using std::size_t;

template <typename T, size_t N>
struct Foo {
    std::array<T, N> arr1;
    std::array<T, N> arr2;

    Foo() : arr1{1, 2, 3, 4}, arr2{5, 6, 7, 8} {}
    Foo(const std::array<size_t, 4>& a1, const std::array<size_t, 4>& a2)
        : arr1(a1), arr2(a2) {}

};

int main() {
    Foo<size_t, 4> foo({1, 2, 3, 4}, {5, 6, 7, 8});
}
