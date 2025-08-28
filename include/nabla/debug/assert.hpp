#ifndef NABLA_DEBUG_ASSERT_HPP
#define NABLA_DEBUG_ASSERT_HPP

#include "nabla/tensor/tensor.hpp"
namespace nabla {

#ifdef NABLA_DEBUG

template <Rank rank>
void assert_equal(std::array<size_t, rank> a, std::array<size_t, rank> b) {
    for (Rank i = 0; i < rank; ++i) {
        if (a[i] != b[i]) {
            std::stringstream ss;
            ss << "Tensor error: assert_equal failed: a[" << i << "] = " << a[i]
                << " does not match b[" << i << "] = " << b[i] << "\n\n"
                << std::stacktrace::current() << std::endl;
            throw std::out_of_range(ss.str());
        }
    }
}

#else

template <Rank rank>
void assert_equal(std::array<size_t, rank> a, std::array<size_t, rank> b) {}

#endif // NABLA_DEBUG

} // namespace nabla

#endif // NABLA_DEBUG_ASSERT_HPP
