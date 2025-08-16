#ifndef TENSOR_CONCEPTS_HPP
#define TENSOR_CONCEPTS_HPP

#include <concepts>
#include "nabla/forward_declarations.hpp"

namespace nabla {

template <typename T>
concept layout_rank2 = requires
    { { T::rank } -> std::convertible_to<std::size_t>; }
    && std::is_base_of_v<LayoutTag, T>
    && (T::rank == 2);

} // namespace nabla

#endif // TENSOR_CONCEPTS_HPP
