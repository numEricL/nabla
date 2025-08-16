#ifndef TENSOR_FORWARD_DECLARATIONS_HPP
#define TENSOR_FORWARD_DECLARATIONS_HPP

namespace nabla {
    using Rank = int;
    struct LayoutTag {};

    template <typename T, Rank rank, typename LayoutT>
    class Tensor;

} // namespace nabla

#endif // TENSOR_FORWARD_DECLARATIONS_HPP
