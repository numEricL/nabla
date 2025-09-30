#ifndef NABLA_SUBSPAN_HPP
#define NABLA_SUBSPAN_HPP

namespace nabla {

template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy, typename... SliceSpecifiers>
constexpr auto
subspan(const TensorSpan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& src,
        SliceSpecifiers... slices) {
    const auto sub_submdspan_mapping_result = submdspan_mapping(src.mapping(), slices...);
    using sub_mapping_t = std::remove_cv_t<decltype(sub_submdspan_mapping_result.mapping)>;
    using sub_extents_t = typename sub_mapping_t::extents_type;
    using sub_layout_t = typename sub_mapping_t::layout_type;
    using sub_accessor_t = typename AccessorPolicy::offset_policy;
    return TensorSpan<ElementType, sub_extents_t, sub_layout_t, sub_accessor_t>(
        src.accessor().offset(src.data_handle(), sub_submdspan_mapping_result.offset),
        sub_submdspan_mapping_result.mapping,
        sub_accessor_t(src.accessor()));
}

template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename Container, typename... SliceSpecifiers>
constexpr auto
subspan(const TensorArray<ElementType, Extents, LayoutPolicy, Container>& src,
        SliceSpecifiers... slices) {
    return subspan(src.to_span(), slices...);
}

} // namespace nabla

#endif //NABLA_SUBSPAN_HPP
