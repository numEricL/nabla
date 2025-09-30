#ifndef NABLA_NESTED_INITIALIZER_LIST_HPP
#define NABLA_NESTED_INITIALIZER_LIST_HPP

#include <initializer_list>
#include "nabla/types.hpp"
#include "mdspan/mdarray.hpp"

namespace nabla {

namespace detail {

template<typename ElementType, std::size_t N>
struct NestedInitializerListImpl {
    using type = std::initializer_list<typename NestedInitializerListImpl<ElementType, N-1>::type>;
};

template <typename ElementType>
struct NestedInitializerListImpl<ElementType, 0> {
    using type = ElementType;
};

} // namespace detail

template<typename ElementType, std::size_t N>
using NestedInitializerList = typename detail::NestedInitializerListImpl<ElementType, N>::type;

namespace detail {
    template <typename ElementType, size_t ListRank, size_t CoordRank>
    constexpr void get_extents_from_initializer_list_impl(NestedInitializerList<ElementType, ListRank> list, std::array<size_t, CoordRank>& exts) {
        exts[CoordRank - ListRank] = list.size();
        if constexpr (ListRank > 1) {
            get_extents_from_initializer_list_impl<ElementType, ListRank-1, CoordRank>(*list.begin(), exts);
        }
    }

    template <typename ElementType, size_t Rank>
    constexpr std::array<size_t, Rank> get_extents_from_initializer_list(NestedInitializerList<ElementType, Rank> list) {
        std::array<size_t, Rank> exts{};
        get_extents_from_initializer_list_impl<ElementType, Rank, Rank>(list, exts);
        return exts;
    }

    template <std::size_t ListRank, typename ElementType, typename Extents, typename LayoutPolicy, typename Container>
    void fill_array_from_initializer_list_impl(
        NestedInitializerList<ElementType, ListRank> list,
        mdspan_ns::Experimental::mdarray<ElementType, Extents, LayoutPolicy, Container>& array,
        std::array<typename Extents::index_type, Extents::rank()>& coord)
    {
        if constexpr (ListRank == 1) {
            std::size_t i = 0;
            for (const auto& value : list) {
                coord[coord.size() - 1] = i;
#if MDSPAN_USE_BRACKET_OPERATOR
                std::apply([&](auto&&... args) { array[args...] = value; }, coord);
#else
                std::apply([&](auto&&... args) { array(args...) = value; }, coord);
#endif // MDSPAN_USE_BRACKET_OPERATOR
                ++i;
            }
        } else {
            for (const auto& sublist : list) {
                fill_array_from_initializer_list_impl<ListRank - 1>(sublist, array, coord);
                coord[coord.size() - ListRank] += 1;
            }
        }
    }


    template <std::size_t ListRank, typename ElementType, typename Extents, typename LayoutPolicy, typename Container>
    void fill_array_from_initializer_list(
        NestedInitializerList<ElementType, ListRank> list,
        mdspan_ns::Experimental::mdarray<ElementType, Extents, LayoutPolicy, Container>& array)
    {
        std::array<typename Extents::index_type, Extents::rank()> coord{};
        fill_array_from_initializer_list_impl<ListRank>(list, array, coord);
    }

} // namespace detail


} // namespace nabla


   //if (exts[i] == 0) {
   //    // If any dimension is zero, all subsequent dimensions must be zero
   //    for (std::size_t j = i + 1; j < Extents::rank(); ++j) {
   //        exts[j] = 0;
   //    }
   //    break;
   //}

#endif // NABLA_NESTED_INITIALIZER_LIST_HPP
