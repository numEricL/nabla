#ifndef TENSOR_LAYOUTS_LEFT_STRIDED_HPP
#define TENSOR_LAYOUTS_LEFT_STRIDED_HPP

#include <array>
#include <sstream>
#include <stacktrace>
#include <stdexcept>
#include "nabla/forward_declarations.hpp"

namespace nabla {
namespace layout {

template <Rank rank_>
class LeftStrided : LayoutTag {
    public:
        static constexpr Rank rank = rank_;
        using Index = size_t;
        using Subscript = std::array<Index, rank>;
        // using SubscriptConstRef = const Subscript&;
        using SubscriptConstRef = Subscript;

    private:
        Subscript _dimensions;
        Subscript _strides;

        Subscript _default_strides(SubscriptConstRef dimensions) const {
            Subscript strides;
            Index stride = 1;
            for (Rank i = 0; i < rank; ++i) {
                strides[i] = stride;
                stride *= dimensions[i];
            }
            return strides;
        }

        void _assert_index(SubscriptConstRef indices) const {
            for (Rank i = 0; i < rank; ++i) {
                if (indices[i] < 0 || indices[i] >= _dimensions[i]) {
                    std::stringstream ss;
                    Subscript bounds = _dimensions;
                    for (auto& x : bounds) { x--; }
                    ss << "tensor error: _assert_index : index " << i << " is out of bounds\n"
                        << "\tindices:     " << indices << "\n"
                        << "\tupper bound: " << bounds << "\n"
                        << "\n\n"
                        << std::stacktrace::current() << std::endl;
                    throw std::out_of_range(ss.str());
                }
            }
        }

    public:
        Index operator()(SubscriptConstRef indices) const {
            _assert_index(indices);
            Index idx = 0;
            for (Rank i = 0; i < rank; ++i) {
                idx += indices[i] * _strides[i];
            }
            return idx;
        }

        Index flat_index(Index idx) const {
            Subscript indices;
            for (Rank i = rank - 1; i >= 0; --i) {
                indices[i] = idx / _strides[i];
                idx -= indices[i] * _strides[i];
            }
            return this->operator()(indices);
        }

        SubscriptConstRef dimensions() const {
            return _dimensions;
        }

        SubscriptConstRef strides() const {
            return _strides;
        }

        Index size() {
            Index total_size = 1;
            for (Rank i = 0; i < rank; ++i) {
                total_size *= _dimensions[i];
            }
            return total_size;
        }

        Index mem_size() const {
            Subscript last_index;
            for (Rank i = 0; i < rank; ++i) {
                last_index[i] = _dimensions[i] - 1;
            }
            return this->operator()(last_index) + 1;
        }

        void assert_container(Index container_size, Index offset) const {
            if (offset + mem_size() > container_size) {
                std::stringstream ss;
                ss << "LeftStrided error: container size is too small for layout.\n"
                    << "\t requested  = " << offset + mem_size() << " (with offset " << offset << ")"
                    << " but container size is " << container_size
                    << "\n\n"
                    << std::stacktrace::current() << std::endl;
                throw std::out_of_range(ss.str());
            }
        }

        //template <Rank rank>
        //    Subscript
        //    mem_to_subscript(Index idx) const {
        //        Subscript indices;
        //        for (Rank i = rank - 1; i >= 0; --i) {
        //            indices[i] = idx / _strides[i];
        //            idx -= indices[i] * _strides[i];
        //        }
        //        return indices;
        //    }

        // constructors
        LeftStrided() = default;

        LeftStrided(SubscriptConstRef dimensions, SubscriptConstRef strides)
            : _dimensions(dimensions), _strides(strides) {
                for (Rank i = 0; i < rank; ++i) {
                    if (dimensions[i] < 0) {
                        std::stringstream ss;
                        ss << "LeftStrided error: dimensions[" << i << "] = " << dimensions[i] << " must be >= 0."
                            << "\n\n"
                            << std::stacktrace::current() << std::endl;
                        throw std::out_of_range(ss.str());
                    }
                }
                for (Rank i = 0; i < rank; ++i) {
                    if (strides[i] <= 0) {
                        std::stringstream ss;
                        ss << "LeftStrided error: strides[" << i << "] = " << strides[i] << " must be > 0."
                            << "\n\n"
                            << std::stacktrace::current() << std::endl;
                        throw std::out_of_range(ss.str());
                    }
                }
                Index min_stride = 1;
                for (Rank i = 0; i < rank; ++i) {
                    if (strides[i] < min_stride) {
                        std::stringstream ss;
                        ss << "LeftStrided error: strides[" << i << "] = " << strides[i] << " < " << min_stride
                            << " at dimensions[" << i << "] = " << dimensions[i] << ". Stride must be at least 1 for the first dimensions."
                            << "\n\n"
                            << std::stacktrace::current() << std::endl;
                        throw std::out_of_range(ss.str());
                    }
                    min_stride *= dimensions[i];
                }
            }

        LeftStrided(SubscriptConstRef dimensions)
            : LeftStrided(dimensions, _default_strides(dimensions)) {}

        // sub-layout constructor
        LeftStrided(const LeftStrided& layout, SubscriptConstRef dimensions, SubscriptConstRef offset = {})
            : _dimensions(dimensions), _strides(layout._strides) {
                for (Rank i = 0; i < rank; ++i) {
                    if (dimensions[i] < 0 || offset[i] < 0 || offset[i] + dimensions[i] > layout._dimensions[i]) {
                        std::stringstream ss;
                        ss << "tensor error: subtensor : index " << i << " is out of bounds\n"
                            << "\tsub offset:        " << offset << "\n"
                            << "\tsub dimensions:    " << dimensions << "\n"
                            << "\ttensor dimensions: " << _dimensions << "\n"
                            << "\n\n"
                            << std::stacktrace::current() << std::endl;
                        throw std::out_of_range(ss.str());
                    }
                }
            }

        LeftStrided sublayout(SubscriptConstRef dimensions, SubscriptConstRef offset = {}) const {
            return LeftStrided(*this, dimensions, offset);
        }
};

} // namespace layout
} // namespace nabla

#endif // TENSOR_LAYOUT_HPP
