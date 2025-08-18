#ifndef NABLA_LAYOUTS_LEFT_STRIDED_HPP
#define NABLA_LAYOUTS_LEFT_STRIDED_HPP

#include <array>
#include <sstream>
#include <stacktrace>
#include <stdexcept>
#include "nabla/traits.hpp"

namespace nabla {
namespace layout {

template <Rank rank_>
class LeftStrided : LayoutTag {
    public:
        static constexpr Rank rank = rank_;
        using index_type = size_t;
        using subscript_type = std::array<index_type, rank>;
        // using subscript_cref_type = const subscript_type&;
        using subscript_cref_type = subscript_type;

    private:
        subscript_type _dimensions;
        subscript_type _strides;

        subscript_type _default_strides(subscript_cref_type dimensions) const {
            subscript_type strides;
            index_type stride = 1;
            for (Rank i = 0; i < rank; ++i) {
                strides[i] = stride;
                stride *= dimensions[i];
            }
            return strides;
        }

        void _assert_index(subscript_cref_type indices) const {
            for (Rank i = 0; i < rank; ++i) {
                if (indices[i] < 0 || indices[i] >= _dimensions[i]) {
                    std::stringstream ss;
                    subscript_type bounds = _dimensions;
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
        index_type operator()(subscript_cref_type indices) const {
            _assert_index(indices);
            index_type idx = 0;
            for (Rank i = 0; i < rank; ++i) {
                idx += indices[i] * _strides[i];
            }
            return idx;
        }

        index_type flat_index(index_type idx) const {
            subscript_type indices;
            for (Rank i = rank - 1; i >= 0; --i) {
                indices[i] = idx / _strides[i];
                idx -= indices[i] * _strides[i];
            }
            return this->operator()(indices);
        }

        subscript_cref_type dimensions() const {
            return _dimensions;
        }

        subscript_cref_type strides() const {
            return _strides;
        }

        index_type size() const {
            index_type total_size = 1;
            for (Rank i = 0; i < rank; ++i) {
                total_size *= _dimensions[i];
            }
            return total_size;
        }

        index_type mem_size() const {
            subscript_type last_index;
            for (Rank i = 0; i < rank; ++i) {
                last_index[i] = _dimensions[i] - 1;
            }
            return this->operator()(last_index) + 1;
        }

        void assert_container(index_type container_size, index_type offset) const {
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

        // constructors
        LeftStrided() = default;

        LeftStrided(subscript_cref_type dimensions, subscript_cref_type strides)
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
                index_type min_stride = 1;
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

        LeftStrided(subscript_cref_type dimensions)
            : LeftStrided(dimensions, _default_strides(dimensions)) {}

        // sub-layout constructor
        LeftStrided(const LeftStrided& layout, subscript_cref_type dimensions, subscript_cref_type offset = {})
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

        LeftStrided sublayout(subscript_cref_type dimensions, subscript_cref_type offset = {}) const {
            return LeftStrided(*this, dimensions, offset);
        }
};

} // namespace layout
} // namespace nabla

#endif // NABLA_LAYOUTS_LEFT_STRIDED_HPP
