// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs index_put(const NodeContext& node) {
    auto x = node.get_input("X");
    auto value = node.get_input("vale");
    auto indices = node.get_input("indices");

    auto accumulate = node.get_attribute<bool>("accumulate");

    if (accumulate) {
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset15::ScatterNDUpdate>(x,
                                                            indices,
                                                            value,
                                                            ov::opset15::ScatterNDUpdate::Reduction::SUM)},
            {"Out"});
    } else {
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset15::ScatterNDUpdate>(x, indices, value)},
            {"Out"});
    }
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
