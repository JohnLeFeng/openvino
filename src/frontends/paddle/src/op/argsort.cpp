// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs argsort(const NodeContext& node) {
    auto data = node.get_input("X");
    auto axis = node.get_attribute<int32_t>("axis");
    if (axis < 0) {
        PADDLE_OP_CHECK(node, data.get_partial_shape().rank().is_static(), "Argsort rank must be static");
        auto data_rank = data.get_partial_shape().rank().get_length();
        axis = static_cast<uint32_t>(data_rank + axis);
    }
    std::string mode = node.get_attribute<bool>("descending") ? "max" : "min";
    std::string sort_type = "value";

    // auto mode = node.get_attribute<bool>("descending") ? ov::op::TopKMode::MAX : ov::op::TopKMode::MIN;
    auto stable = node.get_attribute<bool>("stable");

    // Output<Node> k_expected_node;
    // const auto k_expected = node.get_attribute<int>("k", 1);
    // k_expected_node = default_opset::Constant::create(element::i32, {}, {k_expected});
    
    auto zero_axis = default_opset::Constant::create(element::i32, Shape{1}, {0});
    auto dim_axis = default_opset::Constant::create(element::i64, Shape{1}, {axis});
    auto shape = std::make_shared<default_opset::ShapeOf>(data);
    auto k_values_node = std::make_shared<default_opset::Gather>(shape, dim_axis, zero_axis);
    auto k_values = std::make_shared<default_opset::Squeeze>(k_values_node);

    auto node_topk = std::make_shared<default_opset::TopK>(data, k_values, axis, mode, sort_type, element::i64);
    
    // return node.default_single_output_mapping(
    //     {node_topk->output(1)},
    //     {"Out"});


    // return node.default_single_output_mapping({std::make_shared<default_opset::TopK>(data, k_values, axis, mode, sort_type, element::i64)}, {"Out"});

    // const auto output_info = node.get_output_port_infos("Out");
    // size_t output_size = output_info[0].second.size();
    // if (output_size == 0) {
    //     auto out = std::make_shared<ov::opset6::Squeeze>(node_topk->output(1));
    //     return node.default_single_output_mapping({std::make_shared<ov::opset6::Convert>(out, dtype)}, {"Out"});
    // } else {
    //     return node.default_single_output_mapping(
    //         {std::make_shared<ov::opset6::Convert>(node_topk->output(1), dtype)},
    //         {"Out"});
    // }

    // return node.default_single_output_mapping(
    //     {std::make_shared<default_opset::Convert>(node_topk->output(1), element::i64)},
    //     {"Out"});

    NamedOutputs named_outputs;
    named_outputs["Out"] = OutputVector{node_topk->output(0)};
    named_outputs["Indices"] = OutputVector{node_topk->output(1)};

    return named_outputs;


    // auto zero_axis = context.mark_node(opset11::Constant::create(element::i32, Shape{1}, {0}));
    // auto dim_axis = context.mark_node(opset11::Constant::create(element::i64, Shape{1}, {dim}));
    // auto shape = context.mark_node(std::make_shared<opset11::ShapeOf>(input_tensor));
    // auto k_values_node = context.mark_node(std::make_shared<opset11::Gather>(shape, dim_axis, zero_axis));
    // auto k_values = context.mark_node(std::make_shared<opset11::Squeeze>(k_values_node));

    // return node.default_single_output_mapping({std::make_shared<default_opset::TopK>(data,
    //                                                               k_values,
    //                                                               dim,
    //                                                               mode,
    //                                                               ov::op::TopKSortType::SORT_VALUES,
    //                                                               element::i64,
    //                                                               stable)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
