"""封装聚合反馈打分和 confirmed-index 选择，避免策略直接读取节点级 current CSI。"""

import numpy as np

__all__ = ["confirmation_feedback", "confirmed_index_from_feedback"]


def confirmation_feedback(candidate, args, feedback_noise_std, feedback_power_weight, rng):
    """处理确认、聚合反馈相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    observed_tx_fraction = float(candidate["tx_this_slot"]) / float(max(args.num_nodes, 1))
    if float(feedback_noise_std) > 0.0:
        observed_tx_fraction += float(rng.normal(0.0, float(feedback_noise_std)))
    observed_tx_fraction = float(np.clip(observed_tx_fraction, 0.0, 1.0))

    observed_power = float(candidate["power_avg"])
    observed_score = observed_tx_fraction - float(feedback_power_weight) * observed_power
    observed_score += float(rng.uniform(0.0, 1e-9))
    return {
        "irs_index": int(candidate["irs_index"]),
        "observed_tx_fraction": observed_tx_fraction,
        "observed_power": observed_power,
        "observed_score": float(observed_score),
    }


def confirmed_index_from_feedback(selected_indices, feedbacks):
    """处理confirmed、索引、from、聚合反馈相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    feedback_by_index = {int(feedback["irs_index"]): feedback for feedback in feedbacks}
    return int(
        max(
            selected_indices,
            key=lambda index: (
                float(feedback_by_index[int(index)]["observed_score"]),
                float(feedback_by_index[int(index)]["observed_tx_fraction"]),
                -float(feedback_by_index[int(index)]["observed_power"]),
            ),
        )
    )
