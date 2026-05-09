"""
单回合完整 SAC 策略评估脚本。

这个脚本用于“肉眼检查”一个 episode 内 SAC 的逐时隙行为：
每个时隙打印 SAC 输出的物理参数、IRS 索引、本槽调度节点数、累计成功节点数、
平均功率和理论 MSE。它不适合做论文统计表，但非常适合调试策略是否出现明显异常，
例如 `g_th/alpha_th` 偏高、IRS 索引固定、前几个时隙调度不够激进等。
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np
# 导入 SB3 的 SAC 算法模型
from stable_baselines3 import SAC
# 导入环境包装器（测试时也需要原样包装，否则 AI 会看不懂数据）
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# 导入你现有的自定义物理环境
from test_env import MSAirCompEnv

def evaluate():
    """
    加载完整 SAC 模型并运行一个确定性评估回合。

    关键工程要求：
    - 评估环境必须复用训练时的 `VecFrameStack(n_stack=4)`；
    - 必须加载训练时保存的 `VecNormalize` 统计量；
    - 评估时必须冻结 `venv.training=False`，避免测试样本污染归一化统计。
    """
    # =====================================================================
    # 1. 路径配置：告诉代码去哪里找训练好的“大脑”和“记忆字典”
    # =====================================================================
    model_dir = "./rl_models/"
    model_path = os.path.join(model_dir, "sac_final_model_v3.zip") # 神经网络权重
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")      # 环境归一化统计字典

    print("="*55)
    print("🚀 开始 MS-AirComp 与 IRS 联合调度评估 (基于真实源码)")
    print("="*55)

    # =====================================================================
    # 2. 一比一复原测试环境 (极其关键的工程步骤)
    # =====================================================================
    # 训练时用了 SubprocVecEnv 开了 8 个分身，测试时只需要 1 个，所以用 DummyVecEnv 包装
    venv = DummyVecEnv([lambda: MSAirCompEnv(num_nodes=50, num_slots=10, num_irs_elements=64, num_codebook_states=16)])

    # 【关键修复】：赋予 AI 短期记忆。
    # 训练时你用了 n_stack=4，这里必须严格保持一致！否则 AI 看到的画面维度不对，会直接报错崩溃。
    venv = VecFrameStack(venv, n_stack=4)

    # 加载 VecNormalize 状态 (即物理数据的缩放比例字典)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"❌ 未找到归一化文件: {stats_path}")
        
    venv = VecNormalize.load(stats_path, venv)
    
    # 【极其致命的工程陷阱】：冻结环境统计量
    # 必须设置为 False！否则测试时的单次信道波动会被加进历史平均值里，
    # 导致 AI 眼里的判定标准发生偏移 (分布偏移)，AI 的表现会越来越差。
    venv.training = False     
    # 测试时不归一化奖励，因为测试时我们不靠奖励更新网络，只需观察真实的物理表现
    venv.norm_reward = False  

    # =====================================================================
    # 3. 加载 AI 大脑
    # =====================================================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 未找到模型文件: {model_path}")
        
    model = SAC.load(model_path, env=venv)

    # 初始化评估环境。`venv.reset()` 返回的 obs 已经经过 VecFrameStack 和 VecNormalize 处理。
    obs = venv.reset()
    mse_list = []
    
    # 提取环境常量用于理论计算
    noise_var = 1e-9
    N_SLOTS = 10
    K_TOTAL = 50

    # =====================================================================
    # 4. 开启 10 个时隙的单次回合演习
    # =====================================================================
    for slot in range(N_SLOTS):
        
        # 做出决策：deterministic=True 是让 AI 拿出真实实力 (关闭高斯探索噪声)
        # 考试的时候不需要乱猜，只输出当前它认为的最优解
        action, _states = model.predict(obs, deterministic=True)
        a_raw = action[0] 

        # =====================================================================
        # 5. 逆映射动作空间 (窥探 AI 的内心想法)
        # 这一步严格对齐 test_env.py 里的公式，为了在终端打印出人类能看懂的物理参数
        # =====================================================================
        g_th = 0.001 + (a_raw[0] + 1) * 0.05                             # 准入门限
        alpha_th = 0.05 + (a_raw[1] + 1) * 0.05                          # 目标对齐振幅
        irs_idx = int(np.clip(np.round((a_raw[2] + 1) * 0.5 * 15), 0, 15)) # 挑选的 IRS 波束方向

        # 环境真实步进，执行物理模拟。向量化环境返回的是批量结果，
        # 即使这里只有 1 个环境，也需要从 `info_list[0]` 取出实际字典。
        obs, reward, done, info_list = venv.step(action)
        info = info_list[0] # 从包裹的向量化环境中提取字典

        # 提取当前时隙的战报数据
        tx_this_slot = info["tx_this_slot"]
        power_avg = info["power_avg"]
        total_tx = info["total_tx"]
        
        # 计算理论 MSE：AirComp 的误差完全由 (噪声方差 / 目标振幅的平方) 决定
        slot_mse = noise_var / (alpha_th ** 2)
        mse_list.append(slot_mse)

        # =====================================================================
        # 6. 终端美化打印 (生成评估报告)
        # =====================================================================
        print(f"📍 时隙 [{slot+1:02d}/{N_SLOTS:02d}]")
        print(f"  ├─ 动作决策 : g_th = {g_th:.5f}, α_th = {alpha_th:.4f}, IRS 索引 = {irs_idx}")
        print(f"  ├─ 调度结果 : 本槽激活 = {tx_this_slot:02d} 个节点, 累计成功 = {total_tx:02d} / {K_TOTAL}")
        print(f"  ├─ 功耗误差 : 平均功率 = {power_avg:.4e} W, MSE = {slot_mse:.4e}")
        print(f"  └─ 步骤奖励 : {reward[0]:.4f}\n")

        # 如果回合提前结束 (或者达到 10 个时隙)，打印最终总结
        if done[0]:
            print("="*55)
            print("📊 评估回合结束总结")
            print("="*55)
            print(f"✅ 最终成功发送节点数 : {total_tx} / {K_TOTAL}")
            print(f"📉 平均理论 MSE 误差  : {np.mean(mse_list):.4e}")
            break

# 启动脚本
if __name__ == "__main__":
    evaluate()
