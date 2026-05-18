"""
完整 SAC 训练入口。

本脚本训练的是最原始、动作空间最大的策略：SAC 同时输出
`[g_th, alpha_th, irs_codebook]` 三个动作维度。它适合验证“RL 是否能自动联合优化
传输门限、AirComp 对齐振幅和 IRS 码本选择”这个问题。

从后续实验结果看，这个完整 SAC baseline 不是当前最强策略，但它仍然是论文对比中
非常重要的学习型基线。训练时必须同时保存模型权重和 VecNormalize 统计文件，
否则评估阶段的输入分布会和训练阶段不一致。
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np
# 导入强化学习训练库中的 SAC 算法实现。
from stable_baselines3 import SAC
# 导入环境多开与包装工具
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
# 导入训练过程中的回调函数（相当于监控器和定时保存器）
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# 导入你在 test_env.py 中自己写的物理环境
from test_env import MSAirCompEnv

# =====================================================================
# 自定义回调类：将物理指标（成功调度的节点数）桥接到 AI 训练面板上
# =====================================================================
class TrackSuccessCallback(BaseCallback):
    """
    默认情况下，TensorBoard 只会画 Reward（奖励）曲线。
    这个类的作用像是一个“战地记者”，专门去环境里把“到底成功发送了多少个节点”这个物理指标挖出来并画图。
    """
    def __init__(self, verbose=0):
        """保留 SB3 回调的标准初始化接口，`verbose` 控制日志冗余程度。"""
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        每个环境 step 后由 SB3 自动调用。

        多进程环境里一次会返回多个 `info`，只有某个子环境 episode 结束时，
        `info["total_tx"]` 才代表该 episode 的最终成功节点数。
        """
        # 本地变量表包含当前训练步的全部上下文，`infos` 是环境返回的额外物理指标。
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            # 找到当前环境的索引，检查这个环境当前是否正好结束。
            # 并且确保环境返回信息里有“总成功发送节点数”这个自定义物理指标。
            if "total_tx" in info and env_idx < len(dones) and dones[env_idx]:
                # 将这个数值记录到训练监控面板的自定义指标目录下，便于观察最终调度效果。
                self.logger.record("custom_metrics/final_total_tx_nodes", info["total_tx"])
        return True # 返回 True 表示允许训练继续进行

def main():
    """
    构建并训练完整 SAC 策略。

    主要流程：
    1. 创建 8 个并行环境，提高样本收集速度。
    2. 使用 4 帧观测堆叠，让策略看到短期历史。
    3. 使用 VecNormalize 稳定物理量尺度。
    4. 训练 SAC 并定期 checkpoint。
    5. 保存最终模型和归一化统计文件。
    """
    print(">>> 启动训练引擎...")
    
    # =====================================================================
    # 1. 环境的“平行宇宙”多开与“记忆”包装 (工程化核心)
    # =====================================================================
    num_cpu = 8 # 开启 8 个并行的环境进程，相当于 AI 有 8 个分身同时去收集数据
    
    # 使用 SubprocVecEnv 将单机环境变成 8 核多进程环境
    env = make_vec_env(MSAirCompEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # 赋予 AI 记忆：把过去 4 个时隙的观测数据像录像带一样叠在一起 (n_stack=4)
    # 这样 AI 就能看出信道衰落和调度进度的“变化趋势”，处理时序依赖问题
    env = VecFrameStack(env, n_stack=4)
    
    # 数据归一化保护伞：由于你的噪声极小 (1e-9)，这行代码会自动统计观测值和奖励的均值/方差
    # 把它们平滑地缩放到 [-1, 1] 附近，并限制最大偏差不超过 10.0
    # 这是防止神经网络计算时“梯度爆炸”导致 AI 变傻的最关键一步
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # =====================================================================
    # 2. 文件夹系统初始化
    # =====================================================================
    log_dir = "./rl_logs/"     # 存放 TensorBoard 曲线数据的文件夹
    model_dir = "./rl_models/" # 存放神经网络权重和归一化文件的文件夹
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # =====================================================================
    # 3. 实例化 SAC 大脑 (定义各种超参数)
    # =====================================================================
    model = SAC(
        policy="MlpPolicy",       # 使用多层感知机 (全连接神经网络) 作为 AI 的大脑
        env=env,                  # 绑定刚才用 8 核、堆叠、归一化包装好的超级环境
        learning_rate=3e-4,       # 学习率：AI 每次更新大脑参数的幅度，3e-4 是个极其经典的稳妥值
        batch_size=256,           # 每次从记忆池中随机抽出 256 条经验来进行反思和学习
        gamma=0.99,               # 折扣因子：非常看重未来。为了最后不吃惩罚，愿意在当下做出牺牲
        ent_coef="auto",          # 自动调节温度 (熵)：初期大胆瞎猜乱试，后期谨慎执行最优解 (SAC的灵魂)
        tensorboard_log=log_dir,  # 告诉 AI 把训练日记写到哪里
        verbose=1,                # 在控制台打印进度的详细程度
        seed=42                   # 设定随机种子，保证以后每次跑代码结果都一模一样，方便论文复现
    )

    # =====================================================================
    # 4. 配置回调函数 (自动存档系统)
    # =====================================================================
    # 自动存档回调负责定期保存模型，作用类似训练过程中的检查点。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1), # 每走这么多步就存一次档（注意多核环境下步数是被平摊的）
        save_path=model_dir,
        name_prefix="ms_aircomp_v3_sac"     # 存下来的文件前缀名
    )
    
    # 把“自动存档”和之前的“战地记者”组合成一个列表，一起交给 AI
    track_callback = TrackSuccessCallback()
    callback_list = CallbackList([checkpoint_callback, track_callback])

    print(f">>> 开始训练！{num_cpu} 核火力全开...")
    
    # =====================================================================
    # 5. 启动漫长的训练过程
    # =====================================================================
    total_timesteps = 200000 # AI 要在环境里走 20 万步。如果有 8 个环境，每个环境实际走 2.5 万步
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True # 在终端显示酷炫的进度条
    )

    # =====================================================================
    # 6. 训练结束，保存最终资产 (极其重要)
    # =====================================================================
    # 保存训练好的大脑 (模型权重)
    model.save(os.path.join(model_dir, "sac_final_model_v3"))
    
    # 保存环境的归一化“字典” (观测值的均值和方差)
    # 以后测试时如果不带上这本字典，AI 就看不懂未经缩放的真实物理数据了
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    print(">>> 训练圆满完成！")

# 保护块：在 Windows 系统里进行多进程环境采样时，
# 必须把主程序放在这里面，否则会无限循环打开新终端导致内存崩溃
if __name__ == "__main__":
    main()
