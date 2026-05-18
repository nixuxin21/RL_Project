"""
完整 SAC 的蒙特卡洛批量评估脚本。

相比 `evaluate_agent.py` 的单回合逐时隙打印，本脚本重复运行多个 episode，
统计成功节点数、完美覆盖率和理论 MSE，并输出图像：

1. 每个 episode 的成功节点数轨迹。
2. 成功节点数 CDF。

它主要用于早期评估完整 SAC 的稳定性。后续更完整的多策略对比请使用
`evaluate_policy_comparison.py`。
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import matplotlib.pyplot as plt

# 导入你现有的自定义物理环境
from test_env import MSAirCompEnv

def run_batch_evaluation():
    """
    加载完整 SAC 模型并执行固定数量的 Monte Carlo episode。

    这个函数刻意保持与训练时相同的环境包装顺序：
    `MSAirCompEnv -> DummyVecEnv -> VecFrameStack -> VecNormalize`。
    如果包装顺序或归一化统计不一致，模型输入维度或数值分布都会出错。
    """
    # =====================================================================
    # 1. 路径与全局参数配置
    # =====================================================================
    model_dir = "./rl_models/"
    model_path = os.path.join(model_dir, "sac_final_model_v3.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    # 环境与统计参数
    K_TOTAL = 50        # 系统总节点数
    N_SLOTS = 10        # 每回合的总时隙数
    NUM_EPISODES = 1000  # 蒙特卡洛测试回合数 (日常测试用100，发论文出图建议改到1000)

    print("="*60)
    print(f"🚀 开始 MS-AirComp 蒙特卡洛批量评估 ({NUM_EPISODES} 回合)")
    print("="*60)

    # =====================================================================
    # 2. 初始化环境 (严格保持与训练时的数据管道一致)
    # =====================================================================
    venv = DummyVecEnv([lambda: MSAirCompEnv(num_nodes=K_TOTAL, num_slots=N_SLOTS, num_irs_elements=64, num_codebook_states=16)])
    venv = VecFrameStack(venv, n_stack=4) # 恢复 AI 的 4 帧短期记忆
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"❌ 未找到归一化文件: {stats_path}")
    venv = VecNormalize.load(stats_path, venv)
    
    # 【致命约束】：冻结动态均值计算，只使用训练时固化的字典，防止测试数据污染判定标准
    venv.training = False     
    venv.norm_reward = False  

    # =====================================================================
    # 3. 加载训练好的 SAC 大脑
    # =====================================================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 未找到模型文件: {model_path}")
    model = SAC.load(model_path, env=venv)

    # =====================================================================
    # 💡 核心修复：打破 SB3 的种子死锁，强行重置 NumPy 全局随机性
    # 这样每次底层 test_env.py 调用 np.random 重新生成信道时，都会得到真随机的结果
    # 如果想在写论文时固定某一张图的结果，可以填入固定数字，例如 np.random.seed(2026)
    # =====================================================================
    np.random.seed()

    # =====================================================================
    # 4. 初始化统计容器
    # =====================================================================
    success_nodes_history = [] # 记录这 100 次里，每次最后成功了几个节点
    avg_mse_history = []       # 记录这 100 次里，每次的平均计算误差
    noise_var = 1e-9           # 背景噪声 (用于计算理论误差)

    # =====================================================================
    # 5. 开启蒙特卡洛大循环
    # =====================================================================
    for ep in range(NUM_EPISODES):
        # 每次重置环境时，底层物理环境都会使用刚设置的随机种子，重新生成一组真实衰落信道。
        obs = venv.reset()
        mse_list = []
        total_tx = 0
        
        # 内部时隙循环 (单次通信调度过程)
        for slot in range(N_SLOTS):
            # 获取确定性动作 (关闭随机探索，展现真实性能)
            action, _states = model.predict(obs, deterministic=True)
            a_raw = action[0] 

            # 逆映射获取 alpha_th 用于在此处统计理论 MSE
            alpha_th = 0.05 + (a_raw[1] + 1) * 0.05

            # 环境真实物理推进
            obs, reward, done, info_list = venv.step(action)
            info = info_list[0] 

            # 收集这一个时隙的统计数据
            total_tx = info["total_tx"]
            slot_mse = noise_var / (alpha_th ** 2)
            mse_list.append(slot_mse)

            # 如果提前调度完毕或时间耗尽，则结束本回合
            if done[0]:
                break
                
        # 记录整个回合的最终成绩
        success_nodes_history.append(total_tx)
        avg_mse_history.append(np.mean(mse_list))
        
        # 每跑 10 个回合，在终端打印一次近期战报
        if (ep + 1) % 10 == 0:
            print(f"🔄 进度: [{ep+1:03d}/{NUM_EPISODES:03d}] | 平均成功节点: {np.mean(success_nodes_history[-10:]):.1f}/{K_TOTAL} | 近期 MSE: {np.mean(avg_mse_history[-10:]):.4e}")

    # =====================================================================
    # 6. 数据统计与总结分析
    # =====================================================================
    mean_success = np.mean(success_nodes_history) # 均值
    std_success = np.std(success_nodes_history)   # 标准差 (越小说明算法越稳)
    
    print("="*60)
    print("📊 批量评估总结")
    print("="*60)
    print(f"🎯 平均成功发送节点 : {mean_success:.2f} ± {std_success:.2f} / {K_TOTAL}")
    print(f"📉 全局平均理论 MSE : {np.mean(avg_mse_history):.4e}")
    # 统计完美调度的概率
    print(f"🏆 完美调度占比 (100%覆盖) : {np.sum(np.array(success_nodes_history) == K_TOTAL) / NUM_EPISODES * 100:.1f}%")

    # =====================================================================
    # 7. 渲染学术图表
    # =====================================================================
    plot_results(success_nodes_history, K_TOTAL, NUM_EPISODES)

def plot_results(data, k_total, num_episodes):
    """
    接收蒙特卡洛测试数据，使用 Matplotlib 绘制时序图和 CDF 图

    Args:
        data: 每个 episode 的最终成功节点数。
        k_total: 节点总数，用于绘制满覆盖参考线。
        num_episodes: episode 数量，用于设置横轴范围。
    """
    data = np.array(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------
    # 图1: 每回合成功节点数 (轨迹图)
    # -------------------------------------
    ax1.plot(range(1, num_episodes + 1), data, marker='o', linestyle='-', color='#1f77b4', markersize=4)
    ax1.axhline(y=k_total, color='r', linestyle='--', label=f'Max Nodes ({k_total})') # 顶部基准红线
    ax1.set_ylim(0, k_total + 2)
    ax1.set_title('Success Nodes per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Number of Success Nodes')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # -------------------------------------
    # 图2: 成功节点数的累积分布函数 (CDF)
    # -------------------------------------
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax2.plot(sorted_data, yvals, marker='.', linestyle='-', color='#2ca02c')
    ax2.set_title('CDF of Success Nodes')
    ax2.set_xlabel('Number of Success Nodes')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_xlim(min(data) - 1, k_total + 1)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join("results", "policy_comparison", "batch_evaluation_results.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n📈 图表已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    run_batch_evaluation()
