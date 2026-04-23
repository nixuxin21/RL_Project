import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MSAirCompEnv(gym.Env):
    """
    智能反射面 (IRS) 辅助的多时隙空中计算 (MS-AirComp) 物理环境。
    核心修复：下调 alpha_th 目标振幅，打破 P_max = 1.0W 的功率死锁
    """
    def __init__(self, num_nodes=50, num_slots=10, num_irs_elements=64, num_codebook_states=16):
        super().__init__()
        
        # ==========================================
        # 1. 物理系统与通信参数设定
        # ==========================================
        self.K = num_nodes             # 传感器节点总数 (默认 50 个)
        self.N = num_slots             # 总时间槽数 (规定要在 10 个时隙内完成调度)
        self.M = num_irs_elements      # IRS 反射面单元数量 (64 面小镜子)
        self.C = num_codebook_states   # IRS 预设的反射角度种类 (16 种波束指向)
        self.noise_var = 1e-9          # 空间背景噪声方差 (极小的值，代表低噪声环境)
        self.P_max = 1.0               # 每个节点的最大硬件发射功率 (1 瓦特，硬性约束)
        
        # ==========================================
        # 2. 强化学习大纲设定 (状态空间与动作空间)
        # ==========================================
        # 状态空间 (Observation Space): 告诉 AI 它能看到什么
        # 包含 7 个维度的连续特征 (时隙进度, 剩余节点比例, 噪声对数, 大尺度衰落均值, g_th, alpha_th, 接收能量)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # 动作空间 (Action Space): 告诉 AI 它能做什么
        # 输出 3 个连续值，范围均被标准框架限制在 [-1.0, 1.0]，后续在 step 中会逆映射为物理真实值
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # ==========================================
        # 3. 初始化辅助工具与内部状态记录器
        # ==========================================
        self.codebook = self._generate_dft_codebook() # 预先生成 IRS 离散傅里叶变换 (DFT) 相位码本
        self.current_slot = 0                         # 当前是第几个时隙 (计时器初始化)
        self.transmitted_flags = np.zeros(self.K, dtype=bool) # 记录各节点是否已成功发送的布尔数组

    def _generate_dft_codebook(self):
        """
        生成 IRS 的离散傅里叶变换 (DFT) 码本。
        为了避免连续优化 64 个相位的巨大计算量，这里预先计算出 16 种典型的波束指向向量。
        """
        # 创建 16 行 x 64 列的复数矩阵
        codebook = np.zeros((self.C, self.M), dtype=np.complex128)
        for c in range(self.C):
            # 将 16 种状态均匀映射到 [-1, 1] 的空间角度 (正弦值) 上
            angle_sin = -1 + 2 * c / (self.C - 1)
            # 计算对应的 DFT 相位偏移向量
            phase_vector = np.exp(1j * np.pi * np.arange(self.M) * angle_sin)
            codebook[c] = phase_vector
        return codebook

    def reset(self, seed=None, options=None):
        """
        回合重置函数。每次训练开始，或 10 个时隙耗尽后调用，用于重新生成信道并清零状态。
        """
        super().reset(seed=seed)
        self.current_slot = 0                                 # 时间清零
        self.transmitted_flags = np.zeros(self.K, dtype=bool) # 所有节点重新处于“未发送”状态
        
        # 模拟真实的瑞利衰落信道 (Rayleigh Fading)，包含实部和虚部的复高斯分布
        # 1. 节点直达基站的信道 h_d (乘以 0.1 表示直达路径信号通常较弱)
        hd_rayleigh = (np.random.randn(self.K) + 1j * np.random.randn(self.K)) / np.sqrt(2)
        self.h_d = 0.1 * hd_rayleigh 
        
        # 2. 节点到 IRS 的信道 (h_r) 以及 IRS 到基站的信道 (h_bs_r)
        self.h_r = (np.random.randn(self.K, self.M) + 1j * np.random.randn(self.K, self.M)) / np.sqrt(2)
        self.h_bs_r = (np.random.randn(self.M) + 1j * np.random.randn(self.M)) / np.sqrt(2)
        
        # 提取全局的平均大尺度衰落，作为状态信息提供给 AI，帮助它感知当前环境的整体好坏
        self.avg_large_scale = np.mean(np.abs(self.h_d)**2) 
        
        # 返回初始观测状态和空的 info 字典
        return self._get_obs(0.0, 0.0, 1e-12), {}

    def step(self, action):
        """
        环境步进函数。接收 AI 的动作，进行物理推演，计算奖励并返回下一状态。
        """
        # ==========================================
        # 1. 动作解码 (将 [-1, 1] 映射到真实的物理区间)
        # ==========================================
        # 准入门限 g_th：综合信道增益大于该值的节点，本时隙才有资格尝试发送
        g_th = 0.001 + (action[0] + 1) * 0.05
        
        # 【核心修正】：AirComp 目标对齐振幅 alpha_th，映射到 [0.05, 0.15]
        # 即使 alpha_th 是 0.05，所需最高功率也仅为 0.05^2 / 0.01 = 0.25 W <= 1.0 W
        # 且 SNR 依然高达 0.0025 / 1e-9 = 2.5e6，完美满足计算精度要求，有效防止功率死锁。
        alpha_th = 0.05 + (action[1] + 1) * 0.05 
        
        # IRS 码本索引：将连续动作四舍五入离散化为 0 到 15 的整数
        c_idx = int(np.clip(np.round((action[2] + 1) * 0.5 * (self.C - 1)), 0, self.C - 1))
        irs_vector = self.codebook[c_idx] 
        
        # ==========================================
        # 2. 物理信道计算
        # ==========================================
        # 计算 IRS 级联信道 (节点 -> IRS -> 基站)，并乘以 0.05 的路径损耗系数
        cascade_channel = np.sum(self.h_r * self.h_bs_r * irs_vector, axis=1) 
        cascade_channel = (cascade_channel / np.sqrt(self.M)) * 0.05
        
        # 计算综合信道：总信道 = 直达信道 + 级联信道
        h_total = self.h_d + cascade_channel
        h_gain = np.abs(h_total)**2 # 求模的平方得到信道功率增益
        
        # ==========================================
        # 3. 节点调度与功率约束核查
        # ==========================================
        # 初筛：信道质量达标 (>= g_th) 且 之前未发送过数据的节点
        tx_mask = (h_gain >= g_th) & (~self.transmitted_flags)
        
        # 计算通过初筛的节点为了对齐目标振幅 alpha_th，所需付出的发射功率
        required_power = (alpha_th**2) / (h_gain + 1e-12) 
        
        # 约束：所需功率绝不能超过硬件上限 P_max (1.0 W)
        valid_power_mask = required_power <= self.P_max
        
        # 最终确定的激活名单：同时满足信道达标、未发送过、且功率不超标的节点
        final_tx_mask = tx_mask & valid_power_mask
        
        # 更新状态：将本时隙成功发送的节点标记为 True
        self.transmitted_flags |= final_tx_mask
        num_tx_this_slot = np.sum(final_tx_mask) # 统计本时隙成功调度的节点数量
        
        # 计算基站端接收到的有效信号能量 (所有激活节点振幅叠加的平方 + 噪声)
        rx_energy = (num_tx_this_slot * alpha_th)**2 + self.noise_var
        
        # ==========================================
        # 4. 推进时间与计算 Reward (奖励工程)
        # ==========================================
        self.current_slot += 1
        done = self.current_slot >= self.N # 达到最大时隙 N 时回合结束
        
        reward = 0.0
        snr_val = (alpha_th**2) / self.noise_var # 计算理论信噪比
        
        # 正向激励：信噪比合格的前提下，每多调度一个节点奖励 2 分；否则倒扣分
        if snr_val > 100: 
            reward += num_tx_this_slot * 2.0 
        else:
            reward -= num_tx_this_slot * 2.0 
            
        # 功耗惩罚：引导 AI 学会使用 IRS 改善信道，从而降低节点的平均发射功率
        if num_tx_this_slot > 0:
            reward -= 0.5 * np.mean(required_power[final_tx_mask])
            
        # 终局清算 (Sparse Reward)
        if done:
            # 计算到最后依然没有发送成功的“遗漏节点”数量
            missed_nodes = self.K - np.sum(self.transmitted_flags)
            # 计算理论上的均方误差 (MSE) 惩罚项
            mse_noise_penalty = (self.noise_var / (alpha_th**2)) * 1e6
            # 给出重拳惩罚：遗漏节点的平方级扣分，强迫 AI 追求 100% 覆盖率
            reward -= (missed_nodes**2 * 0.5 + mse_noise_penalty)

        # 获取下一步的观测状态
        obs = self._get_obs(g_th, alpha_th, rx_energy)
        
        # 记录详细日志，供 evaluate_agent.py 提取和打印
        info = {
            "tx_this_slot": num_tx_this_slot, 
            "total_tx": np.sum(self.transmitted_flags),
            "snr_val": snr_val,
            "power_avg": np.mean(required_power[final_tx_mask]) if num_tx_this_slot > 0 else 0.0
        }
        
        return obs, reward, done, False, info

    def _get_obs(self, g, a, rxe):
        """
        状态提取辅助函数。将当前的物理变量打包成 AI 神经网络可读的 7 维向量。
        """
        # 对方差和能量等跨度极大的物理量取对数 (Log10)，稳定神经网络的梯度更新
        log_noise = np.log10(self.noise_var)
        log_rxe = np.log10(rxe + 1e-12) 
        
        return np.array([
            self.current_slot / self.N,                    # 1. 归一化后的时隙进度条 [0, 1]
            1.0 - np.sum(self.transmitted_flags) / self.K, # 2. 尚未发送数据的节点比例 [0, 1]
            log_noise,                                     # 3. 背景噪声对数值
            self.avg_large_scale,                          # 4. 当前全局平均大尺度衰落
            g,                                             # 5. 上一步动作设定的门限 g_th
            a,                                             # 6. 上一步动作设定的振幅 alpha_th
            log_rxe                                        # 7. 上一步基站接收到的能量对数值
        ], dtype=np.float32)