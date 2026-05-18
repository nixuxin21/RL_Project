"""模块 `test_env.py`：封装本项目实验、分析或测试所需的代码逻辑。"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MSAirCompEnv(gym.Env):
    """
    智能反射面 (IRS) 辅助的多时隙空中计算 (MS-AirComp) 物理环境。
    核心修复：下调 alpha_th 目标振幅，打破 P_max = 1.0W 的功率死锁
    """
    def __init__(
        self,
        num_nodes=50,
        num_slots=10,
        num_irs_elements=64,
        num_codebook_states=16,
        irs_phase_mode="codebook",
        include_codebook_features=False,
        codebook_feature_g_th=0.001,
        codebook_feature_alpha_th=0.05,
        codebook_feature_noise_std=0.0,
    ):
        """
        初始化环境的物理参数、观测空间、动作空间和内部状态。

        Args:
            num_nodes: 传感器节点数量 K。
            num_slots: 每个 episode 的最大时隙数量 N。
            num_irs_elements: IRS 反射单元数量 M。
            num_codebook_states: IRS 离散码本大小 C。
            irs_phase_mode: IRS 模式；`codebook` 用动作选择码本，`random` 用随机相位，`none` 表示无 IRS。
            include_codebook_features: 是否在观测后追加 C 维码本质量特征。
            codebook_feature_g_th: 计算码本质量特征时使用的固定信道增益门限。
            codebook_feature_alpha_th: 计算码本质量特征时使用的固定 AirComp 目标振幅。
            codebook_feature_noise_std: 对 C 维码本质量特征加入的高斯噪声标准差；
                噪声作用在归一化后的 `[0, 1]` 特征尺度上，默认 0 表示精确特征。
        """
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if num_irs_elements <= 0:
            raise ValueError("num_irs_elements must be positive")
        if num_codebook_states <= 1:
            raise ValueError("num_codebook_states must be greater than 1")
        if irs_phase_mode not in {"codebook", "random", "none"}:
            raise ValueError("irs_phase_mode must be one of: 'codebook', 'random', 'none'")
        if codebook_feature_noise_std < 0:
            raise ValueError("codebook_feature_noise_std must be non-negative")
        
        # ==========================================
        # 1. 物理系统与通信参数设定
        # ==========================================
        self.K = num_nodes             # 传感器节点总数 (默认 50 个)
        self.N = num_slots             # 总时间槽数 (规定要在 10 个时隙内完成调度)
        self.M = num_irs_elements      # 智能反射面单元数量 (64 面小镜子)
        self.C = num_codebook_states   # 智能反射面预设的反射角度种类 (16 种波束指向)
        self.irs_phase_mode = irs_phase_mode # 智能反射面相位模式：码本优化、随机相位基线或禁用智能反射面。
        self.include_codebook_features = include_codebook_features
        self.codebook_feature_g_th = codebook_feature_g_th
        self.codebook_feature_alpha_th = codebook_feature_alpha_th
        self.codebook_feature_noise_std = codebook_feature_noise_std
        self.noise_var = 1e-9          # 空间背景噪声方差 (极小的值，代表低噪声环境)
        self.P_max = 1.0               # 每个节点的最大硬件发射功率 (1 瓦特，硬性约束)
        
        # ==========================================
        # 2. 强化学习大纲设定 (状态空间与动作空间)
        # ==========================================
        # 状态空间：默认包含 7 个连续物理特征。
        # 启用码本特征时，额外追加 C 个码本候选的预计可调度节点比例。
        obs_dim = 7 + (self.C if self.include_codebook_features else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 动作空间：告诉策略网络它可以控制哪些物理量。
        # 输出 3 个连续值，范围均被标准框架限制在 [-1.0, 1.0]，后续在环境步进中会逆映射为物理真实值。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # ==========================================
        # 3. 初始化辅助工具与内部状态记录器
        # ==========================================
        self.codebook = self._generate_dft_codebook() # 预先生成 IRS 离散傅里叶变换 (DFT) 相位码本
        self.current_slot = 0                         # 当前是第几个时隙 (计时器初始化)
        self.transmitted_flags = np.zeros(self.K, dtype=bool) # 记录各节点是否已成功发送的布尔数组

    def _select_irs_vector(self, irs_action):
        """
        根据环境模式选择 IRS 相位向量。
        codebook 模式用于 SAC 训练/评估，random 模式用于随机相位 baseline，none 模式用于无 IRS baseline。

        Args:
            irs_action: SAC 输出的归一化 IRS 动作，范围通常是 [-1, 1]。

        Returns:
            `(irs_vector, index)`：
            - `irs_vector` 是长度为 M 的复数相位向量；
            - `index` 是码本索引。随机 IRS 记为 -1，无 IRS 记为 -2，便于日志区分。
        """
        if self.irs_phase_mode == "random":
            # 随机 IRS baseline：每个 IRS 单元独立采样一个 [0, 2π) 相位。
            phases = self.np_random.uniform(0.0, 2.0 * np.pi, size=self.M)
            return np.exp(1j * phases), -1
        if self.irs_phase_mode == "none":
            # 无 IRS baseline：把级联链路直接置零，只保留直达链路 h_d。
            return np.zeros(self.M, dtype=np.complex128), -2

        # 码本模式：把连续动作线性映射到离散码本索引，再四舍五入。
        # 这样 SAC 仍可使用连续动作空间，但实际执行的是离散 IRS 波束。
        c_idx = int(np.clip(np.round((irs_action + 1) * 0.5 * (self.C - 1)), 0, self.C - 1))
        return self.codebook[c_idx], c_idx

    def _decode_action(self, action):
        """
        将归一化动作解码为物理参数。

        action[0] -> `g_th`：最小信道增益门限，门限越高，能发送的节点越少但信道更好。
        action[1] -> `alpha_th`：AirComp 对齐振幅，越大通常需要更高发射功率。

        Returns:
            `(g_th, alpha_th)` 两个真实物理值。
        """
        g_th = 0.001 + (action[0] + 1) * 0.05
        alpha_th = 0.05 + (action[1] + 1) * 0.05
        return g_th, alpha_th

    def _sanitize_action(self, action):
        """处理sanitize、action相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")
        clipped_action = np.clip(
            action_array,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        return clipped_action, bool(np.any(clipped_action != action_array))

    def _compute_slot_metrics(self, g_th, alpha_th, irs_vector):
        """
        计算当前信道和已发送状态下，一个 IRS 相位向量会带来的本时隙调度结果。
        这个函数不修改环境状态，因此可以用于 greedy baseline 预估候选码本。

        Args:
            g_th: 节点允许发送所需的最小等效信道增益。
            alpha_th: AirComp 目标接收振幅。
            irs_vector: 当前准备使用的 IRS 相位向量。

        Returns:
            一个只读指标字典，包括本时隙可发送节点 mask、发送数量、所需功率、
            平均功率、接收能量和剩余节点平均信道增益。
        """
        # 智能反射面级联链路：节点到 IRS 信道 h_r、IRS 到基站信道 h_bs_r、IRS 相位向量三者逐单元相乘后求和。
        # `/sqrt(M)` 用于控制 IRS 单元数变化时的尺度，`0.05` 是当前实验中的级联链路缩放因子。
        cascade_channel = np.sum(self.h_r * self.h_bs_r * irs_vector, axis=1)
        cascade_channel = (cascade_channel / np.sqrt(self.M)) * 0.05

        # 总信道由直达链路和 IRS 级联链路叠加而成；调度判断只使用功率增益 |h|^2。
        h_total = self.h_d + cascade_channel
        h_gain = np.abs(h_total)**2

        # 第一层筛选：信道增益超过门限，且节点此前还没有发送成功。
        tx_mask = (h_gain >= g_th) & (~self.transmitted_flags)

        # 空中计算对齐要求每个参与节点的接收振幅接近 alpha_th，
        # 因此需要的发射功率近似为 alpha_th^2 / |h|^2。
        required_power = (alpha_th**2) / (h_gain + 1e-12)
        valid_power_mask = required_power <= self.P_max

        # 最终发送集合必须同时满足：未发送、信道过门限、所需功率不超过硬件上限。
        final_tx_mask = tx_mask & valid_power_mask
        num_tx_this_slot = np.sum(final_tx_mask)

        return {
            "final_tx_mask": final_tx_mask,
            "tx_this_slot": num_tx_this_slot,
            "required_power": required_power,
            "power_avg": np.mean(required_power[final_tx_mask]) if num_tx_this_slot > 0 else 0.0,
            "rx_energy": (num_tx_this_slot * alpha_th)**2 + self.noise_var,
            "mean_gain_remaining": np.mean(h_gain[~self.transmitted_flags])
            if np.any(~self.transmitted_flags)
            else 0.0,
        }

    def preview_codebook_index(self, codebook_index, g_th, alpha_th):
        """
        预估某个 DFT 码本索引在当前时隙的效果，不推进环境。

        这个接口是所有 rule-based baseline 的关键：Greedy IRS 会逐个 preview 全部码本，
        Feature Argmax 的 codebook quality features 也由它生成。因为它不修改
        `transmitted_flags` 或 `current_slot`，所以可以安全地在同一时隙比较多个候选。
        """
        c_idx = int(np.clip(codebook_index, 0, self.C - 1))
        metrics = self._compute_slot_metrics(g_th, alpha_th, self.codebook[c_idx])
        return {
            "irs_index": c_idx,
            "tx_this_slot": metrics["tx_this_slot"],
            "power_avg": metrics["power_avg"],
            "mean_gain_remaining": metrics["mean_gain_remaining"],
        }

    def _get_codebook_features(self):
        """
        生成 C 维码本质量特征：每个码本在当前剩余节点集合上预计能调度的节点比例。
        使用固定的 codebook_feature_g_th/alpha_th，使特征与 IRS selector 训练目标一致。
        若配置了 codebook_feature_noise_std，则在归一化比例上加入高斯观测噪声并裁剪到 [0, 1]。
        """
        if not self.include_codebook_features:
            return np.empty(0, dtype=np.float32)

        # 每个 feature 是“如果本时隙选择该码本，预计可调度节点数量 / 总节点数”。
        # 这使 IRS selector 可以直接看到每个候选码本的即时覆盖潜力。
        counts = [
            self.preview_codebook_index(c_idx, self.codebook_feature_g_th, self.codebook_feature_alpha_th)[
                "tx_this_slot"
            ]
            for c_idx in range(self.C)
        ]
        features = (np.asarray(counts, dtype=np.float32) / self.K).astype(np.float32)
        if self.codebook_feature_noise_std > 0.0:
            noise = self.np_random.normal(
                loc=0.0,
                scale=self.codebook_feature_noise_std,
                size=self.C,
            ).astype(np.float32)
            features = np.clip(features + noise, 0.0, 1.0).astype(np.float32)
        return features

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

        Returns:
            Gymnasium 标准 `(obs, info)`；这里 info 暂时为空字典。
        """
        super().reset(seed=seed)
        self.current_slot = 0                                 # 时间清零
        self.transmitted_flags = np.zeros(self.K, dtype=bool) # 所有节点重新处于“未发送”状态
        
        # 模拟真实的瑞利衰落信道，包含实部和虚部的复高斯分布。
        # 1. 节点直达基站的信道 h_d (乘以 0.1 表示直达路径信号通常较弱)
        hd_rayleigh = (self.np_random.normal(size=self.K) + 1j * self.np_random.normal(size=self.K)) / np.sqrt(2)
        self.h_d = 0.1 * hd_rayleigh 
        
        # 2. 节点到 IRS 的信道 (h_r) 以及 IRS 到基站的信道 (h_bs_r)
        self.h_r = (
            self.np_random.normal(size=(self.K, self.M))
            + 1j * self.np_random.normal(size=(self.K, self.M))
        ) / np.sqrt(2)
        self.h_bs_r = (
            self.np_random.normal(size=self.M)
            + 1j * self.np_random.normal(size=self.M)
        ) / np.sqrt(2)
        
        # 提取全局的平均大尺度衰落，作为状态信息提供给 AI，帮助它感知当前环境的整体好坏
        self.avg_large_scale = np.mean(np.abs(self.h_d)**2) 
        
        # 返回初始观测状态和空的 info 字典
        return self._get_obs(0.0, 0.0, 1e-12), {}

    def step(self, action):
        """
        环境步进函数。接收 AI 的动作，进行物理推演，计算奖励并返回下一状态。

        Returns:
            Gymnasium 标准五元组 `(obs, reward, terminated, truncated, info)`。
            本环境没有额外截断逻辑，因此 `truncated` 固定为 False；
            `terminated` 在所有节点完成或时隙用尽时变为 True。
        """
        # ==========================================
        # 1. 动作解码 (将 [-1, 1] 映射到真实的物理区间)
        # ==========================================
        action, action_clipped = self._sanitize_action(action)

        # 准入门限 g_th 与 AirComp 目标对齐振幅 alpha_th
        g_th, alpha_th = self._decode_action(action)
        
        # 智能反射面相位向量：默认选择 DFT 码本，基础对照可切换为独立随机相位。
        irs_vector, c_idx = self._select_irs_vector(action[2])
        
        # ==========================================
        # 2. 物理信道计算、节点调度与功率约束核查
        # ==========================================
        metrics = self._compute_slot_metrics(g_th, alpha_th, irs_vector)
        final_tx_mask = metrics["final_tx_mask"]
        
        # 更新状态：将本时隙成功发送的节点标记为 True
        self.transmitted_flags |= final_tx_mask
        num_tx_this_slot = metrics["tx_this_slot"] # 统计本时隙成功调度的节点数量
        rx_energy = metrics["rx_energy"]
        
        # ==========================================
        # 4. 推进时间与计算 Reward (奖励工程)
        # ==========================================
        self.current_slot += 1
        total_tx = int(np.sum(self.transmitted_flags))
        all_nodes_done = total_tx >= self.K
        time_limit_reached = self.current_slot >= self.N
        done = all_nodes_done or time_limit_reached
        
        reward = 0.0
        snr_val = (alpha_th**2) / self.noise_var # 计算理论信噪比
        
        # 正向激励：当前 alpha_th 范围下 SNR 始终充足，覆盖节点数是主优化目标。
        reward += num_tx_this_slot * 2.0
            
        # 功耗惩罚：引导 AI 学会使用 IRS 改善信道，从而降低节点的平均发射功率
        if num_tx_this_slot > 0:
            reward -= 0.5 * metrics["power_avg"]
            
        # 终局清算 (Sparse Reward)
        if done:
            # 计算到最后依然没有发送成功的“遗漏节点”数量
            missed_nodes = self.K - total_tx
            # 计算理论上的均方误差 (MSE) 惩罚项
            mse_noise_penalty = (self.noise_var / (alpha_th**2)) * 1e6
            # 给出重拳惩罚：遗漏节点的平方级扣分，强迫 AI 追求 100% 覆盖率
            reward -= (missed_nodes**2 * 0.5 + mse_noise_penalty)

        # 获取下一步的观测状态
        obs = self._get_obs(g_th, alpha_th, rx_energy)
        
        # 记录详细日志，供 evaluate_agent.py 提取和打印
        info = {
            "tx_this_slot": int(num_tx_this_slot),
            "total_tx": total_tx,
            "snr_val": snr_val,
            "power_avg": metrics["power_avg"],
            "irs_index": c_idx,
            "irs_phase_mode": self.irs_phase_mode,
            "slots_used": self.current_slot,
            "is_complete": all_nodes_done,
            "missed_nodes": self.K - total_tx,
            "termination_reason": "complete" if all_nodes_done else "time_limit" if time_limit_reached else "running",
            "action_clipped": action_clipped,
        }
        
        return obs, reward, done, False, info

    def _get_obs(self, g, a, rxe):
        """
        状态提取辅助函数。将当前的物理变量打包成 AI 神经网络可读的 7 维向量。

        Args:
            g: 上一步使用的 `g_th`，用于让策略知道最近一次门限选择。
            a: 上一步使用的 `alpha_th`，用于让策略知道最近一次 AirComp 振幅选择。
            rxe: 上一步的接收能量，用对数形式加入观测以稳定数值尺度。

        Returns:
            基础 7 维观测，或在其后拼接 C 维 codebook quality features。
        """
        # 对方差和能量等跨度极大的物理量取对数 (Log10)，稳定神经网络的梯度更新
        log_noise = np.log10(self.noise_var)
        log_rxe = np.log10(rxe + 1e-12) 
        
        base_obs = np.array([
            self.current_slot / self.N,                    # 1. 归一化后的时隙进度条 [0, 1]
            1.0 - np.sum(self.transmitted_flags) / self.K, # 2. 尚未发送数据的节点比例 [0, 1]
            log_noise,                                     # 3. 背景噪声对数值
            self.avg_large_scale,                          # 4. 当前全局平均大尺度衰落
            g,                                             # 5. 上一步动作设定的门限 g_th
            a,                                             # 6. 上一步动作设定的振幅 alpha_th
            log_rxe                                        # 7. 上一步基站接收到的能量对数值
        ], dtype=np.float32)

        if not self.include_codebook_features:
            return base_obs

        return np.concatenate([base_obs, self._get_codebook_features()]).astype(np.float32)
