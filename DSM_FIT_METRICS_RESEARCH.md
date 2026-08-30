# DSM 波形拟合评价与对齐问题调研

> 起因：在 DePhaseKit 的 DSM 拟合对比子系统里，发现缺少残差和互相关系数等定量评价指标。
> 同时遇到具体问题——"按 pP 对齐，但 P 波到时没有对上，还有一小段距离"。
> 本文档从问题出发，调研波形拟合的评价方法、对不上的根因，落到本项目的代码改动方案。

---

## 0. 问题原文

> 在 dsm 拟合对比子系统里，缺少了残差和互相关系数，残差是指观测值与预测值之间的差异。
> 在地震波形拟合中最直接的计算方法是：测量预测波形和观测波形在每个时刻的差异，将这些差异
> 平方以确保都是正值，然后将它们全部相加。这被称为最小二乘（Least-square）方法。
>
> 我想知道波形拟合的过程，怎么判断观测数据的地震波波形和正演合成的波形拟合效果好坏，
> 看拟合度吗，波形的形状和振幅来比较吗，我自己肉眼去比较波形形状和振幅吗？
>
> 如果我现在只是粗略的正演了一个比较接近的整数值[^1]，但是肉眼看理论和观测数据还有波形没有对上，

[^1]: 注：经核计算服务器上的 .inf 文件，本项目正演参数（震源机制/深度/震中）均来自 GCMT，是精确值而非粗略整数。但用户当时描述的"粗略"可能是主观感受或指早期测试阶段。下文分析已据此修正。
> 也就是按照一个头段进行对齐，相同震相相对到时相差有点多，比如我按 pP 对齐，但是 P 波到时
> 没有对上，还有一小段距离。怎么办？

本文按以下顺序展开：

1. 波形拟合的过程与评价方法（怎么判断好坏）
2. "P 没对上"的所有可能原因与排查顺序
3. 现有代码现状（缺什么）
4. 代码层面怎么改（互相关自动对齐 + 指标计算 + 显示）

---

## 1. 波形拟合的过程与评价方法

### 1.1 波形拟合在做什么

地震波形拟合 = 用一个震源模型（位置、深度、机制解、震级）+ 速度模型，正演合成理论地震图 `s(t)`，
与实际观测 `d(t)` 比较，通过调参数让 `s` 尽量逼近 `d`。

判断"拟合好坏"有两类标准，**缺一不可**：

- **定量指标**：互相关系数、方差缩减、残差、振幅标定因子、时间偏移——给客观的数字
- **定性（肉眼）判断**：到时、形状/极性、振幅的逐震相对比——指标是标量，看不出哪个震相差

### 1.2 定量指标（本项目缺的部分）

直接用最小二乘残差 $\sum(d_i - s_i)^2$ 是**最基础但容易误导**的做法。原因：

> 即使震源参数（机制/深度/震级）精确（本项目均来自 GCMT，见第 2.0 节），合成与观测在
> **未对齐、未标定**的状态下直接相减，振幅量级差异和整体时移仍会把"形状其实拟合得不错"
> 完全掩盖掉。残差只有在对齐+标定之后算才有意义。

所以地震学界的标准做法是**组合用一组指标**，且按"对齐→标定→算指标"的顺序：

| 指标 | 含义 | 公式（观测 d，合成 s，长度 N） | 用途 / 阈值经验 |
|------|------|------|------|
| **互相关系数 CC** (cross-correlation coefficient) | 形状相似度，与振幅标定无关。在时间窗内平移取最大 | $\displaystyle CC = \max_\tau \frac{\sum(d_i-\bar d)(s_{i-\tau}-\bar s)}{\sqrt{\sum(d_i-\bar d)^2 \sum(s_i-\bar s)^2}}$ | 判断波形形状是否相似。CC>0.8 通常认为拟合较好，<0.5 较差 |
| **方差缩减 VR** (variance reduction) | 合成解释了多少观测能量，归一化残差 | $VR = 1 - \dfrac{\sum(d_i - s_i)^2}{\sum(d_i - \bar d)^2}$ | 综合评价（含振幅）。VR>0.7 较好，<0 表示合成还不如均值 |
| **振幅标定因子 A** | 合成相对观测的振幅缩放（最小二乘解） | $A = \dfrac{\sum d_i s_i}{\sum s_i^2}$ | 判断震级/辐射花样是否合理。A≈1 说明振幅本身匹配 |
| **时间偏移 τ** | 互相关最大时的时移 | $\tau = \arg\max_\tau \text{corr}(d, s)$ | **诊断用**：合成相对观测整体早到/晚到多少秒 |
| **L2 残差** | 平方残差和（你最初提到的） | $\sum(d_i - s_i)^2$ | 受振幅影响大，单独用会误导，需在标定后算 |
| **L1 残差** | 绝对残差和 | $\sum\|d_i - s_i\|$ | 对尖峰不敏感，比 L2 稳健 |

**关键认识**：标准流程是

> 先在时间窗内做互相关找最佳时移 τ → 把合成按 τ 平移对齐 → 再算标定因子 A 把合成振幅缩放匹配观测 → 最后在"对齐+标定"后算 VR / CC。

这才是波形拟合里"用得上的"残差。直接对未对齐、未标定的 d 和 s 算 L2 残差，数值没意义。

### 1.3 定性判断（肉眼也是必要的）

定量指标只是辅助，地震波形拟合最终**必须肉眼对比**，因为：

- CC / VR 是整个窗的标量，看不出"哪个震相拟合好、哪个差"
- 深度震相 (pP / sP) 的相对到时对深度敏感，形状对机制敏感，这些只有看波形才能判断

肉眼对比的标准图：**观测（黑实线）和合成（红虚线）上下叠画**，零线对齐，按震相到时标竖线。
看三点：

1. **到时（相位）**：各震相 P / pP / sP 的相对到时对不对——主要约束**震源深度**和**速度模型**
2. **形状 / 极性**：正还是负、有没有明显的脉冲特征——约束**震源机制解**
3. **振幅**：相对各台站的振幅衰减趋势——约束**震级**和**辐射花样**

本项目现有的"观测黑线 + 理论红线叠绘 + 震相对齐竖线"图（[dsm_fit_compare_dialog.py:650-654](dsm_fit_compare_dialog.py#L650-L654)）
正是这个标准图的范式，定性判断部分已经具备，缺的是定量指标。

### 1.4 拟合流程总览

```
观测 d(t) ──┐
            ├─→ [1] 截分析窗（P 前几秒到 P 后几十秒）
合成 s(t) ──┘     [2] 互相关找最优时移 τ  ──→ τ（诊断：整体偏移）
                  [3] 平移 s 对齐
                  [4] 算振幅标定 A         ──→ A（诊断：震级/辐射花样）
                  [5] 标定后算 VR / CC / L2 ──→ 拟合度数字
                  [6] 肉眼叠绘对比         ──→ 哪个震相差
```

---

## 2. "P 没对上"的所有原因与排查顺序

> 现象：按 pP 对齐，但 P 波到时没对上，还有一小段距离。

这是波形拟合里**最常见、最关键的问题**。根因几乎必为以下之一，按发生概率和排查成本排序。

### 2.0 前提澄清：震源参数本身是精确的（来自 GCMT），地壳厚度根据观测反推

经核计算服务器上的 .inf 文件（`<DSMTI_ROOT>/configs/<event>.dsm.inf`），
本项目每个事件的正演参数**来自 GCMT，是精确值**：

- **震源机制**：哈佛 CMT 矩张量，精确到小数点后三位。例 `25 -3.481 1.02 2.461 -1.236 0.885 -0.374`
- **震源深度**：精确值，例 273.9 / 136.0 / 20.0 km（非整数舍入）
- **震中位置**：精确经纬度，例 `-59.7400 -29.2000`
- **速度模型**：PREM，11 层结构，密度/速度多项式系数精确到小数点后四位
- **台站位置**：精确经纬度

**地壳厚度**情况不同：是从观测波形的震相到时差**反推算出来**的，非粗略估计：
- 所有事件震源在南桑威奇群岛弧（洋弧环境），地壳偏薄是合理的
- 反推后四舍五入取整（如 12.3 km → 12 km，或 13.6 km → 14 km）
- 加上人工拾取震相到时的几秒不确定性，反推地壳厚度有 1-2 km 误差

从 .inf 分层边界看（半径 km）：
```
6291 -> 6358  (厚度 67.0 km)   ← 下地壳/岩石圈地幔顶部
6358 -> 6371  (厚度 13.0 km)   ← 地壳（Moho 到地表）
```

地壳厚度设为 **13 km**，对南桑威奇群岛弧（洋弧）是合理值，但：
- 四舍五入 + 拾取误差带来的 1-2 km 偏差仍会造成 **0.2-0.3 秒级** P 到时偏差
- 与对齐方式问题（2.1）叠加后，可能表现为"P 没对上"

**因此"P 没对上"的主要原因就是对齐方式问题（2.1）**，地壳厚度的 1-2 km 误差是次要因素。

### 2.1 原因 A：对齐方式本身的问题（最可能，且最容易修）

**根因**：手动按单一震相对齐是**有缺陷的范式**。

现有代码逻辑（[dsm_fit_compare_core.py:249-250](dsm_fit_compare_core.py#L249-L250)）：

```python
observed_t, observed_y = extract_window(observed_trace, observed_align_time_s, ...)
synthetic_t, synthetic_y = extract_window(synthetic_trace, synthetic_align_time_s, ...)
```

`observed_align_time_s` 和 `synthetic_align_time_s` 是同一震相（如 pP）的到时，分别从
观测/合成 SAC 头段读，或 TauPy 现算。**问题**：即使震源参数精确，pP−P 的相对到时仍会因
**真实地幔速度结构与 PREM 的偏差**而有几秒级误差。你强行把观测和合成的 pP 都拉到 x=0，
P 的相对到时就必然错位。这正是你看到的现象。

**正解**：用**互相关自动对齐**，而不是手动按单震相硬拉：

1. 截一个足够长的分析窗（P 前几秒到 P 后几十秒）
2. 在窗内对 d 和 s 做互相关：`corr = np.correlate(d, s, mode='full')`
3. 取 `τ = argmax(corr) - (len(s)-1)` 作为最优时移
4. 把合成整体平移 τ 秒，让整段波形"最像"观测
5. τ 本身就是一个有意义的拟合量（模型偏差）

互相关给的是**整段波形整体最优**的对齐，不会出现"按 pP 对齐 P 就歪"。

→ **这是本次代码改动的核心**，详见第 4 节。

### 2.2 原因 A'：地壳厚度根据观测反推（对洋弧合理），但四舍五入+拾取误差带来 1-2 km 偏差

**现象**：按 pP 对齐时，P 波相对理论到时偏晚。

**根因**：.inf 文件里地壳厚度设为整数 13 km（`6358 -> 6371`）。这是从观测波形震相到时差**反推算出来**的，对南桑威奇群岛弧（洋弧环境）是合理值。但：
- **四舍五入**：如算出 12.3 km → 取整 12 km，或 13.6 km → 取整 14 km，引入 ±0.5 km 偏差
- **人工拾取震相到时的几秒不确定性**：反推地壳厚度会有 1-2 km 误差
- 综合导致 P 到时有 0.2-0.3 秒级偏差

→ **解决**：互相关自动对齐后看 τ 方向——若 τ 系统性为负（合成晚到），说明理论整体偏早，可能是地壳厚度略小 + 对齐方式共同作用。此时可微调 .inf 里地壳厚度 ±1-2 km 重新正演，看 CC/VR 是否改善。但优先先做互相关自动对齐（2.1）——它能吸收这种小偏差。

### 2.3 原因 B：震源深度与真实结构有系统偏差（次要）

pP−P 的相对到时主要由**深度**控制。虽然 .inf 里的深度来自 GCMT（精确值），但 GCMT 解本身
基于特定速度模型反演，与真实地幔结构可能有几公里级深度偏差。这会导致 pP−P 间隔有几秒误差：
- 深度偏大 → pP−P 间隔偏大 → 按 pP 对齐时 P 偏到前面（早到侧）
- 深度偏小 → pP−P 间隔偏小 → 按 pP 对齐时 P 偏到后面

→ **解决**：互相关自动对齐后，若各台站 τ scatter 较大（不同台站时移不一致），往往提示
深度/震源位置需要微调。可尝试在 GCMT 深度附近 ±5km 范围内重新正演，看 CC/VR 是否改善。
但**优先先做互相关对齐**——τ 的分布形态会告诉你是否需要调深度。

### 2.4 原因 C：速度模型不对

P 的**绝对到时**受路径上速度模型控制。如果模型偏了，P 系统性早到 / 晚到：

- 模型速度偏快 → P 早到（τ 系统性为负）
- 模型速度偏慢 → P 晚到（τ 系统性为正）

项目里有两套 DSM 树：`data/dsm` 和 `data/dsm/24.4`（[dsm_fit_compare_dialog.py:95-98](dsm_fit_compare_dialog.py#L95-L98)），
对应不同速度模型。TauP 侧用 iasp91（[dsm_fit_compare_dialog.py:366](dsm_fit_compare_dialog.py#L366)）。

→ **解决**：换不同模型（24.4 / prem / iasp91）看 P 到时变化。注意 TauP 命令用 `--mod` 不是
`-mod`（见记忆 [dsm-bhz-postprocess-filter-taup](../.claude/projects/-home-winner/memory/dsm-bhz-postprocess-filter-taup.md)）。

### 2.5 原因 D：震中位置 / 发震时刻的系统偏差（次要）

绝对到时偏差来自定位误差。虽然震中经纬度和发震时刻来自 GCMT（精确值），但 GCMT 定位
本身有几公里级不确定性。表现为：所有台站 τ 同号、同量级（整体平移）。

→ **解决**：这是 GCMT 解的固有不确定性，通常无需修改。互相关自动对齐能吸收这种整体时移
（τ 就是它的量度）。若 τ 系统性较大（>5s），可交叉核对 USGS/NEIC 定位是否有显著差异。

### 2.6 原因 E：采样率 / 时间轴不一致

互相关要求观测和合成采样率一致。若不一致，需要对齐前先重采样（obspy `trace.resample` 或
`Trace.interpolate`）。本项目观测是 .sac、合成是 .bhz，采样率可能不同。

→ **解决**：在 `cross_correlate_align` 里检测 dt 不一致时先重采样到同一采样率（第 4 节代码已考虑）。

### 2.7 排查决策树

```
P 没对上
  │
  ├─ 先改用互相关自动对齐（原因 A，必做）
  │     └─ 看时间偏移 τ 的分布
  │
  ├─ τ 系统性为负（合成晚到，需右移）
  │     ├─ 地壳偏薄（原因 A'）→ 改 .inf 地壳厚度（大陆 30-50 km）
  │     └─ 整体速度模型偏快（原因 C）→ 换模型或调参数
  │
  ├─ τ 系统性为正（合成早到，需左移）
  │     ├─ 地壳偏厚（原因 A'）→ 减小地壳厚度
  │     └─ 整体速度模型偏慢（原因 C）→ 换模型
  │
  ├─ τ 在各台站 scatter 大（不一致）
  │     ├─ 深度问题（原因 B）→ 在 GCMT 深度 ±5km 微调重新正演
  │     └─ 震源位置问题（原因 D）→ 交叉核对 USGS/NEIC 定位
  │
  └─ 采样率/时间轴问题（原因 E）→ 检查 SAC vs BHZ 采样率
```

τ 的分布形态本身就是诊断信息，这是为什么要先把互相关对齐做出来的核心价值——
**它把"对不上"从一个模糊的肉眼感觉，变成一个可分析的数字**。

---

## 3. 现有代码现状

### 3.1 模块结构

| 文件 | 角色 | 关键函数 |
|------|------|---------|
| [dsm_fit_compare_core.py](dsm_fit_compare_core.py) | 数据准备：配对 / 对齐 / 滤波 / 归一 | `build_pairs(args)`、`WaveformPair`、`extract_window`、`normalize_pair` |
| [dsm_fit_compare_dialog.py](dsm_fit_compare_dialog.py) | UI：拟合窗（5 台站/页叠绘）+ 组总览窗（整组剖面） | `DSMFitCompareWindow._plot_page`、`DSMGroupOverviewWindow._draw` |
| [ppk.py](ppk.py) | 主程序集成入口（场景 A/B/manual） | `open_dsm_fit_compare()`（ppk.py:737-816） |

### 3.2 现有对齐逻辑（问题所在）

`build_pairs` 当前对齐方式（[dsm_fit_compare_core.py:221-250](dsm_fit_compare_core.py#L221-L250)）：

- 按 `align_phase`（默认 t7，可选 P/pP/sP 等）从 SAC 头段读理论到时
- 头段没有则 TauPy 现算（`align_source = header_then_taup`）
- 观测和合成各自用该震相到时做零点，截窗

→ 这就是"手动按单震相对齐"，会触发第 2.1 节的问题。

### 3.3 现有归一逻辑

`normalize_pair`（[dsm_fit_compare_core.py:153-177](dsm_fit_compare_core.py#L153-L177)）两种模式：

- `separate`：观测 / 理论各自除自己的最大振幅，都归一到 [-1,1]，只比形状，丢振幅
- `pair`：同除两者最大值，保留振幅相对关系

→ 现状是**只归一、不标定**。没有最小二乘振幅标定因子 A，也没有互相关。

### 3.4 现有绘图逻辑（定性部分已具备）

`_plot_page`（[dsm_fit_compare_dialog.py:646-668](dsm_fit_compare_dialog.py#L646-L668)）：
观测黑线 + 理论红线叠绘 + x=0 对齐竖线 + 台站名/震中距/方位角标注。这正是第 1.3 节的标准图范式，
**定性判断部分已具备，缺的是定量指标显示**。

### 3.5 全文检索结论

对 `residual / misfit / metric / correlation / cross_corr / fit_score / error_sum` 等关键词全文搜索，
**均无任何结果**。项目内不存在任何现成的拟合评价指标实现。需要新增。

---

## 4. 代码层面怎么改

最小改动方案：在现有架构上**叠加**互相关自动对齐 + 指标计算 + 显示，不破坏现有手动对齐流程
（作为可选模式，由参数面板 checkbox 控制）。

### 4.1 改动 1：`dsm_fit_compare_core.py` —— 新增互相关对齐 + 指标函数

在 `normalize_pair` 之后新增（不改现有逻辑）：

```python
def cross_correlate_align(
    observed_y: np.ndarray,
    synthetic_y: np.ndarray,
    observed_t: np.ndarray,
    synthetic_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """互相关自动对齐 + 振幅标定 + 计算指标。

    流程：互相关找最优时移 τ → 平移合成 → 截公共窗 → 振幅标定 A → 算 CC/VR。
    不改变观测侧，只返回对齐+标定后的合成。

    Returns:
        aligned_synth_y:  对齐+标定后的合成波形（公共窗内）
        aligned_synth_t:  对齐后的合成时间轴（公共窗内）
        time_shift_s:     最优时移 τ（秒，合成相对观测；正=合成需右移/晚到）
        amplitude_factor: 振幅标定因子 A
        cross_corr_max:   互相关系数 CC（形状相似度，-1..1）
        variance_reduction: 方差缩减 VR（标定后，<=1）
    """
    if observed_y.size == 0 or synthetic_y.size == 0:
        return synthetic_y, synthetic_t, 0.0, 1.0, 0.0, 0.0

    # 采样率不一致时，把合成重采样到观测的采样率（obspy 在 build_pairs 里已读成 trace，
    # 这里只处理已采样的数组，故用线性插值对齐时间网格）
    dt_obs = float(observed_t[1] - observed_t[0]) if len(observed_t) > 1 else 1.0
    dt_syn = float(synthetic_t[1] - synthetic_t[0]) if len(synthetic_t) > 1 else 1.0
    if abs(dt_obs - dt_syn) > 1e-6:
        # 合成重采样到观测时间网格
        s_resamp = np.interp(observed_t, synthetic_t, synthetic_y)
        s = s_resamp
        dt = dt_obs
    else:
        s = synthetic_y
        dt = dt_obs

    d = observed_y
    # 1. 互相关找最优时移（样本数）
    corr = np.correlate(d, s, mode="full")
    lag_samples = int(np.argmax(corr) - (len(s) - 1))
    time_shift_s = lag_samples * dt

    # 2. 平移合成（用 obspy/np 滚动；这里用 interp 在观测时间轴上重采样平移后的合成）
    shifted_synth_t = observed_t - time_shift_s  # 合成的"原始时间"对应到观测轴
    s_shifted = np.interp(observed_t, shifted_synth_t, synthetic_y if dt_obs == dt_syn else s_resamp)

    # 3. 振幅标定 A = sum(d*s)/sum(s*s)（最小二乘解）
    ss = s_shifted
    denom = float(np.sum(ss * ss))
    amplitude_factor = float(np.sum(d * ss) / denom) if denom > 0 else 1.0
    aligned_synth_y = ss * amplitude_factor
    aligned_synth_t = observed_t.copy()

    # 4. 互相关系数 CC（去均值后的归一化内积）
    d_norm = d - np.mean(d)
    s_norm = aligned_synth_y - np.mean(aligned_synth_y)
    cc_num = float(np.sum(d_norm * s_norm))
    cc_den = float(np.sqrt(np.sum(d_norm ** 2) * np.sum(s_norm ** 2)))
    cross_corr_max = cc_num / cc_den if cc_den > 0 else 0.0

    # 5. 方差缩减 VR = 1 - sum((d-s)^2) / sum((d-mean(d))^2)
    resid = d - aligned_synth_y
    vr_den = float(np.sum(d_norm ** 2))
    variance_reduction = 1.0 - float(np.sum(resid ** 2)) / vr_den if vr_den > 0 else 0.0

    return aligned_synth_y, aligned_synth_t, time_shift_s, amplitude_factor, cross_corr_max, variance_reduction
```

### 4.2 改动 2：`WaveformPair` 加指标字段

```python
@dataclass
class WaveformPair:
    station_key: str
    distance_deg: float
    azimuth_deg: float | None
    align_time_s: float
    observed_path: Path
    synthetic_path: Path
    observed_t: np.ndarray
    observed_y: np.ndarray
    synthetic_t: np.ndarray
    synthetic_y: np.ndarray
    # 新增：互相关对齐 + 指标（仅当 use_crosscorr_align=True 时填充）
    time_shift_s: float = 0.0          # 互相关时移 τ（秒）
    amplitude_factor: float = 1.0      # 振幅标定因子 A
    cross_corr_max: float = 0.0        # 互相关系数 CC（形状相似度）
    variance_reduction: float = 0.0    # 方差缩减 VR
```

### 4.3 改动 3：`build_pairs` 里按模式调用

在 `extract_window` 之后、`append` 之前（[dsm_fit_compare_core.py:249-270](dsm_fit_compare_core.py#L249-L270)）：

```python
observed_t, observed_y = extract_window(observed_trace, observed_align_time_s, args.time_min, args.time_max)
synthetic_t, synthetic_y = extract_window(synthetic_trace, synthetic_align_time_s, args.time_min, args.time_max)
if observed_t.size == 0 or synthetic_t.size == 0:
    skipped.append(f"{key}: empty plotting window")
    continue

# --- 新增：互相关自动对齐 vs 旧的手动归一 ---
use_xcorr = getattr(args, "use_crosscorr_align", False)
tau = 0.0
amp_factor = 1.0
cc = 0.0
vr = 0.0
if use_xcorr:
    (synthetic_y, synthetic_t, tau, amp_factor, cc, vr) = cross_correlate_align(
        observed_y, synthetic_y, observed_t, synthetic_t
    )
    # 已标定振幅，不再单独归一；但为保证 ylim 一致，把观测也按同标定缩放显示
    # （观测侧不动，显示时合成已带 amp_factor）
    observed_y_disp = observed_y  # 观测保持原样
else:
    observed_y, synthetic_y = normalize_pair(observed_y, synthetic_y, args.normalize)
    observed_y_disp = observed_y

azimuth_deg = get_sac_value(observed_sac, "az")
pairs.append(
    WaveformPair(
        station_key=key,
        distance_deg=float(distance_deg),
        azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else None,
        align_time_s=float(observed_align_time_s),
        observed_path=observed_path,
        synthetic_path=synthetic_path,
        observed_t=observed_t,
        observed_y=observed_y_disp,
        synthetic_t=synthetic_t,
        synthetic_y=synthetic_y,
        time_shift_s=tau,
        amplitude_factor=amp_factor,
        cross_corr_max=cc,
        variance_reduction=vr,
    )
)
```

> 注：互相关模式下振幅标定后，合成 y 的量纲和观测一致，`_plot_page` 里 `set_ylim(-1.3, 1.3)`
> 假设归一到 [-1,1] 的逻辑要相应调整（见 4.5）。

### 4.4 改动 4：参数面板加 checkbox

`_ParamPanel._build`（[dsm_fit_compare_dialog.py:226-232](dsm_fit_compare_dialog.py#L226-L232) 归一行附近）：

```python
self.xcorr_align_chk = QCheckBox("互相关自动对齐")
self.xcorr_align_chk.setToolTip(
    "用互相关自动找最优时移对齐，而非手动按单一震相对齐。\n"
    "避免'按 pP 对齐但 P 没对上'。输出 τ/CC/VR/A 指标。\n"
    "勾选后归一模式被忽略（改为最小二乘振幅标定）。"
)
form.addRow("对齐模式:", self.xcorr_align_chk)
```

`build_args`（[dsm_fit_compare_dialog.py:361-379](dsm_fit_compare_dialog.py#L361-L379)）加：

```python
use_crosscorr_align=self.xcorr_align_chk.isChecked(),
```

### 4.5 改动 5：拟合窗 `_plot_page` 显示指标

`_plot_page`（[dsm_fit_compare_dialog.py:646-668](dsm_fit_compare_dialog.py#L646-L668)），在 `ax.text` 标题之后加：

```python
# 互相关模式下显示指标
if getattr(args, "use_crosscorr_align", False) and hasattr(pair, "cross_corr_max"):
    metrics_txt = f"CC={pair.cross_corr_max:.2f}  VR={pair.variance_reduction:.2f}  τ={pair.time_shift_s:+.1f}s  A={pair.amplitude_factor:.2f}"
    ax.text(0.99, 0.95, metrics_txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#1E88A8",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
    # 互相关模式下合成已标定，ylim 改为按观测峰值自适应
    ymax = float(np.max(np.abs(pair.observed_y))) * 1.3
    ax.set_ylim(-ymax, ymax)
```

### 4.6 改动 6：组总览窗 `_draw` 显示指标

`_draw`（[dsm_fit_compare_dialog.py:854-896](dsm_fit_compare_dialog.py#L854-L896)）在每条波形旁标注 CC（可选）：

```python
# 在 ax.plot(synthetic...) 之后
if getattr(args, "use_crosscorr_align", False) and hasattr(pair, "cross_corr_max"):
    ax.text(tmax, offset, f" CC={pair.cross_corr_max:.2f}",
            fontsize=7, color="#1E88A8", va="center", ha="left")
```

并在标题区显示整组平均指标：

```python
if getattr(args, "use_crosscorr_align", False) and self.pairs:
    mean_cc = np.mean([p.cross_corr_max for p in self.pairs])
    mean_vr = np.mean([p.variance_reduction for p in self.pairs])
    ax.set_title(f"Observed (black) vs Synthetic (red)  |  mean CC={mean_cc:.2f} mean VR={mean_vr:.2f}\n{obs_name}", pad=12)
```

### 4.7 改动 7：CSV 导出加指标列

`write_pair_csv`（[dsm_fit_compare_core.py:285-311](dsm_fit_compare_core.py#L285-L311)）fieldnames 加：

```python
"time_shift_s", "amplitude_factor", "cross_corr_max", "variance_reduction",
```

对应行加：

```python
"time_shift_s": f"{pair.time_shift_s:.5f}",
"amplitude_factor": f"{pair.amplitude_factor:.5f}",
"cross_corr_max": f"{pair.cross_corr_max:.5f}",
"variance_reduction": f"{pair.variance_reduction:.5f}",
```

---

## 5. 落地顺序与验证

按以下顺序实现，每步验证后再进下一步（遵循"先单道验证再批量"原则）：

1. **加 `cross_correlate_align` 函数 + WaveformPair 字段**（4.1, 4.2）
   - 单元测试：构造一段已知波形 + 平移版本，验证 τ 恢复正确、CC≈1、VR≈1
2. **`build_pairs` 接入 + checkbox**（4.3, 4.4）
   - 单道真实数据验证：勾选互相关对齐，看 τ 是否合理、P 是否对上
3. **拟合窗显示指标**（4.5）
   - 看每行 CC/VR/τ/A 是否合理，τ 分布是否符合第 2.6 节诊断树
4. **组总览窗显示 + CSV**（4.6, 4.7）
   - 整组批量看 mean CC/VR，导出 CSV 留档
5. **回到物理问题**：根据 τ 分布，按第 2.6 节决策树判断是深度 / 模型 / 定位问题，
   调 .inf 重新正演

---

## 6. 小结

| 你的问题 | 答案 |
|---------|------|
| 怎么判断拟合好坏 | 定量（CC/VR/A/τ/L2）+ 定性（肉眼叠绘看到时/形状/振幅）组合，不能只看残差 |
| 看拟合度吗 | 看，但要看组合指标，且必须在对齐+标定后算 |
| 肉眼比较形状和振幅吗 | 是，肉眼是必要的，指标只是辅助 |
| 按 pP 对齐但 P 没对上怎么办 | 主因：手动单震相对齐范式有缺陷（即使震源参数精确，真实结构与 PREM 偏差也会导致几秒误差）；改用互相关自动对齐。地壳厚度 13 km 是从观测反推的合理值（南桑威奇群岛弧洋弧环境），四舍五入+拾取误差带来 1-2 km 偏差，可微调 .inf 后重正演验证 |
| 残差怎么算 | 不能直接算 $\sum(d-s)^2$，要先互相关对齐→振幅标定→再算 VR（归一化残差） |

核心一句话：**先把互相关自动对齐做出来，"对不上"就从肉眼感觉变成可分析的 τ 数字**，
剩下的是按 τ 分布去调深度 / 模型 / 定位。
