# D-Lys 第二嵌段筛选：最终方法卡

## 1. 任务边界

固定 `D-Lys` 功能块，只对 17 种天然氨基酸作为第二嵌段 `X` 做**实验前候选优选**。本阶段不声称直接预测完整 `C6-X_m-(D-Lys)10` 的最终长链材料性能，也不在这里优化聚合度 `m` 或端基。

## 2. 三个机制域

- **D1 芳香疏水-π作用域**：F / W / Y
- **D2 非芳香疏水-构象域**：A / C / I / L / M / V
- **D3 极性-氢键-构象相容域**：N / Q / S / T / D / E / G / P

域划分先于最终排名冻结；不同机制域不做跨域总分。

## 3. AI 在哪里

AI/ICL **不直接输出赢家**。上游先通过文献证据检索与语义重排得到上下文，再让 LLM/ICL 在每个域内提出“科学上值得比较的物化维度”。这些指标之后必须经过资格化、成熟工具映射、方向门禁和冗余审计。

## 4. 数值从哪里来

- **AAindex**：已发表的单残基理化标度，例如 Kyte-Doolittle hydropathy、side-chain volume。
- **RDKit**：从氨基酸结构计算/标注芳香性、β-支化、酰胺官能团、HBD/HBA 等结构性质。
- **peptides.py**：理论电荷与多种成熟氨基酸/肽描述符；D1 额外用其 hydrophobicity tables 做尺度敏感性审计。

LLM 不生成最终数值，也不生成 TOPSIS 权重。

## 5. 最终决策逻辑

### D1：F / W / Y

ICL 曾提出不止一个候选维度，但资格化和域内冗余审计后，只保留 **Kyte-Doolittle hydropathy** 作为独立、方向明确的正式排序轴。因此不强行凑第二指标，不伪装成 TOPSIS，而使用预注册单轴排序。

另外单独做 **hydropathy-scale sensitivity audit**：遍历 `peptides.py` 暴露的多种已发表疏水标度，统一方向后检查 F/W/Y 的 winner 稳健性。该审计只做敏感性说明，不反向修改主规则。

### D2：A / C / I / L / M / V

正式三轴：

1. hydropathy：higher better（AAindex）
2. β-branching：absence preferred（RDKit）
3. side-chain volume：higher better（AAindex）

三个轴等权进入 TOPSIS；Borda 只作独立序数诊断。

### D3：N / Q / S / T / D / E / G / P

正式三轴：

1. side-chain charge at pH 7.4：closer to zero（peptides.py）
2. side-chain amide：presence preferred（RDKit）
3. side-chain volume：higher better（AAindex）

这里应把 **Q>N** 理解为当前机制假设下的实验优先级，而不是普适材料定律。Q 与 N 在电荷和酰胺上非常接近，Q 的主要算法优势来自更大的 side-chain size/reach proxy。

## 6. TOPSIS 做什么

对真正的多指标域（D2/D3）：

1. 构建 `候选 × 指标` 决策矩阵。
2. 将 benefit/cost/closer-to-zero/absence 等方向转换成可操作的 benefit/cost 类型。
3. 使用 PyMCDM TOPSIS，指标等权。
4. 构造正理想解与负理想解。
5. 计算每个候选到两类理想解的距离，得到 closeness/preference；越高越优。
6. 输出主排名。

Borda、criterion ablation、±20% 单指标权重扰动和 Pareto 关系均为诊断，不用于反向调权重。

## 7. 当前主实验候选

- **D1 → 苯丙氨酸 F**：预注册单轴排序
- **D2 → 亮氨酸 L**：TOPSIS；Borda 同意；局部权重扰动稳定
- **D3 → 谷氨酰胺 Q**：TOPSIS；Borda 同意；局部权重扰动稳定

如实验预算允许，可把 **N** 作为 D3 的机制近邻备选/对照。

## 8. 最重要的表述边界

推荐说：**AI 辅助指标发现 + 成熟数据库/工具定量 + 确定性多指标候选优选。**

不要说：**有监督机器学习已经预测或证明 F/L/Q 的完整材料性能最好。**
