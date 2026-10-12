# 录屏建议（约 4–6 分钟）

1. **20–30 秒：任务定义**
   - 固定 D-Lys，从 17 个 X 中做实验优先级压缩。
   - 三个机制域，不做跨域统一总分。

2. **40–60 秒：AI/数据来源**
   - 展示 frozen `discovered_indicators.csv`：ICL 负责发现指标，不直接选赢家。
   - 展示 `criteria_used.csv`：AAindex / RDKit / peptides.py 才提供最终数值。

3. **1–2 分钟：现场运行确定性算法**
   - Notebook 点击“重跑 ranking” Cell。
   - D1：只有一个独立排序轴，因此单轴，不硬凑 TOPSIS。
   - D2/D3：现场跑 PyMCDM TOPSIS。

4. **1 分钟：稳健性**
   - D2：TOPSIS=Borda=亮氨酸，±20% 权重扰动稳定。
   - D3：TOPSIS=Borda=谷氨酰胺；说明 Q>N 是实验优先级，不是普适定律。
   - 展示 criterion ablation 是“删除整个机制轴”的诊断，不是权重调参。

5. **30–45 秒：D1 scale audit**
   - 展示多个 published hydrophobicity scales 对 F/W/Y 的敏感性检查。
   - 强调它只做 robustness audit，不反向改 Kyte-Doolittle 主规则。

6. **20 秒：最后结果**
   - F：芳香路线
   - L：非芳香疏水路线
   - Q：极性酰胺路线
   - N 可作为 Q 的近邻实验对照（预算允许时）。
