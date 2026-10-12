# D-Lys 第二嵌段最终候选报告 V3.1

> 方法：合格文献证据 → LLM/ICL域内指标发现 → 成熟AAindex/RDKit/peptides.py数值映射 → 自适应确定性决策。保留≥2个合格非冗余指标时采用等权TOPSIS；若预注册允许且仅剩一个独立轴，则采用单轴排序，不人为制造第二指标。Borda、criterion-ablation、±20%权重扰动与Pareto仅在真正多指标域中作为诊断。已知参考代表仅post-hoc对照，不参与筛选。

- indicator run: `E:\Project\tiaozhansai\runs\20260924T112825Z__dlys-domain-indicator-discovery-v2`
- ranking run: `E:\Project\tiaozhansai\runs\20260924T122523Z__dlys-multicriteria-ranking-v3`

## 正式域内代表

- **芳香疏水-π作用域 → 苯丙氨酸**；方法=preregistered_single_axis_rank；资格化/冗余审计后仅保留一个独立决策轴，因此不伪造多指标TOPSIS、Borda或权重敏感性；诊断标记=single_criterion_fallback_used|no_independent_second_axis_after_qualification_or_redundancy_audit。
- **非芳香疏水-构象域 → 亮氨酸**；方法=TOPSIS；Borda一致=True；criterion-ablation=1.0/3.0；权重扰动=6.0/6.0；诊断标记=criterion_ablation_changes_winner|pareto_dominance_present。
- **极性-氢键-构象相容域 → 谷氨酰胺**；方法=TOPSIS；Borda一致=True；criterion-ablation=2.0/3.0；权重扰动=6.0/6.0；诊断标记=criterion_ablation_changes_winner|pareto_dominance_present。

## 解释边界

- 这是候选优选/prioritization，不宣称是有监督机器学习对长链材料性能的直接预测。
- AI的作用是从限定证据中发现域内可比较物化维度；数值来自成熟工具/数据库，最终决策是确定性的。
- 只有真正存在≥2个独立合格指标的域才使用TOPSIS；单轴域强行凑第二指标会造成重复投票，因此显式降级为单轴排序。
- criterion-ablation改变winner说明某个物理机制轴不可随意删除；它不是自动判定结果无效。
- Borda只看名次、TOPSIS保留数值距离，因此两者不一致时必须报告，不反向调权重。
- post-hoc reference match只用于结果核验，不能触发重新调指标、域划分或权重。
