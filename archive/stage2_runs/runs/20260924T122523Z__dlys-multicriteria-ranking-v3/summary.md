# D-Lys 第二嵌段自适应决策排序 V3.1

> 第一阶段已固定D-赖氨酸。域内若保留≥2个合格且非冗余指标，使用等权TOPSIS；若预注册允许且资格化/冗余审计后仅剩一个独立决策轴，则使用单轴确定性排序，而不是人为制造第二指标。已知项目/实验参考代表仅在全部排名完成后做post-hoc对照。

## 结果

- **芳香疏水-π作用域 → 苯丙氨酸**；方法=preregistered_single_axis_rank；仅保留一个独立合格决策轴，因此不伪造TOPSIS/Borda/权重扰动；诊断=single_criterion_fallback_used|no_independent_second_axis_after_qualification_or_redundancy_audit；post-hoc参考一致=True。
- **非芳香疏水-构象域 → 亮氨酸**；方法=TOPSIS；TOPSIS/Borda一致=True；criterion-ablation=1/3；权重±20%=6/6；诊断=criterion_ablation_changes_winner|pareto_dominance_present；post-hoc参考一致=True。
- **极性-氢键-构象相容域 → 谷氨酰胺**；方法=TOPSIS；TOPSIS/Borda一致=True；criterion-ablation=2/3；权重±20%=6/6；诊断=criterion_ablation_changes_winner|pareto_dominance_present；post-hoc参考一致=True。

## 方法边界

- Stage 1 already selected D-lysine; this module ranks the remaining 17 X candidates within three human-preregistered comparison domains defined by the frozen taxonomy contract.
- The LLM is upstream only: it discovers domain-relevant physicochemical indicators from qualified evidence and system constraints. It never assigns candidates to domains, ranks candidates, or assigns numerical weights.
- Metric mapping is deterministic and local. Unresolved concepts are excluded rather than converted into ad-hoc custom scores.
- When a domain retains at least two qualified nonredundant primary_rank criteria, all receive equal TOPSIS weight and TOPSIS is the primary decision method. D1 may use a preregistered single-axis hydropathy fallback only if qualification/redundancy auditing leaves hydropathy as the sole independent decision axis; this avoids fabricating a second criterion merely to force MCDA.
- Borda/rank-sum is an independent ordinal diagnostic only for genuine multicriteria domains. It is not presented as an independent check for a one-criterion fallback, where it would be mathematically redundant.
- Criterion-ablation (formerly LOO) asks whether the conclusion depends on retaining each physical mechanism axis. It is a mechanism-dependence diagnostic, not a pass/fail test.
- The deterministic ±20% one-criterion-at-a-time weight perturbation check tests local robustness around equal weights for multicriteria domains and never tunes weights; it is not applicable to a one-criterion fallback.
- Pareto dominance and TOPSIS winner margin are reported explicitly so library warnings do not substitute for interpretation.
- All 17 candidates remain in their frozen domain decision matrix; hard_gate suggestions are audit-only and do not silently remove candidates.
- Post-hoc reference targets are loaded only after rankings are computed and cannot affect discovery, mapping, weights, taxonomy, or ranking.
- Short-peptide AP, CPP, CamSol, generic immune scores, and molecular dynamics do not enter final V3 stage-2 ranking.
