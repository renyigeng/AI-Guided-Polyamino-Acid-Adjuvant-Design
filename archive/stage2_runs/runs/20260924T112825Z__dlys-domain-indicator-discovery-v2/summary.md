# D-Lys 第二嵌段三域指标发现 V3

> 第一阶段已固定D-赖氨酸。本阶段人工预注册三个可比机制域，17种X全部进入某一域；LLM只根据D-Lys体系约束与合格文献证据发现域内应比较的物化指标，不分类、不排序候选、不赋权。

## D1 芳香疏水-π作用域

- **Side-chain hydrophobicity** — role=primary_rank, direction=higher, basis=both.
- **Side-chain hydrogen-bond donor and acceptor site count** — role=primary_rank, direction=lower, basis=system_constraint.
- Knowledge gaps: No qualified native-X literature evidence was supplied for tyrosine (Y) in this domain. The descriptors selected are mature single-residue properties applicable to all three candidates, but the evidence base for Y is absent.; Aromatic ring count, side-chain size/volume, and geometric packing descriptors are mentioned in the domain guidance as validation axes, but no mature single-residue metric with a preregistered monotonic direction was identified. These remain validation_only and are not included as primary_rank indicators.; No mature single-residue descriptor for 'aromatic interaction strength' or 'pi-stacking propensity' was found that is not a custom composite score; therefore, such broad mechanisms are not translated into primary_rank indicators.

## D2 非芳香疏水-构象域

- **single-residue hydrophobicity** — role=primary_rank, direction=higher, basis=both.
- **side-chain volume** — role=primary_rank, direction=higher, basis=both.
- **beta-branching** — role=primary_rank, direction=absence, basis=literature.
- Knowledge gaps: No qualified native-X evidence was supplied for cysteine (C) in this domain; its neutral thiol special chemistry is flagged but not represented by a mature single-residue descriptor in the supplied evidence.; Mature single-residue descriptors for side-chain flexibility or conformational entropy are not represented in the supplied evidence and are not included as primary_rank indicators.; Alpha-helix propensity and beta-sheet propensity are competing conformational preferences and are not included as primary_rank indicators per domain guardrails; they remain validation_only/context_dependent.; The exact hydrophobicity scale and side-chain volume scale are not specified; tool mapping will occur later.

## D3 极性-氢键-构象相容域

- **Net side-chain charge at pH 7.4** — role=primary_rank, direction=closer_to_zero, basis=both.
- **Side-chain amide presence** — role=primary_rank, direction=presence, basis=literature.
- **Side-chain volume** — role=primary_rank, direction=higher, basis=literature.
- Knowledge gaps: No mature single-residue descriptor for side-chain conformational entropy or backbone conformational freedom is available that would allow a defensible preference direction for G and P within this domain; these remain validation_only/context_dependent.; Simple HBD/HBA donor/acceptor counts are not discriminative among C/D/E/N/Q/S/T and are therefore not returned as primary_rank descriptors.; No mature single-residue descriptor for side-chain pKa or ionization state at pH 7.4 is available that would refine the net charge descriptor beyond the standard formal charge calculation.; The supplied evidence does not provide a quantitative single-residue metric for side-chain hydrogen-bonding capacity that is both mature and discriminative within this candidate set.
