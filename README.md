# 全球大学生生命科学挑战赛：面向肿瘤疫苗的二元聚氨基酸佐剂与递送载体分层筛选

## 1. 项目简介

本项目用于对二元聚氨基酸疫苗递送材料的组成进行**实验前候选优先级筛选**。材料形式记为：

`C6-X_m-(D-Lys)_n`

其中 D-赖氨酸作为已确定的功能块类型，`X` 为第二嵌段候选氨基酸；`m` 与 `n` 的具体聚合度及嵌段比例均不由本计算流程预设，留待后续合成与实验优化。

本包同时提供两种使用方式：

- **`main.py`**：一键复现最终筛选结果的命令行入口，适合自动化、评测与回归验证。
- **`项目演示教程.ipynb`**：面向评委与读者的完整交互式教程，逐步展示检索、证据综合、指标映射、排序与稳健性分析，并实时核对结果。

## 2. 最终候选

| 机制域 | 优先候选 | 材料表示 | 说明 |
|---|---|---|---|
| 芳香疏水-π作用域 | 苯丙氨酸（F） | `C6-F_m-(D-Lys)_n` | 当前域内优先候选 |
| 非芳香疏水-构象域 | 亮氨酸（L） | `C6-L_m-(D-Lys)_n` | 当前指标体系下综合排序靠前 |
| 极性-氢键-构象相容域 | 谷氨酰胺（Q） | `C6-Q_m-(D-Lys)_n` | 当前指标体系下综合排序靠前 |

功能块：**D-赖氨酸（D-Lys）**。

正式候选清单见：`results/results.csv`。

## 3. 目录结构

```text
.
├── LICENSE                     # MIT 开源许可
├── README.md
├── main.py                     # 主运行入口（一键复现）
├── 项目演示教程.ipynb           # 完整演示/教程（详见第 4 节）
├── requirements.txt            # 最终可复现运行所需依赖
├── requirements-upstream.txt   # 上游证据/描述符再生所需额外依赖
├── configs/                    # Stage 1/2 冻结配置
├── data/                       # 冻结证据与 Stage 2 决策输入
├── notebook_demo_output/       # 跑 项目演示教程.ipynb 生成的结果文件
├── prompts/                    # 关键 LLM/ICL 提示词
├── provenance/                 # 文献 API、LLM 输入输出、环境等溯源记录
├── results/                    # 标准候选清单及稳健性结果
├── src/                        # 最终运行代码
└── tests/                      # 可复现性测试
```

## 4. 项目演示教程（Jupyter Notebook）

`项目演示教程.ipynb` 是本项目的**完整演示与说明文档**，与 `main.py` 互为补充：

- `main.py` 调用 `src/` 中的正式管线代码，是“可复现的命令行入口”；
- 教程 Notebook 独立重算并可视化关键步骤（文献漏斗、MedCPT 热图、描述符热图、TOPSIS 距离与排名、稳健性热图），同时把数学原理与科学边界写清楚，适合答辩演示与读者理解。

运行教程时只读取 `data/` 与 `results/` 中的冻结记录，图表统一输出到 `notebook_demo_output/`，**不会覆盖正式结果文件**。

## 5. 运行环境与安装

- Python：**3.12.14**；最终筛选为 CPU 任务，不需要 GPU。
- 完整已观察环境快照保存在：`provenance/environment/`。

```bash
python -m pip install -r requirements.txt
```

如需重新生成上游 AAindex / RDKit / peptides.py 描述符或 MedCPT 文献证据，再安装：

```bash
python -m pip install -r requirements-upstream.txt
```

> 最终提交入口使用已冻结的文献证据和描述符输入，**不会重新访问 Europe PMC、不会调用 LLM，也不需要 API Key**。

## 6. 一键复现与测试

```bash
python main.py
python -m pytest -q
```

`python main.py` 默认输出到 `results/`，也可指定输出目录：

```bash
python main.py --output-dir reproduced_results
```

预期终端核心输出：

```text
Stage 1 functional block: D-Lys

Stage 2 within-domain priorities:
  芳香疏水-π作用域 -> 苯丙氨酸 (F)
  非芳香疏水-构象域 -> 亮氨酸 (L)
  极性-氢键-构象相容域 -> 谷氨酰胺 (Q)

Cross-domain total score: NO
Frozen-result regression check: PASS
```

生成文件包括：`results.csv`、`stage2_all_rankings.csv`、`robustness_summary.csv`、`d1_hydropathy_scale_summary.csv`、`criterion_ablation.csv`、`weight_perturbation.csv`、`run_summary.json`。

## 7. 方法与第三方工具说明

### Stage 1

- 文献来源：Europe PMC；
- 语义重排：`ncbi/MedCPT-Cross-Encoder`；
- LLM：开发/证据综合阶段调用记录中保存了实际模型、提示词、响应和 token 使用情况；已保存记录中的模型标识包括 `deepseek-v4-flash`；
- 最终比赛运行：只读取已经冻结的结构化证据，不再次调用外部服务。

### Stage 2

上游成熟描述符来源包括：AAindex、RDKit、peptides.py；多准则排序实现：PyMCDM TOPSIS；Borda 排名、指标删除和权重扰动用于辅助稳健性检查。

## 8. 结果解释边界

本项目是**计算辅助的实验前优先级筛选**，并未对聚合材料最终实验性能的直接预测：

- F/L/Q 分别来自三个不同机制域，不存在跨域总排名；
- TOPSIS 排名依赖当前选定的描述符、方向和权重，应理解为当前筛选框架下的优先候选；
- `m` 与 `n`、端基优化、粒径、自组装形貌、递送效率与免疫效果仍需实验验证；
- MedCPT/RAG 分数用于文献相关性与证据整理，不等价于免疫激活强度。

## 9. 作者与项目贡献

团队名称：AI辅助设计疫苗佐剂高分子材料

软件开发：
- 耿仁义：核心算法设计、代码实现、数据处理与可视化、Notebook演示教程与文档撰写
- 彭博：核心算法设计、数据处理与可视化、文档撰写
- 王大鹏：核心算法设计、项目统筹监督

实验研究：
- 王晓龙：材料设计、湿实验、数据处理与可视化、项目统筹、项目文件撰写、路演PPT制作、路演展示
- 赵智彤：材料设计、湿实验、数据处理与可视化
- 王怀志：材料设计、湿实验、数据处理与可视化、路演展示
- 王宏博：材料设计、湿实验、数据处理与可视化
- 王小斌：材料设计、湿实验、数据处理与可视化
- 丁建勋：材料设计、项目统筹监督

## 10. 可追溯性与开源

为便于复核，本包保留检索配置与结果、MedCPT 重排结果、结构化 RAG 证据、LLM/ICL 提示词与原始响应、Stage 2 指标发现记录、冻结描述符矩阵与最终排序输入、最终结果与稳健性结果，以及已验证的 Python 环境快照。这些内容主要位于 `data/`、`configs/`、`prompts/` 和 `provenance/`。

本项目以 **MIT License** 开源，详见 `LICENSE`。