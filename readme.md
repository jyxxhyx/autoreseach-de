# DE Autoresearch

在 **300 维平移旋转 Rosenbrock** 问题上，用 **差分进化（Differential Evolution）** 做固定时间预算（5 分钟）的优化实验。`prepare.py` 固定问题与评测；`train.py` 为可调算法与超参的实验入口。

## 依赖

- Python 3
- NumPy

## 快速开始

**1. 一次性生成问题实例**（写入 `~/.cache/de_autoresearch/problem.pkl`）：

```bash
python prepare.py
```

**2. 运行一次优化**：

```bash
python train.py
```

若使用 [uv](https://github.com/astral-sh/uv)，也可：

```bash
uv run train.py
```

**3. 从日志中取最优适应度**（例如重定向到 `run.log` 后）：

```bash
grep "^best_fitness:" run.log
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `prepare.py` | 问题定义、维度与边界、时间预算、评测接口；实验协议中视为只读。 |
| `train.py` | DE 实现与超参（种群、F、CR、策略等）；为改进算法时主要修改的文件。 |
| `program.md` | 自主实验流程、输出格式、`results.tsv` 记录约定等完整说明。 |
| `results.tsv` | 可选：按 `program.md` 记录各次实验的 commit、best_fitness、状态与描述。 |

## 指标说明

- 目标：**最小化** `best_fitness`（全局最优为 `0.0`）。
- 标准结尾会打印 `best_fitness:`、`training_seconds:`、`pop_size`、`F`、`CR`、`strategy`、`total_generations:` 等行，便于脚本解析。

更详细的实验循环、约束（如仅使用 NumPy）与记录规范见 **`program.md`**。
