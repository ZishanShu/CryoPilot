# cryoem_relion_agent

一个基于 RELION 的 cryo-EM 调度与决策 Agent（单机 CLI 版）。

## 项目目标

`cryoem_relion_agent` 不是替代 RELION 重建，而是“骑在 RELION 上面”做自动化调度：

- 监控 RELION project 目录
- 抽取 STAR/LOG/QC 指标并统一成状态对象
- 根据可配置规则给出下一步 Action
- 调用 Scheme/命令执行下一步（支持 dry-run）
- 在 map 达标后路由结构建模工具
- 输出结构化 JSON + Markdown 报告
- 所有状态可落盘、可恢复、可追踪

## 运行环境

- Python 3.11（推荐）
- 依赖见 `requirements.txt`

安装：

```bash
cd /root/share/szs/CryoAgent/cryoem_relion_agent
python -m pip install -r requirements.txt
```

## 项目结构

```text
cryoem_relion_agent/
├── app/
├── agent/
├── relion/
├── memory/
├── configs/
├── schemas/
├── tests/
├── examples/
├── scripts/
├── docs/
├── requirements.txt
└── README.md
```

## 配置文件说明

所有默认配置位于 `configs/`：

- `policy_rules.yaml`: 决策阈值和优先级
- `router_rules.yaml`: 建模路由阈值和优先级
- `schemes.yaml`: action -> 命令模板映射

如果配置文件缺失，代码内有默认兜底逻辑，不会直接崩溃。

## RELION project 目录假设

第一版默认按常见目录扫描（存在即视为阶段已出现）：

- `Import/`
- `MotionCorr/`
- `CtfFind/`
- `AutoPick/`
- `Extract/`
- `Class2D/`
- `InitialModel/`
- `Class3D/`
- `Refine3D/`
- `CtfRefine/`
- `Polish/`
- `PostProcess/`

STAR/LOG 解析采用“保守且可扩展”的启发式实现，解析失败时会优雅降级。

## CLI 用法

查看帮助：

```bash
python -m app.main --help
```

初始化案例库：

```bash
python -m app.main init-db --db-path cases.db
```

扫描项目：

```bash
python -m app.main scan-project --project /path/to/relion_project
```

决策下一步：

```bash
python -m app.main decide-next --project /path/to/relion_project
```

执行下一步（默认 dry-run）：

```bash
python -m app.main run-next --project /path/to/relion_project --dry-run
```

结构建模路由：

```bash
python -m app.main route-structure \
  --project /path/to/relion_project \
  --known-sequence \
  --chain-count 2 \
  --af3-available
```

生成报告：

```bash
python -m app.main generate-report \
  --project /path/to/relion_project \
  --out report.md \
  --out-json report.json
```

## 阶段冒烟验证（可选）

使用本地**完整 RELION 工程**复制并裁剪出多个「阶段就绪」子工程，用于手工跑 `scan-project` / `run-next --dry-run` / `generate-report` 做决策与识别冒烟检查（不引入新数据、不替代单元测试）。

详见 [`docs/stage_smoke_validation.md`](docs/stage_smoke_validation.md)。脚本入口：

- `scripts/make_stage_smoke_projects.sh` — 生成六个冒烟目录
- `scripts/run_stage_smoke_checks.sh` — 批量跑 CLI 并落盘输出
- `scripts/summarize_stage_smoke.sh` — 汇总 `scan` / `run-next` 结果表格

## 运行测试

当前使用 `unittest`，无需额外安装 `pytest`：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 当前版本边界

- 单颗粒 single-particle 流水线
- 不含 cryo-ET 分支
- 不含显微镜硬件控制
- 不训练 foundation model
- 不依赖 Celery/Redis/FastAPI 等复杂框架

## 下一步可扩展方向

- 增强 STAR 解析（loop 表头级解析）
- 增强 LOG 语义抽取（更细粒度 QC）
- 引入更细的策略 DSL（多分支策略图）
- 对接真实 RELION Scheme 参数模板
- 报告中增加历史案例对比与趋势
