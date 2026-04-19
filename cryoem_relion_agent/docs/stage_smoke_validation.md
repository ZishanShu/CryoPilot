# 阶段冒烟项目（stage smoke）验证

## 目的

在**不下载新数据**的前提下，用本地已有的完整 RELION 工程（例如 `Tutorial5.0`）复制出多个「截断到某一阶段」的目录树，用来**冒烟验证** Agent 的：

- 阶段识别（`scan-project` / `ProjectWatcher`）
- 下一步决策与 dry-run 路径（`run-next --dry-run`）
- 报告生成（`generate-report`，可选）

这不是科学重建或收敛性测试，只验证**编排与决策逻辑**在真实目录结构下是否自洽。

## 为何要删除 `.cryoem_agent`

Agent 在工程根目录下使用 `.cryoem_agent/` 保存 `sample_state.json`、`action_history.jsonl` 等。若从「跑过 Agent」的工程直接复制，旧状态会干扰本次扫描与决策。每个冒烟副本都会**删除** `.cryoem_agent`，相当于从「仅 RELION」视角重新观察工程。

## 创建六个冒烟工程

在 `cryoem_relion_agent` 根目录下执行（路径按你的机器修改）：

```bash
./scripts/make_stage_smoke_projects.sh /path/to/full_relion_project /path/to/smoke_parent
```

会在 `smoke_parent/` 下生成：

| 目录名 | 含义（相对完整工程的裁剪） |
|--------|---------------------------|
| `picking_ready` | 去掉 AutoPick 及之后所列阶段目录 |
| `extract_ready` | 去掉 Extract 及之后 |
| `class2d_ready` | 去掉 Class2D 及之后 |
| `init3d_ready` | 去掉 InitialModel 及之后 |
| `refine3d_ready` | 去掉 Refine3D 及之后 |
| `postprocess_ready` | 去掉 PostProcess 及之后（保留 Refine3D 等更早阶段） |

具体删除列表与代码中的 `STAGE_SMOKE_REMOVALS` 一致（见 `app/stage_smoke_layout.py`），缺失的目录会跳过，不报错。

**磁盘与时间**：每个子目录都是**完整拷贝**一份源工程，请预留足够空间。

## 批量运行 CLI 检查

```bash
./scripts/run_stage_smoke_checks.sh /path/to/smoke_parent
# 可选：指定输出目录
./scripts/run_stage_smoke_checks.sh /path/to/smoke_parent /tmp/my_smoke_out
# 同时生成 report.md / report.json
WITH_REPORT=1 ./scripts/run_stage_smoke_checks.sh /path/to/smoke_parent /tmp/my_smoke_out
```

会在输出目录下为每个 fixture 生成 `scan.json`、`run_next.json`（以及可选的 `report.md` / `report.json`），stderr 单独写入 `*.stderr`。

汇总表格：

```bash
./scripts/summarize_stage_smoke.sh /tmp/my_smoke_out
# 等价于：
python -m app.stage_smoke_layout summarize --out-dir /tmp/my_smoke_out
```

## 手工单条命令（等价操作）

```bash
python -m app.main scan-project --project /path/to/smoke_parent/picking_ready
python -m app.main run-next --project /path/to/smoke_parent/picking_ready --dry-run
python -m app.main generate-report --project /path/to/smoke_parent/picking_ready --out report.md --out-json report.json
```

对其余五个子目录重复即可。

## 期望的「下一步动作」（决策对齐）

在 QC 与目录证据正常的前提下，通常期望 `run-next --dry-run` 指向：

| Fixture | 期望 next action（参考） |
|---------|-------------------------|
| `picking_ready` | `run_autopick` |
| `extract_ready` | `run_extract` |
| `class2d_ready` | `run_class2d` |
| `init3d_ready` | `run_init3d_or_class3d` |
| `refine3d_ready` | `run_refine3d` |
| `postprocess_ready` | `run_postprocess` |

说明：

- **`postprocess_ready`** 会删除 `PostProcess/` 目录，因此 `scan-project` 里的 **stage 字符串往往是 `Refine3D`**（文件系统上「已完成」的最高阶段），这与「下一步应跑后处理」并不矛盾：策略依据「已有 Refine3D、尚无 PostProcess」仍可给出 `run_postprocess`。
- 若某一步出现与上表不符的 action，先检查源工程是否缺少 STAR/地图等证据，再对照 `configs/policy_rules.yaml` 与 QC 判定。

## 若 watcher 仍识别出「更后」阶段

可能原因包括：删除目录后仍有 **STAR/软链接/其它路径** 指向残留输出，或 RELION 目录命名与默认假设不一致。可：

1. 确认对应阶段目录确实已删除；
2. 在问题工程上跑 `scan-project`，查看 `state.stage` 与 `notes`；
3. 必要时在副本中再手动删除残留 job 目录或修正符号链接。

## 与单元测试里「微型 fixture」的关系

`tests/helpers/fixture_projects.py` 中的微型目录树用于 CI 快速回归；本文档描述的是**全量工程复制**的**手工/本地**冒烟路径，二者互补。
