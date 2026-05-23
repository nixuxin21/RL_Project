# Experiments

当前仓库仍保留多数实验脚本在项目根目录，原因是这些脚本之间存在直接导入关系，贸然移动会破坏已有命令和结果复现路径。

本目录用于区分实验分支：

- `archive/`: 早期探索分支，保留用于复现历史结果，不作为新增实验默认入口。
- `paper_experiment_presets.json`: paper-grade execution-mismatch preset 配置。
- `run_paper_experiment_suite.py`: 将 preset 展开为可恢复、可 dry-run 的 evaluator jobs，并保存 command/config snapshots。
- `analyze_paper_experiment_logs.py`: 读取 raw structured logs，生成 bootstrap CI、paired deltas、win/tie/loss 表、LaTeX snippet、PNG/PDF 图和 markdown interpretation。

运行入口：

```bash
../.venv/bin/python experiments/run_paper_experiment_suite.py --presets smoke_test
../.venv/bin/python experiments/run_paper_experiment_suite.py --presets main_hard --dry-run
../.venv/bin/python experiments/analyze_paper_experiment_logs.py --input-dir /tmp/irs_aircomp_paper_suite --output-dir /tmp/irs_aircomp_paper_suite_analysis_smoke
```

新增实验建议优先放在根目录现有主题脚本旁边，或在确认导入关系稳定后再迁移到本目录的主题子目录。迁移脚本时必须同步更新：

- `README.md`
- `docs/PROJECT_MAP.md`
- `Makefile`
- `tests/smoke_checks.py`
