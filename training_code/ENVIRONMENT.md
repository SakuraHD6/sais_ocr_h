# Environment Status

Checked on 2026-06-08 after cleanup.

The current system Python already has the core training dependencies:

```text
torch 2.8.0+cu128
cuda available: true
ultralytics 8.4.51
timm 1.0.27
```

Default for this rebuild: use the existing system Python first to avoid spending
time reinstalling large GPU wheels. If dependency conflicts appear, create a
fresh `.venv` from `requirements.txt`.

## Pip Mirror

Use Tsinghua PyPI mirror for future dependency downloads:

```bash
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

