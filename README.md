# visual_task

Marmoset visual and eye-tracking experiment, built on
[mxbiflow](https://github.com/CHiPmxbi/mxbiflow).

## Setup

```bash
git clone https://github.com/kimdenyse99/visual_task.git
cd visual_task
uv sync --locked
uv run main.py
```

mxbiflow depends on two private repositories, `cagelab/pymoteGO` and
`cagelab/cogmatrix-python`, so `uv sync` needs a GitHub account with read access
to them.
