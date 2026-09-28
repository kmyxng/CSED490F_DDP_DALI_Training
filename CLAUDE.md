# CLAUDE.md

Before starting work, read:

- [report/spec.md](report/spec.md): the Lab 3 assignment (DDP / DALI implementation & profiling), converted from the PDF
- [report/progress.md](report/progress.md): the plan, checklist, and work log. Keep it updated as work progresses.

Notes:

- Local machine is macOS with no GPU. CUDA/NCCL/DALI can't run locally, so check with `py_compile` / `bash -n` only; real runs happen on a Vast.ai instance.
- Remotes: `origin` = team fork (`kmyxng/...`), where commits are pushed; `upstream` = course repo (`csed490f2026/...`), read-only.
