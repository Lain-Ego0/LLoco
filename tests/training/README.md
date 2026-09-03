# Training acceptance

This tier records tests that execute optimizer updates, capacity runs, fixed
budgets, or behavior evaluation. GPU-heavy and long-running cases are not part
of per-commit CPU CI; their exact task/profile, environment count, iteration
count, checkpoint path, metrics, and code revision must be recorded in
`PROJECT_PROGRESS.md`.

Minimum runner updates for PPO, CTS, DreamWaQ, AMP, teacher, and student have
been executed against the current architecture. Historical 1024×1000 results
remain stability evidence for their recorded revision, not convergence proof
for the current one.
