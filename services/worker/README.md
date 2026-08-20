# Local Worker

`robolab-worker` implements the platform-side half of `robolab-job-v1`.
The platform creates a unique `var/runs/<uuid>/` directory through
`robolab_core.create_job_run`; the Worker starts the executable Skill in its
own POSIX process group and records `stdout.log`, `stderr.log`,
`events.jsonl`, `result.json`, and content-hashed files in `artifacts/`.

It refuses to start third-party code as root. Cancellation sends `SIGTERM` to
the whole group and escalates to `SIGKILL` after a timeout.
