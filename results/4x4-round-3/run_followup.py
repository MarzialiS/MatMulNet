"""Round 3, follow-up: LM continued from the low-residual points of phase 1, and the runs at 47
and 48 terms with the settings of those at 46 and 49, all in one pool on the 8 P-cores. Jobs that
fail (e.g. a W&B timeout before training starts) are tried again, at most three rounds. Run from
the repository root."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "experiments")
import lm_continue  # noqa: E402
import phase1  # noqa: E402
import run_ranks  # noqa: E402

for attempt in range(3):
    jobs = lm_continue.jobs() + run_ranks.rank_jobs([47, 48])
    if not jobs:
        break
    print(f"round {attempt + 1}", flush=True)
    phase1.pinned_pool(run_ranks.A, jobs)
print("FOLLOWUP DONE", flush=True)
