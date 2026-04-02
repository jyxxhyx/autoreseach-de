# DE Autoresearch

This is an experiment to have the LLM autonomously optimize a Differential Evolution algorithm for the Shifted Rotated Rosenbrock problem (300D).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun15`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed problem definition, evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Algorithm, hyperparameters, operators.
4. **Verify the problem exists**: Check that `~/.cache/de_autoresearch/problem.pkl` exists. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each experiment runs on a single CPU core. The optimization runs for a **fixed time budget of 5 minutes** (wall clock optimization time, excluding startup).
Launch it simply as:

`python train.py`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: mutation strategy (rand1, best1, or invent your own), crossover operator, boundary handling, adaptive parameter control, population topology, local search hybrids, F/CR/pop_size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed problem instance, evaluation function, dimension, bounds, and time budget.
- Change the problem or its dimension. The landscape is fixed.
- Modify the output format. The `best_fitness:` line is the ground truth metric.
- Install new packages. Use only numpy (already available).

**The goal is simple: get the lowest best_fitness.**

The problem is a 300-dimensional Shifted Rotated Rosenbrock function with a known
global optimum of 0.0. In practice, reaching below 1e+1 is strong, below 1e-2
is excellent. Your target is to push best_fitness as close to 0 as possible
within the 5-minute budget.

**Simplicity criterion**: All else being equal, simpler is better.
A marginal improvement that adds 30 lines of ad-hoc logic is not worth it.
Removing code and getting equal or better results is a great outcome.

**The first run**: Your very first run should always establish the baseline
by running train.py as-is (no modifications).

## Output format

Once the script finishes it prints a summary like this:

best_fitness: 3.741928e+01
training_seconds: 300.2
total_seconds: 305.1
pop_size: 150
F: 0.8
CR: 0.9
strategy: rand1
total_generations: 1842


You can extract the key metric from the log file:

`grep "^best_fitness:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).
The TSV has a header row and 5 columns:

commit best_fitness time_seconds status description

1. git commit hash (short, 7 chars)
2. best_fitness achieved (e.g. 3.74e+01) — use 0 for crashes
3. training time in seconds (e.g. 300.1) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:
commit best_fitness time_seconds status description
a1b2c3d 3.7419e+01 300.1 keep baseline rand1 F=0.8
b2c3d4e 2.8934e+01 300.2 keep increase pop_size to 300
c3d4e5f 3.8100e+01 300.0 discard switch to best1 strategy
d4e5f6g 0.0000e+00 0.0 crash pop_size=10000 OOM killed


## The experiment loop

LOOP FOREVER:
1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: `grep "^best_fitness:" run.log`
6. If the grep output is empty, the run crashed. Read `tail -n 30 run.log` to diagnose. If unfixable, log "crash" and move on.
7. Record the results in the tsv.
8. If best_fitness improved (lower), keep the commit (advance) a and go back to step 1.
9. If best_fitness is equal or worse, git reset back and start the next loop.

**Timeout**: Each experiment should take ~5 minutes. If a run exceeds 10 minutes,
kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (bug, math error, etc.), use your judgment:
easy fix → fix and re-run. Fundamentally broken → skip and move on.

**NEVER STOP**: Once the loop has begun, do not pause to ask the human.
Do NOT ask "should I keep going?". Run indefinitely until manually stopped.
If you run out of ideas, think harder — try adaptive F/CR schedules,
try JADE-style self-adaptation, try hybrid local search, try multi-population,
try chaos perturbation, try entirely new mutation operators.


