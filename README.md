# smon

htop-style live monitor for slurm: cluster meters, nodes, per-user usage, running jobs, the pending queue and QOS limits.

## Setup

```
pixi install
pixi run smon
```

To call it as plain `smon` (fish):

```
alias --save smon /lfs/furiosa/0/vedanga/smon/.pixi/envs/default/bin/smon
```

## Usage

```
smon                 # live full-screen UI, refreshes every 3s
smon -d 10           # refresh every 10s
smon --me            # start with only your rows shown
smon -p il -q il-lo  # restrict to partitions / QOS (repeatable)
smon --once          # print once and exit (also used automatically when piped)
smon --once queue -g # print only some views: nodes, users, jobs, queue, qos
```

## Keys

| key | action |
| --- | --- |
| `1`–`5` | Nodes, Users, Jobs, Queue, QOS tabs |
| `↑ ↓ PgUp PgDn` | move through rows |
| `enter` | details for the row: jobs on a node, a user's jobs, `scontrol show job`, jobs in a queue group |
| `/` | filter rows by any text (enter keeps it, esc clears it) |
| `m` | only rows involving you |
| `s` / `i` | next sort column / reverse the sort (or click a column header) |
| `g` | pending queue as one row per job, or grouped by partition, user, QOS and reason |
| `+` / `-` | refresh faster / slower |
| `r` | refresh now |
| `q` | quit (or close a detail popup) |

## What is shown

- **Header**: CPU, memory and per-GPU-type meters over usable nodes (capacity on down or drained nodes listed as `+N down`), job counts, node states and your own usage.
- **Nodes**: CPU, memory and GPU allocation per node and which users hold it (`user:gpus g/cpus c`).
- **Users**: running jobs, CPUs, memory and GPUs per user, split by QOS, plus pending jobs and the GPUs they ask for.
- **Jobs**: running jobs with elapsed time, limit and time left.
- **Queue**: pending jobs with priority rank in their partition, wait time, reason and slurm's estimated start.
- **QOS**: priority, max wall time, what each QOS preempts, usage and per-user limits; below it, each user's usage against those limits, marked when capped.

Data comes from `scontrol show node`, `squeue`, `scontrol show assoc_mgr flags=qos` and `sacctmgr show qos`; nothing needs elevated permissions.
