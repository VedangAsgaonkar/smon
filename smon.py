import argparse
import getpass
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

ME = getpass.getuser()
VIEWS = ["nodes", "users", "jobs", "queue", "qos"]
RUNNING = {"RUNNING", "COMPLETING", "SUSPENDED"}
UNUSABLE = ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING", "INVAL", "FUTURE", "MAINT", "REBOOT", "POWERED_DOWN")
MEM_MB = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}
JOB_FIELDS = [
    "JobID", "ArrayJobID", "ArrayTaskID", "UserName", "Account", "Partition", "QOS", "State", "Reason",
    "tres-alloc", "NumNodes", "NumCPUs", "NodeList", "TimeUsed", "TimeLimit", "SubmitTime", "StartTime",
    "PriorityLong", "Name",
]
KV = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")
INF = float("inf")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def mem_mb(s):
    m = re.fullmatch(r"([\d.]+)([KMGTP]?)", s.strip())
    return float(m[1]) * MEM_MB[m[2] or "M"] if m else 0.0


def num(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


def dur_secs(s):
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", s)
    if not m:
        return INF if s in ("UNLIMITED", "INFINITE") else 0.0
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def parse_tres(s):
    out = {}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = mem_mb(v) if k == "mem" else num(v)
    return out


def gpus_by_type(tres):
    typed = {k.split(":", 1)[1]: v for k, v in tres.items() if k.startswith("gres/gpu:") and v}
    total = tres.get("gres/gpu", 0)
    if total and not typed:
        typed = {"gpu": total}
    return typed


def split_top(s):
    parts, depth, cur = [], 0, ""
    for c in s:
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += (c == "[") - (c == "]")
        cur += c
    return parts + [cur] if cur else parts


def expand_range(r):
    out = []
    for piece in r.split(","):
        if "-" in piece:
            a, b = piece.split("-", 1)
            w = len(a) if a.startswith("0") else 0
            out += [str(i).zfill(w) for i in range(int(a), int(b) + 1)]
        else:
            out.append(piece)
    return out


def expand_hosts(s):
    if not s or s.startswith("(") or s == "N/A":
        return []
    hosts = []
    for part in split_top(s):
        chunks = re.split(r"\[([^\]]*)\]", part)
        lists = [[c] if i % 2 == 0 else expand_range(c) for i, c in enumerate(chunks)]
        hosts += ["".join(p) for p in product(*lists)]
    return hosts


def natural(s):
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", s) if t)


@dataclass
class Node:
    name: str
    state: str
    partitions: list
    cfg: dict
    alloc: dict
    load: float
    reason: str
    users: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))

    @property
    def usable(self):
        return not self.state.endswith("*") and not any(u in self.state for u in UNUSABLE)


@dataclass
class Job:
    id: str
    user: str
    account: str
    partition: str
    qos: str
    state: str
    reason: str
    tres: dict
    nnodes: int
    cpus: float
    nodelist: str
    used: str
    limit: str
    submit: datetime | None
    start: datetime | None
    priority: int
    name: str

    @property
    def gpus(self):
        return gpus_by_type(self.tres)

    @property
    def ngpu(self):
        return self.tres.get("gres/gpu", 0)


def parse_time(s):
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def get_nodes():
    nodes = {}
    for line in run(["scontrol", "-o", "show", "node"]).splitlines():
        d = dict(KV.findall(line.strip()))
        if "NodeName" not in d:
            continue
        nodes[d["NodeName"]] = Node(
            name=d["NodeName"],
            state=d.get("State", "?"),
            partitions=[p for p in d.get("Partitions", "").split(",") if p],
            cfg=parse_tres(d.get("CfgTRES", "")),
            alloc=parse_tres(d.get("AllocTRES", "")),
            load=num(d.get("CPULoad", "0")),
            reason=d.get("Reason", ""),
        )
    return nodes


def get_jobs():
    fmt = ",".join(f"{f}:|" for f in JOB_FIELDS)
    jobs = []
    for line in run(["squeue", "-h", "-r", "-a", "-O", fmt]).splitlines():
        v = line.split("|", len(JOB_FIELDS) - 1)
        if len(v) < len(JOB_FIELDS):
            continue
        d = dict(zip(JOB_FIELDS, v))
        task = d["ArrayTaskID"]
        jid = f"{d['ArrayJobID']}_{task}" if task not in ("N/A", "") else d["JobID"]
        tres = parse_tres(d["tres-alloc"])
        jobs.append(Job(
            id=jid,
            user=d["UserName"],
            account=d["Account"],
            partition=d["Partition"],
            qos=d["QOS"],
            state=d["State"],
            reason=d["Reason"],
            tres=tres,
            nnodes=max(int(num(d["NumNodes"])), 1),
            cpus=tres.get("cpu") or num(d["NumCPUs"]),
            nodelist=d["NodeList"],
            used=d["TimeUsed"],
            limit=d["TimeLimit"],
            submit=parse_time(d["SubmitTime"]),
            start=parse_time(d["StartTime"]),
            priority=int(num(d["PriorityLong"])),
            name=d["Name"].rstrip("|").strip(),
        ))
    return jobs


def parse_limits(v):
    out = {}
    for kv in v.split(","):
        m = re.fullmatch(r"(.+?)=(\w+)\(([^)]*)\)", kv)
        if m:
            out[m[1]] = (None if m[2] == "N" else num(m[2]), num(m[3]))
    return out


def get_qos():
    qos, cur, section, user = {}, None, None, None
    for line in run(["scontrol", "show", "assoc_mgr", "flags=qos"]).splitlines():
        s = line.strip()
        if s.startswith("QOS="):
            name = s[4:].split("(")[0]
            cur = qos[name] = {"name": name, "attrs": {}, "users": {}}
            section = None
            continue
        if cur is None or not s:
            continue
        if s in ("Account Limits", "User Limits"):
            section = s
            continue
        if section == "Account Limits":
            continue
        if section == "User Limits":
            if "=" not in s:
                if s != "No Users":
                    user = s.split("(")[0]
                    cur["users"][user] = {}
                continue
            target = cur["users"][user]
        else:
            target = cur["attrs"]
        for k, v in re.findall(r"(\S+?)=(\S*)", s):
            target[k] = v
    return qos


_sacct_cache = {}


def get_qos_static():
    if not _sacct_cache:
        fmt = "Name,Priority,Preempt,PreemptMode,MaxWall,GrpTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU"
        try:
            out = run(["sacctmgr", "-nP", "show", "qos", f"format={fmt}"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            out = ""
        for line in out.splitlines():
            d = dict(zip(fmt.split(","), line.split("|")))
            _sacct_cache[d["Name"]] = d
    return _sacct_cache


def tres_name(k):
    return k.split(":", 1)[1] if k.startswith("gres/gpu:") else k.replace("gres/", "")


def fmt_num(x):
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def fmt_mem(mb):
    return f"{mb / 1024**2:.1f}T" if mb >= 1024**2 * 10 else f"{mb / 1024:.0f}G"


def fmt_tres_val(k, v):
    return fmt_mem(v) if k == "mem" else fmt_num(v)


def fmt_gpus(g):
    return " ".join(f"{t}:{fmt_num(n)}" for t, n in sorted(g.items(), key=lambda x: -x[1]))


def fmt_dur(td):
    s = int(td.total_seconds())
    if s < 0:
        return "-"
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    return f"{d}d{h}h" if d else f"{h}h{m:02d}m" if h else f"{m}m"


def pretty_tres_str(s):
    return " ".join(f"{tres_name(k)}={v}" for k, v in (kv.split("=", 1) for kv in s.split(",") if "=" in kv))


def load_color(frac):
    return "green" if frac < 0.5 else "yellow" if frac < 0.9 else "red"


def bar(frac, width=10):
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    return Text("█" * n, style=load_color(frac)) + Text("░" * (width - n), style="grey35")


def ratio(used, total, width=8):
    t = Text(f"{fmt_num(used):>4}/{fmt_num(total):<4} ")
    return t + bar(used / total if total else 0, width)


def state_style(st):
    if any(u in st for u in UNUSABLE) or st.endswith("*"):
        return "red"
    if st.startswith("IDLE"):
        return "green"
    if st.startswith("MIXED"):
        return "yellow"
    if st.startswith("ALLOC"):
        return "magenta"
    return ""


def reason_style(r):
    if r in ("Resources", "Priority", "None"):
        return "yellow"
    if "Limit" in r or "Max" in r or r.startswith("QOS"):
        return "red"
    if r in ("Dependency", "BeginTime", "JobHeldUser", "JobHeldAdmin") or "held" in r:
        return "grey50"
    return ""


def user_text(u):
    return Text(u, style="bold cyan" if u == ME else "")


def short_state(st):
    head, *rest = st.split("+")
    flags = [f for f in rest if f in ("DRAIN", "MAINT", "REBOOT", "FAIL")]
    s = "+".join([head] + flags).lower()
    return s.replace("*", "") + ("*" if "NOT_RESPONDING" in st or head.endswith("*") else "")


@dataclass
class Snapshot:
    nodes: dict
    jobs: list
    qos: dict
    taken: datetime
    errors: list


def snapshot(args=None):
    errors = []

    def safe(fn, default):
        try:
            return fn()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errors.append(f"{fn.__name__}: {getattr(e, 'stderr', '') or e}".strip())
            return default

    nodes = safe(get_nodes, {})
    jobs = safe(get_jobs, [])
    qos = safe(get_qos, {})
    for j in jobs:
        if j.state not in RUNNING:
            continue
        hosts = expand_hosts(j.nodelist)
        for h in hosts:
            if h in nodes:
                u = nodes[h].users[j.user]
                u["cpu"] += j.cpus / len(hosts)
                u["gpu"] += j.ngpu / len(hosts)
    if args is not None:
        if args.partition:
            nodes = {k: n for k, n in nodes.items() if set(n.partitions) & set(args.partition)}
            jobs = [j for j in jobs if j.partition in args.partition]
        if args.user:
            jobs = [j for j in jobs if j.user in args.user]
        if args.qos:
            jobs = [j for j in jobs if j.qos in args.qos]
            qos = {k: q for k, q in qos.items() if k in args.qos}
    return Snapshot(nodes, jobs, qos, datetime.now(), errors)


@dataclass
class Row:
    key: str
    cells: list
    sort: list
    users: frozenset | None = None

    @property
    def text(self):
        return " ".join(c.plain if isinstance(c, Text) else str(c) for c in self.cells).lower()


@dataclass
class View:
    id: str
    title: str
    cols: list
    rows: list
    sort_col: int = 0
    reverse: bool = False

    def numeric(self, col):
        return bool(self.rows) and isinstance(self.rows[0].sort[col], (int, float))


L, R = "left", "right"


def cluster_totals(nodes):
    used, total, down = defaultdict(float), defaultdict(float), defaultdict(float)
    states = defaultdict(int)
    for n in nodes:
        states[short_state(n.state).split("+")[0].rstrip("*~#!%$@^-")] += 1
        for k in ["cpu", "mem"] + [k for k in n.cfg if k.startswith("gres/gpu:")]:
            c = n.cfg.get(k, 0)
            if n.usable:
                total[k] += c
                used[k] += n.alloc.get(k, 0)
            else:
                down[k] += c
    gpus = sorted((k for k in set(total) | set(down) if k.startswith("gres/gpu:")), key=lambda k: -total[k])
    return gpus, used, total, down, states


class Meter:
    def __init__(self, label, used, total, fmt=fmt_num, down=0):
        self.label, self.used, self.total, self.fmt, self.down = label, used, total, fmt, down

    def __rich_console__(self, console, options):
        val = f"{self.fmt(self.used)}/{self.fmt(self.total)}"
        if self.down:
            val += f" +{self.fmt(self.down)} down"
        n = max(options.max_width - 10, len(val) + 2)
        frac = self.used / self.total if self.total else 0
        filled = min(round(frac * n), n)
        start = n - len(val)
        bars = min(filled, start)
        t = Text(f"{self.label[:8]:<8}", style="bold cyan") + Text("[", style="bold")
        t += Text("|" * bars, style=load_color(frac)) + Text(" " * (start - bars))
        t += Text(val, style="bold" if filled < start else f"bold {load_color(frac)}")
        yield t + Text("]", style="bold")


def meters(snap, status=None):
    gpus, used, total, down, states = cluster_totals(snap.nodes.values())
    left = [Meter("CPU", used["cpu"], total["cpu"], fmt_num, down["cpu"]),
            Meter("Mem", used["mem"], total["mem"], fmt_mem, down["mem"])]
    right = [Meter(tres_name(k), used[k], total[k], fmt_num, down[k]) for k in gpus]
    run_ = [j for j in snap.jobs if j.state in RUNNING]
    pend = [j for j in snap.jobs if j.state == "PENDING"]
    my_run = [j for j in run_ if j.user == ME]
    my_pend = [j for j in pend if j.user == ME]
    node_states = []
    for s, c in sorted(states.items()):
        node_states += [(s, state_style(s.upper())), f" {c}  "]
    stats = [
        Text.assemble(("Jobs    ", "bold cyan"), f"{len(run_)} running, ", (f"{len(pend)} pending", "yellow"),
                      f", {len({j.user for j in run_})} active users,",
                      f" {fmt_num(sum(j.ngpu for j in pend))} GPUs requested"),
        Text.assemble(("Nodes   ", "bold cyan"), *node_states),
        Text.assemble(("You     ", "bold cyan"), f"{len(my_run)} running on ",
                      f"{fmt_num(sum(j.ngpu for j in my_run))} GPUs / {fmt_num(sum(j.cpus for j in my_run))} CPUs, ",
                      (f"{len(my_pend)} pending", "yellow")),
    ]
    lrows = left + stats
    grid = Table.grid(expand=True, padding=(0, 3))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    for i in range(max(len(lrows), len(right))):
        grid.add_row(lrows[i] if i < len(lrows) else "", right[i] if i < len(right) else "")
    parts = [grid]
    if status is not None:
        parts.append(status)
    parts += [Text(e, style="red") for e in snap.errors]
    return Group(*parts)


def nodes_view(snap):
    rows = []
    for n in snap.nodes.values():
        gtypes = sorted(k for k in n.cfg if k.startswith("gres/gpu:"))
        gpu = Text()
        for k in gtypes:
            if gpu:
                gpu += Text("  ")
            gpu += Text(f"{tres_name(k)} ") + ratio(n.alloc.get(k, 0), n.cfg[k], 6)
        users = Text()
        for u, r in sorted(n.users.items(), key=lambda x: (-x[1]["gpu"], -x[1]["cpu"])):
            if users:
                users += Text("  ")
            label = f"{fmt_num(r['gpu'])}g/{fmt_num(r['cpu'])}c" if r["gpu"] else f"{fmt_num(r['cpu'])}c"
            users += user_text(u) + Text(f":{label}", style="grey62")
        state = Text(short_state(n.state), style=state_style(n.state))
        if n.reason and not n.usable:
            users += Text(n.reason, style="red")
        mu, mt = n.alloc.get("mem", 0), n.cfg.get("mem", 0)
        cu, ct = n.alloc.get("cpu", 0), n.cfg.get("cpu", 0)
        gu = sum(n.alloc.get(k, 0) for k in gtypes)
        gt = sum(n.cfg.get(k, 0) for k in gtypes)
        rows.append(Row(
            n.name,
            [n.name, state, ",".join(n.partitions), ratio(cu, ct), f"{n.load:.0f}",
             Text(f"{fmt_mem(mu):>5}/{fmt_mem(mt):<5} ") + bar(mu / mt if mt else 0, 6), gpu or Text("-", style="grey50"),
             users],
            [natural(n.name), n.state, ",".join(n.partitions), cu / ct if ct else 0, n.load, mu / mt if mt else 0,
             gu / gt if gt else -1, len(n.users)],
            frozenset(n.users),
        ))
    cols = [("Node", L), ("State", L), ("Partitions", L), ("CPU", L), ("Load", R), ("Mem", L), ("GPU", L),
            ("Users (gpu/cpu)", L)]
    return View("nodes", "Nodes", cols, rows)


def users_view(snap):
    agg = defaultdict(lambda: {
        "rj": 0, "cpu": 0.0, "mem": 0.0, "gpu": defaultdict(float), "qos": defaultdict(lambda: [0.0, 0.0]),
        "pj": 0, "pcpu": 0.0, "pgpu": defaultdict(float), "nodes": set(),
    })
    for j in snap.jobs:
        a = agg[j.user]
        if j.state in RUNNING:
            a["rj"] += 1
            a["cpu"] += j.cpus
            a["mem"] += j.tres.get("mem", 0)
            for g, c in j.gpus.items():
                a["gpu"][g] += c
            a["qos"][j.qos][0] += j.ngpu
            a["qos"][j.qos][1] += j.cpus
            a["nodes"].update(expand_hosts(j.nodelist))
        elif j.state == "PENDING":
            a["pj"] += 1
            a["pcpu"] += j.cpus
            for g, c in j.gpus.items():
                a["pgpu"][g] += c
    rows = []
    for u, a in agg.items():
        ngpu, npg = sum(a["gpu"].values()), sum(a["pgpu"].values())
        byq = "  ".join(
            f"{q}:{fmt_num(g)}g/{fmt_num(c)}c" if g else f"{q}:{fmt_num(c)}c"
            for q, (g, c) in sorted(a["qos"].items(), key=lambda x: (-x[1][0], -x[1][1]))
        )
        blank = lambda v, f=fmt_num: f(v) if v else ""
        rows.append(Row(
            u,
            [user_text(u), blank(a["rj"]), blank(a["cpu"]), blank(a["mem"], fmt_mem), blank(ngpu),
             fmt_gpus(a["gpu"]), byq, blank(len(a["nodes"])), Text(blank(a["pj"]), style="yellow"),
             blank(a["pcpu"]), Text(fmt_gpus(a["pgpu"]), style="yellow")],
            [u, a["rj"], a["cpu"], a["mem"], ngpu, ngpu, byq, len(a["nodes"]), a["pj"], a["pcpu"], npg],
            frozenset([u]),
        ))
    cols = [("User", L), ("Run", R), ("CPUs", R), ("Mem", R), ("GPUs", R), ("GPU types", L), ("By QOS (gpu/cpu)", L),
            ("Nodes", R), ("Pend", R), ("Pend CPUs", R), ("Pend GPUs", L)]
    return View("users", "Users", cols, rows, sort_col=4, reverse=True)


def jobs_view(snap):
    rows = []
    for j in snap.jobs:
        if j.state not in RUNNING:
            continue
        el, lim = dur_secs(j.used), dur_secs(j.limit)
        left = lim - el
        rows.append(Row(
            j.id,
            [j.id, user_text(j.user), j.partition, j.qos, j.name, Text(j.state.lower(), style="green"),
             fmt_gpus(j.gpus), fmt_num(j.cpus), fmt_mem(j.tres.get("mem", 0)), j.used, j.limit,
             fmt_dur_secs(left), j.nodelist],
            [natural(j.id), j.user, j.partition, j.qos, j.name, j.state, j.ngpu, j.cpus, j.tres.get("mem", 0), el, lim,
             left, natural(j.nodelist)],
            frozenset([j.user]),
        ))
    cols = [("JobID", L), ("User", L), ("Partition", L), ("QOS", L), ("Name", L), ("State", L), ("GPUs", L),
            ("CPUs", R), ("Mem", R), ("Elapsed", R), ("Limit", R), ("Left", R), ("Nodes", L)]
    return View("jobs", "Running jobs", cols, rows, sort_col=6, reverse=True)


def fmt_dur_secs(s):
    if s == INF:
        return "∞"
    return fmt_dur(datetime.fromtimestamp(s) - datetime.fromtimestamp(0)) if s >= 0 else "-"


def ts(d):
    return d.timestamp() if d else INF


def queue_view(snap):
    jobs = [j for j in snap.jobs if j.state == "PENDING"]
    now = snap.taken
    rank = {}
    for part in {j.partition for j in jobs}:
        for i, j in enumerate(sorted((j for j in jobs if j.partition == part), key=lambda j: -j.priority)):
            rank[j.id] = i + 1
    rows = []
    for j in jobs:
        wait = (now - j.submit).total_seconds() if j.submit else 0
        rows.append(Row(
            j.id,
            [str(rank[j.id]), j.id, user_text(j.user), j.partition, j.qos, j.name, fmt_gpus(j.gpus), fmt_num(j.cpus),
             fmt_mem(j.tres.get("mem", 0)), j.limit, fmt_dur_secs(wait), str(j.priority),
             Text(j.reason, style=reason_style(j.reason)), j.start.strftime("%m-%d %H:%M") if j.start else ""],
            [rank[j.id], natural(j.id), j.user, j.partition, j.qos, j.name, j.ngpu, j.cpus, j.tres.get("mem", 0),
             dur_secs(j.limit), wait, j.priority, j.reason, ts(j.start)],
            frozenset([j.user]),
        ))
    cols = [("#", R), ("JobID", L), ("User", L), ("Partition", L), ("QOS", L), ("Name", L), ("GPUs", L), ("CPUs", R),
            ("Mem", R), ("Limit", R), ("Wait", R), ("Prio", R), ("Reason", L), ("Est. start", L)]
    return View("queue", "Pending jobs", cols, rows, sort_col=11, reverse=True)


def queue_grouped_view(snap):
    groups = defaultdict(list)
    for j in snap.jobs:
        if j.state == "PENDING":
            groups[(j.partition, j.user, j.qos, j.reason)].append(j)
    rows = []
    for (part, user, qos, reason), js in groups.items():
        g = defaultdict(float)
        for j in js:
            for k, v in j.gpus.items():
                g[k] += v
        start = min((j.start for j in js if j.start), default=None)
        oldest = min((j.submit for j in js if j.submit), default=None)
        wait = (snap.taken - oldest).total_seconds() if oldest else 0
        prio = max(j.priority for j in js)
        cpus = sum(j.cpus for j in js)
        rows.append(Row(
            "|".join((part, user, qos, reason)),
            [part, user_text(user), qos, Text(reason, style=reason_style(reason)), str(len(js)), fmt_gpus(g),
             fmt_num(cpus), str(prio), fmt_dur_secs(wait), start.strftime("%m-%d %H:%M") if start else ""],
            [part, user, qos, reason, len(js), sum(g.values()), cpus, prio, wait, ts(start)],
            frozenset([user]),
        ))
    cols = [("Partition", L), ("User", L), ("QOS", L), ("Reason", L), ("Jobs", R), ("GPUs", L), ("CPUs", R),
            ("Max prio", R), ("Oldest wait", R), ("Est. start", L)]
    return View("queue_grouped", "Pending jobs (grouped)", cols, rows, sort_col=7, reverse=True)


def limited(lims):
    return {k: v for k, v in lims.items() if v[0] is not None}


def usage_text(k, lim, use):
    s = f"{tres_name(k)} {fmt_tres_val(k, use)}/{fmt_tres_val(k, lim)}"
    style = "red bold" if lim and use >= lim else "yellow" if lim and use >= 0.75 * lim else ""
    return Text(s, style=style)


def join_texts(texts, sep="  "):
    out = Text()
    for i, t in enumerate(texts):
        if i:
            out += Text(sep)
        out += t
    return out


def qos_views(snap):
    static = get_qos_static()
    counts = defaultdict(lambda: defaultdict(float))
    pend = defaultdict(int)
    for j in snap.jobs:
        c = counts[j.qos]
        if j.state in RUNNING:
            c["run"] += 1
            c["gpu"] += j.ngpu
            c["cpu"] += j.cpus
        elif j.state == "PENDING":
            c["pend"] += 1
            c["pgpu"] += j.ngpu
            pend[(j.qos, j.user)] += 1
    rows, lrows = [], []
    for name, q in snap.qos.items():
        a, s, c = q["attrs"], static.get(name, {}), counts.get(name, {})
        grp = dict(limited(parse_limits(a.get("GrpTRES", ""))))
        for k in ("GrpJobs", "GrpSubmitJobs"):
            grp.update(limited(parse_limits(f"{k}={a.get(k, '')}")))
        peruser = " ".join(filter(None, [
            pretty_tres_str(s.get("MaxTRESPU", "")),
            f"jobs={s['MaxJobsPU']}" if s.get("MaxJobsPU") else "",
            f"submit={s['MaxSubmitPU']}" if s.get("MaxSubmitPU") else "",
        ]))
        if not (c or grp or peruser or q["users"]):
            continue
        prio = num(a.get("Priority", "0"))
        rows.append(Row(
            name,
            [Text(name, style="bold"), fmt_num(prio), s.get("MaxWall", ""), s.get("Preempt", ""),
             a.get("PreemptMode", "").lower(), fmt_num(c.get("run", 0)), fmt_num(c.get("gpu", 0)),
             fmt_num(c.get("cpu", 0)), Text(fmt_num(c.get("pend", 0)), style="yellow"),
             Text(fmt_num(c.get("pgpu", 0)), style="yellow"), peruser,
             join_texts([usage_text(k, *v) for k, v in grp.items()])],
            [name, prio, dur_secs(s.get("MaxWall", "")), s.get("Preempt", ""), a.get("PreemptMode", ""),
             c.get("run", 0), c.get("gpu", 0), c.get("cpu", 0), c.get("pend", 0), c.get("pgpu", 0), peruser,
             len(grp)],
        ))
        for user, d in q["users"].items():
            lim = limited(parse_limits(d.get("MaxTRESPU", "")))
            jl = parse_limits(f"j={d.get('MaxJobsPU', '')}").get("j", (None, 0))
            sl = parse_limits(f"s={d.get('MaxSubmitJobsPU', '')}").get("s", (None, 0))
            busy = any(v[1] for v in lim.values()) or pend.get((name, user))
            if not (busy or (jl[0] is not None and jl[1]) or (sl[0] is not None and sl[1])):
                continue
            tres = [usage_text(k, *v) for k, v in lim.items() if v[1] or (k == "gres/gpu" and v[0])]
            capped = any(v[0] is not None and v[1] >= v[0] for v in [*lim.values(), jl, sl] if v[0])
            fj = lambda v: Text(f"{fmt_num(v[1])}/{fmt_num(v[0])}", style="red bold" if v[1] >= v[0] else "") \
                if v[0] is not None else Text(fmt_num(v[1]) if v[1] else "")
            lrows.append(Row(
                f"{name}|{user}",
                [name, user_text(user), Text("yes", style="red bold") if capped else "", join_texts(tres), fj(jl),
                 fj(sl), Text(str(pend.get((name, user), "") or ""), style="yellow")],
                [name, user, capped, len(tres), jl[1], sl[1], pend.get((name, user), 0)],
                frozenset([user]),
            ))
    cols = [("QOS", L), ("Prio", R), ("MaxWall", R), ("Preempts", L), ("Mode", L), ("Run", R), ("GPUs", R),
            ("CPUs", R), ("Pend", R), ("Pend GPUs", R), ("Per-user limits", L), ("Group limits (used/limit)", L)]
    if not any(r.sort[-1] for r in rows):
        cols = cols[:-1]
        for r in rows:
            r.cells, r.sort = r.cells[:-1], r.sort[:-1]
    lcols = [("QOS", L), ("User", L), ("Capped", L), ("TRES (used/limit)", L), ("Jobs", R), ("Submitted", R),
             ("Pending", R)]
    return (View("qos", "QOS", cols, rows, sort_col=1, reverse=True),
            View("limits", "Per-user QOS usage vs limits", lcols, lrows, sort_col=0))


def build_views(snap, grouped=False):
    qos, limits = qos_views(snap)
    return {
        "nodes": nodes_view(snap),
        "users": users_view(snap),
        "jobs": jobs_view(snap),
        "queue": queue_grouped_view(snap) if grouped else queue_view(snap),
        "qos": qos,
        "limits": limits,
    }


def sorted_rows(view, col=None, reverse=None):
    col = view.sort_col if col is None else col
    reverse = view.reverse if reverse is None else reverse
    return sorted(view.rows, key=lambda r: r.sort[col], reverse=reverse)


def to_rich(view, limit=0):
    t = Table(title=view.title, title_justify="left", header_style="bold")
    for name, just in view.cols:
        t.add_column(name, justify=just)
    rows = sorted_rows(view)
    for r in rows[:limit] if limit else rows:
        t.add_row(*r.cells)
    if limit and len(rows) > limit:
        t.caption = f"showing {limit} of {len(rows)}; use -n 0 for all"
    return t


def print_once(args):
    console = Console()
    snap = snapshot(args)
    views = build_views(snap, grouped=args.group)
    parts = [Text(f"smon  {snap.taken:%Y-%m-%d %H:%M:%S}  user={ME}", style="grey50"), meters(snap), Text("")]
    for v in args.views:
        names = ["qos", "limits"] if v == "qos" else [v]
        for n in names:
            parts += [to_rich(views[n], args.limit if n in ("jobs", "queue") else 0), Text("")]
    console.print(Group(*parts))


def parse_args(argv):
    p = argparse.ArgumentParser(prog="smon", description="htop-style slurm monitor")
    p.add_argument("views", nargs="*", metavar="VIEW",
                   help=f"with --once: any of {', '.join(VIEWS)} (default: all)")
    p.add_argument("-1", "--once", action="store_true", help="print once and exit instead of the live UI")
    p.add_argument("-d", "--delay", type=float, default=3.0, help="seconds between refreshes (default 3)")
    p.add_argument("-u", "--user", action="append", help="only these users' jobs (repeatable)")
    p.add_argument("--me", action="store_true", help="start with the 'mine' filter on")
    p.add_argument("-p", "--partition", action="append", help="only these partitions (repeatable)")
    p.add_argument("-q", "--qos", action="append", help="only these QOS (repeatable)")
    p.add_argument("-g", "--group", action="store_true", help="group the pending queue by user/QOS/reason")
    p.add_argument("-n", "--limit", type=int, default=50, help="with --once: max job rows (0 = all)")
    args = p.parse_args(argv)
    bad = [v for v in args.views if v not in VIEWS]
    if bad:
        p.error(f"unknown view {', '.join(bad)}; choose from {', '.join(VIEWS)}")
    args.views = args.views or VIEWS
    if args.me and args.once:
        args.user = (args.user or []) + [ME]
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.once or not sys.stdout.isatty():
        print_once(args)
        return
    from smon_tui import SmonApp
    SmonApp(args).run()


if __name__ == "__main__":
    main()
