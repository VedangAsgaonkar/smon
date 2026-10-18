import subprocess

from rich.console import Group
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Static, TabbedContent, TabPane

import smon as S

TABS = [
    ("nodes", "Nodes", ["nodes"]),
    ("users", "Users", ["users"]),
    ("jobs", "Jobs", ["jobs"]),
    ("queue", "Queue", ["queue"]),
    ("qos", "QOS", ["qos", "limits"]),
]


class Detail(ModalScreen):
    BINDINGS = [Binding("escape,q,enter,space", "close", "Close")]

    def __init__(self, title, body):
        super().__init__()
        self.title_text, self.body = title, body

    def compose(self) -> ComposeResult:
        with Vertical(id="detail"):
            yield Static(Text(self.title_text, style="bold"), id="detail-title")
            with VerticalScroll():
                yield Static(self.body)

    def action_close(self):
        self.dismiss()


class SmonApp(App):
    TITLE = "smon"
    CSS = """
    Screen { layout: vertical; }
    #meters { height: auto; padding: 0 1; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    DataTable { height: 1fr; }
    #t-qos { height: auto; max-height: 12; margin-bottom: 1; }
    #filter { display: none; }
    #filter.visible { display: block; }
    Detail { align: center middle; }
    #detail { width: 90%; height: 85%; border: round $accent; background: $surface; padding: 0 1; }
    #detail-title { padding-bottom: 1; }
    """
    BINDINGS = [
        Binding("1", "tab('nodes')", "Nodes"),
        Binding("2", "tab('users')", "Users"),
        Binding("3", "tab('jobs')", "Jobs"),
        Binding("4", "tab('queue')", "Queue"),
        Binding("5", "tab('qos')", "QOS"),
        Binding("slash", "filter", "Filter"),
        Binding("m", "mine", "Mine"),
        Binding("s", "sort", "Sort"),
        Binding("i", "invert", "Invert"),
        Binding("g", "group", "Group queue"),
        Binding("r", "refresh", "Refresh"),
        Binding("plus", "delay(-1)", "Faster", show=False),
        Binding("minus", "delay(1)", "Slower", show=False),
        Binding("escape", "clear_filter", "Clear filter", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.delay = max(1.0, args.delay)
        self.mine = args.me
        self.grouped = args.group
        self.filter = ""
        self.snap = None
        self.views = {}
        self.sort = {}
        self.timer = None

    def compose(self) -> ComposeResult:
        yield Static(Text("loading…", style="grey50"), id="meters")
        with TabbedContent(initial="nodes"):
            for tid, label, tables in TABS:
                with TabPane(label, id=tid):
                    for t in tables:
                        yield DataTable(id=f"t-{t}", cursor_type="row", zebra_stripes=True)
        yield Input(placeholder="filter rows (any column) — enter to keep, esc to clear", id="filter")
        yield Footer()

    def on_mount(self):
        self.timer = self.set_interval(self.delay, self.action_refresh)
        self.action_refresh()
        self.query_one("#t-nodes", DataTable).focus()

    def action_refresh(self):
        self.fetch()

    @work(thread=True, exclusive=True, group="fetch")
    def fetch(self):
        snap = S.snapshot(self.args)
        self.call_from_thread(self.apply, snap)

    def apply(self, snap):
        self.snap = snap
        self.render_all()

    def render_all(self):
        if self.snap is None:
            return
        self.views = S.build_views(self.snap, grouped=self.grouped)
        for name, view in self.views.items():
            self.populate(name, view)
        tabs = self.query_one(TabbedContent)
        for tid, label, tables in TABS:
            n = self.query_one(f"#t-{tables[0]}", DataTable).row_count
            unit = " groups" if tid == "queue" and self.grouped else ""
            try:
                tabs.get_tab(tid).label = f"{label} ({n}{unit})"
            except Exception:
                pass
        self.update_meters()

    def update_meters(self):
        if self.snap is None:
            return
        bits = [("updated ", "grey50"), (f"{self.snap.taken:%H:%M:%S}", ""), (f" · every {S.fmt_num(self.delay)}s", "grey50")]
        dt = self.active_table()
        if dt is not None and dt.id[2:] in self.views:
            v = self.views[dt.id[2:]]
            col, rev = self.sort_state(v)
            bits.append((f" · sort {v.cols[col][0]} {'▼' if rev else '▲'}", "grey50"))
        if self.mine:
            bits.append((f" · mine ({S.ME})", "bold cyan"))
        if self.filter:
            bits.append((f" · filter '{self.filter}'", "bold yellow"))
        if self.grouped:
            bits.append((" · queue grouped", "grey50"))
        self.query_one("#meters", Static).update(S.meters(self.snap, Text.assemble(*bits)))

    def sort_state(self, view):
        return self.sort.get(view.id, (view.sort_col, view.reverse))

    def keep(self, row):
        if self.mine and row.users is not None and S.ME not in row.users:
            return False
        return not self.filter or self.filter.lower() in row.text

    def populate(self, name, view):
        dt = self.query_one(f"#t-{name}", DataTable)
        col, rev = self.sort_state(view)
        rows = sorted((r for r in view.rows if self.keep(r)), key=lambda r: r.sort[col], reverse=rev)
        old_key = None
        if dt.row_count:
            try:
                old_key = dt.coordinate_to_cell_key(dt.cursor_coordinate).row_key.value
            except Exception:
                old_key = None
        sx, sy = dt.scroll_x, dt.scroll_y
        dt.clear(columns=True)
        for i, (cname, just) in enumerate(view.cols):
            label = f"{cname} {'▼' if rev else '▲'}" if i == col else cname
            dt.add_column(Text(label, style="bold reverse" if i == col else "bold", justify=just), key=str(i))
        for r in rows:
            cells = []
            for c, (_, just) in zip(r.cells, view.cols):
                t = c.copy() if isinstance(c, Text) else Text(str(c))
                t.justify = just
                cells.append(t)
            dt.add_row(*cells, key=r.key)
        if old_key is not None:
            try:
                dt.move_cursor(row=dt.get_row_index(old_key), animate=False, scroll=False)
            except Exception:
                pass
        self.call_after_refresh(dt.scroll_to, sx, sy, animate=False)

    def active_table(self):
        f = self.focused
        if isinstance(f, DataTable):
            return f
        try:
            pane = self.query_one(TabbedContent).active
            return self.query_one(f"#{pane} DataTable", DataTable)
        except Exception:
            return None

    def repopulate(self, dt):
        name = dt.id[2:]
        if name in self.views:
            self.populate(name, self.views[name])
        self.update_meters()

    def on_data_table_header_selected(self, event):
        name = event.data_table.id[2:]
        view = self.views.get(name)
        if view is None:
            return
        col, rev = self.sort_state(view)
        if event.column_index == col:
            self.sort[view.id] = (col, not rev)
        else:
            self.sort[view.id] = (event.column_index, view.numeric(event.column_index))
        self.repopulate(event.data_table)

    def action_sort(self):
        dt = self.active_table()
        if dt is None or dt.id[2:] not in self.views:
            return
        view = self.views[dt.id[2:]]
        col = (self.sort_state(view)[0] + 1) % len(view.cols)
        self.sort[view.id] = (col, view.numeric(col))
        self.repopulate(dt)

    def action_invert(self):
        dt = self.active_table()
        if dt is None or dt.id[2:] not in self.views:
            return
        view = self.views[dt.id[2:]]
        col, rev = self.sort_state(view)
        self.sort[view.id] = (col, not rev)
        self.repopulate(dt)

    def action_tab(self, tid):
        self.query_one(TabbedContent).active = tid

    def on_tabbed_content_tab_activated(self, event):
        self.call_after_refresh(self.focus_active)

    def action_mine(self):
        self.mine = not self.mine
        self.render_all()

    def action_group(self):
        self.grouped = not self.grouped
        self.render_all()
        self.action_tab("queue")

    def action_delay(self, step):
        self.delay = max(1.0, self.delay + step)
        self.timer.stop()
        self.timer = self.set_interval(self.delay, self.action_refresh)
        self.update_meters()

    def action_filter(self):
        box = self.query_one("#filter", Input)
        box.add_class("visible")
        box.value = self.filter
        box.focus()

    def action_clear_filter(self):
        box = self.query_one("#filter", Input)
        box.remove_class("visible")
        box.value = ""
        if self.filter:
            self.filter = ""
            self.render_all()
        self.focus_active()

    def focus_active(self):
        pane = self.query_one(TabbedContent).active
        f = self.focused
        if not (isinstance(f, DataTable) and any(getattr(a, "id", None) == pane for a in f.ancestors)):
            self.query_one(f"#{pane} DataTable", DataTable).focus()
        self.update_meters()

    def on_input_changed(self, event):
        self.filter = event.value.strip()
        self.render_all()

    def on_input_submitted(self, event):
        self.query_one("#filter", Input).remove_class("visible")
        self.focus_active()

    def on_data_table_row_selected(self, event):
        name, key = event.data_table.id[2:], event.row_key.value
        if self.snap is None or key is None:
            return
        title, body = self.detail(name, key)
        self.push_screen(Detail(title, body))

    def jobs_table(self, jobs, title):
        snap = S.Snapshot(self.snap.nodes, jobs, self.snap.qos, self.snap.taken, [])
        running = S.jobs_view(snap)
        pending = S.queue_view(snap)
        running.title, pending.title = f"{title}: running ({len(running.rows)})", f"{title}: pending ({len(pending.rows)})"
        return [S.to_rich(running), Text(""), S.to_rich(pending)]

    def detail(self, name, key):
        jobs = self.snap.jobs
        if name == "nodes":
            on = [j for j in jobs if key in S.expand_hosts(j.nodelist)]
            raw = cmd(["scontrol", "show", "node", key])
            return f"node {key}", Group(*self.jobs_table(on, key)[:1], Text(""), Text(raw, style="grey70"))
        if name == "users":
            return f"user {key}", Group(*self.jobs_table([j for j in jobs if j.user == key], key))
        if name in ("jobs", "queue") and not self.grouped or name == "jobs":
            return f"job {key}", Text(cmd(["scontrol", "show", "job", key]))
        if name == "queue":
            part, user, qos, reason = key.split("|", 3)
            sel = [j for j in jobs if (j.partition, j.user, j.qos, j.reason) == (part, user, qos, reason)]
            return f"{user} · {qos} · {reason}", Group(*self.jobs_table(sel, user)[2:])
        if name == "qos":
            return f"qos {key}", Text(cmd(["sacctmgr", "show", "qos", key, "-p"]).replace("|", "  "))
        if name == "limits":
            qos, user = key.split("|", 1)
            return f"{user} in {qos}", Group(*self.jobs_table([j for j in jobs if j.user == user and j.qos == qos], user))
        return key, Text("")


def cmd(c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=10).stdout or "(no output)"
    except Exception as e:
        return str(e)
