import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import re
import hashlib
from datetime import datetime, date, time
from pandas.errors import EmptyDataError

# =========================================================
# PERSISTENT DATABASE (STREAMLIT CLOUD SAFE)
# =========================================================
DB_DIR = "/mount/src/.data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "smt_spit.db")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="SMT Spit Analytics", layout="wide")

REJECT_CODES = {2, 3, 4, 5, 7}

# Line setup (board counting model)
LINE1_MACHINES = {"EPS16"}
LINE2_MACHINES = {"IINEO682", "IIN2-053-2", "IIN2-053-1"}
LINE2_DIVISOR = 3  # line2 boards estimated as logs/3

# Master BOM format
MASTER_BOM_COMP_COL_INDEX = 0  # Column A
MASTER_BOM_COST_COL_INDEX = 9  # Column J

# Filename example: 20260106091251-IIN2-053-2.csv
FILENAME_RE = re.compile(r"^(?P<dt>\d{14})-(?P<machine>.+?)(?:\.[A-Za-z0-9]+)?$")

# =========================================================
# DB HELPERS
# =========================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init(conn: sqlite3.Connection):
    # Versioned BOMs
    conn.execute("""
    CREATE TABLE IF NOT EXISTS bom_versions (
        bom_id INTEGER PRIMARY KEY AUTOINCREMENT,
        bom_name TEXT NOT NULL,
        uploaded_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS bom_items (
        bom_id INTEGER NOT NULL,
        component TEXT NOT NULL,
        unit_cost REAL NOT NULL,
        PRIMARY KEY (bom_id, component),
        FOREIGN KEY(bom_id) REFERENCES bom_versions(bom_id) ON DELETE CASCADE
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bom_items_component ON bom_items(component)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bom_items_bomid ON bom_items(bom_id)")

    # Logs + Events
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        file_hash TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_dt TEXT,          -- ISO datetime (derived from filename)
        machine TEXT,          -- derived from filename
        board_name TEXT,       -- from B1
        mo TEXT,               -- from D1
        ingested_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        component TEXT,
        description TEXT,
        location TEXT,
        board_name TEXT,
        mo TEXT,
        file_dt TEXT,
        machine TEXT,
        unit_cost REAL,
        cost REAL,
        FOREIGN KEY(file_hash) REFERENCES logs(file_hash) ON DELETE CASCADE
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_dt ON events(file_dt)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_board ON events(board_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_mo ON events(mo)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_machine ON events(machine)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_comp ON events(component)")
    conn.commit()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# =========================================================
# PARSERS
# =========================================================
def parse_cost(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except:
        return np.nan

def parse_dt_machine_from_filename(filename: str):
    base = os.path.basename(filename)
    m = FILENAME_RE.match(base)
    if not m:
        return None, None
    dt_str = m.group("dt")
    machine = m.group("machine")
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        return dt.isoformat(sep=" "), machine
    except:
        return None, machine

def safe_read_csv(bytes_data, **kwargs):
    try:
        return pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1", **kwargs)
    except EmptyDataError:
        raise

def _clean_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s

def read_header_board_mo(file_bytes: bytes, filename: str):
    ext = filename.lower().split(".")[-1]
    try:
        if ext in ("xls", "xlsx"):
            header = pd.read_excel(io.BytesIO(file_bytes), nrows=1, header=None)
        else:
            header = safe_read_csv(file_bytes, nrows=1, header=None)
    except Exception:
        return None, None

    try:
        board = _clean_cell(header.iloc[0, 1])  # B1
        mo = _clean_cell(header.iloc[0, 3])     # D1
        return board, mo
    except Exception:
        return None, None

def read_body_df(file_bytes: bytes, filename: str):
    ext = filename.lower().split(".")[-1]
    try:
        if ext in ("xls", "xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes), skiprows=2, header=None, usecols=range(12))
        else:
            df = safe_read_csv(file_bytes, skiprows=2, header=None, usecols=range(12))
    except EmptyDataError:
        return None
    except Exception:
        return None

    if df is None or df.empty:
        return None
    df = df.iloc[:, :12]
    if df.shape[1] < 12:
        return None
    df.columns = list("ABCDEFGHIJKL")
    return df

# =========================================================
# BOM (VERSIONED, MASTER BOM A/J across all sheets)
# =========================================================
def ingest_master_bom(conn: sqlite3.Connection, bom_bytes: bytes, bom_name: str) -> int:
    xls = pd.ExcelFile(io.BytesIO(bom_bytes))
    items = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if df.shape[1] <= max(MASTER_BOM_COMP_COL_INDEX, MASTER_BOM_COST_COL_INDEX):
            continue

        for i in range(len(df)):
            comp = _clean_cell(df.iat[i, MASTER_BOM_COMP_COL_INDEX])
            if not comp:
                continue
            if comp.lower() in {"component", "part", "part number", "item", "sku"}:
                continue

            cost = parse_cost(df.iat[i, MASTER_BOM_COST_COL_INDEX])
            if pd.isna(cost):
                continue

            items.append((comp, float(cost)))

    if not items:
        return 0

    uploaded_at = datetime.now().isoformat(sep=" ")
    cur = conn.execute(
        "INSERT INTO bom_versions(bom_name, uploaded_at) VALUES (?, ?)",
        (bom_name, uploaded_at)
    )
    bom_id = cur.lastrowid

    # last wins per component within this version
    tmp = {}
    for comp, cost in items:
        tmp[comp] = cost

    conn.executemany(
        "INSERT INTO bom_items(bom_id, component, unit_cost) VALUES (?,?,?)",
        [(bom_id, c, tmp[c]) for c in tmp.keys()]
    )
    conn.commit()
    return len(tmp)

def list_boms(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT bom_id, bom_name, uploaded_at FROM bom_versions ORDER BY bom_id DESC",
        conn
    )

def get_bom_lookup(conn: sqlite3.Connection, selected_bom_ids: list[int] | None) -> dict:
    if not selected_bom_ids:
        sql = """
        SELECT bi.component, bi.unit_cost
        FROM bom_items bi
        JOIN (
            SELECT component, MAX(bom_id) AS max_bom_id
            FROM bom_items
            GROUP BY component
        ) latest
        ON bi.component = latest.component AND bi.bom_id = latest.max_bom_id
        """
        rows = conn.execute(sql).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    placeholders = ",".join(["?"] * len(selected_bom_ids))
    sql = f"""
    SELECT bi.component, bi.unit_cost
    FROM bom_items bi
    JOIN (
        SELECT component, MAX(bom_id) AS max_bom_id
        FROM bom_items
        WHERE bom_id IN ({placeholders})
        GROUP BY component
    ) latest
    ON bi.component = latest.component AND bi.bom_id = latest.max_bom_id
    """
    rows = conn.execute(sql, selected_bom_ids).fetchall()
    return {r[0]: float(r[1]) for r in rows}

# =========================================================
# INGEST LOGS
# =========================================================
def ingest_logs(conn: sqlite3.Connection, uploads):
    bom_lookup = get_bom_lookup(conn, selected_bom_ids=None)

    skipped = []
    inserted_files = 0
    inserted_events = 0

    for up in uploads:
        filename = up.name
        b = up.getvalue()

        if not b:
            skipped.append((filename, "Empty file"))
            continue

        file_hash = sha256_bytes(b)
        if conn.execute("SELECT 1 FROM logs WHERE file_hash=?", (file_hash,)).fetchone():
            skipped.append((filename, "Already ingested (same hash)"))
            continue

        dt_iso, machine = parse_dt_machine_from_filename(filename)
        board, mo = read_header_board_mo(b, filename)

        df = read_body_df(b, filename)
        if df is None:
            skipped.append((filename, "No readable data after skiprows=2"))
            continue

        conn.execute(
            "INSERT INTO logs(file_hash, filename, file_dt, machine, board_name, mo, ingested_at) VALUES (?,?,?,?,?,?,?)",
            (file_hash, filename, dt_iso, machine, board, mo, datetime.now().isoformat(sep=" "))
        )
        inserted_files += 1

        ev_rows = []
        for _, r in df.iterrows():
            try:
                if int(r["L"]) in REJECT_CODES:
                    comp = str(r["B"]).strip()
                    desc = str(r["C"]).strip()
                    loc = str(r["D"]).strip()
                    cost = float(bom_lookup.get(comp, 0.0))
                    ev_rows.append((file_hash, comp, desc, loc, board, mo, dt_iso, machine, cost, cost))
            except Exception:
                continue

        if ev_rows:
            conn.executemany(
                """
                INSERT INTO events(file_hash, component, description, location, board_name, mo, file_dt, machine, unit_cost, cost)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                ev_rows
            )
            inserted_events += len(ev_rows)

        conn.commit()

    return inserted_files, inserted_events, skipped

# =========================================================
# FILTER SQL
# =========================================================
def _build_where(dt_start, dt_end, boards, mos, machines, components=None):
    where = []
    params = []

    if dt_start is not None:
        where.append("(file_dt IS NOT NULL AND file_dt >= ?)")
        params.append(dt_start.isoformat(sep=" "))
    if dt_end is not None:
        where.append("(file_dt IS NOT NULL AND file_dt <= ?)")
        params.append(dt_end.isoformat(sep=" "))

    if boards:
        where.append("board_name IN (%s)" % ",".join(["?"] * len(boards)))
        params.extend(boards)
    if mos:
        where.append("mo IN (%s)" % ",".join(["?"] * len(mos)))
        params.extend(mos)
    if machines:
        where.append("machine IN (%s)" % ",".join(["?"] * len(machines)))
        params.extend(machines)
    if components is not None and components:
        where.append("component IN (%s)" % ",".join(["?"] * len(components)))
        params.extend(components)

    return where, params

# =========================================================
# BOARD COUNT (LINE2 = logs/3)
# =========================================================
def estimate_total_boards(conn, dt_start, dt_end, boards, mos, machines) -> float:
    where, params = _build_where(dt_start, dt_end, boards, mos, machines, components=None)
    sql = "SELECT machine, COUNT(*) AS n FROM logs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " GROUP BY machine"
    df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return 0.0

    line1_logs = float(df.loc[df["machine"].isin(LINE1_MACHINES), "n"].sum())
    line2_logs = float(df.loc[df["machine"].isin(LINE2_MACHINES), "n"].sum())

    # treat unknown machines as 1-to-1 unless filtered explicitly
    other_logs = float(df.loc[~df["machine"].isin(LINE1_MACHINES.union(LINE2_MACHINES)), "n"].sum())

    return line1_logs + (line2_logs / LINE2_DIVISOR) + other_logs

def estimate_boards_by_board(conn, dt_start, dt_end, boards, mos, machines, boards_limit=None) -> pd.DataFrame:
    where, params = _build_where(dt_start, dt_end, boards, mos, machines, components=None)
    sql = "SELECT board_name AS Board, machine AS Machine, COUNT(*) AS n FROM logs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if boards_limit:
        if where:
            sql += " AND "
        else:
            sql += " WHERE "
        sql += "board_name IN (%s)" % ",".join(["?"] * len(boards_limit))
        params.extend(boards_limit)
    sql += " GROUP BY board_name, machine"
    df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["Board", "BoardsRun"])

    def weight(m):
        if m in LINE1_MACHINES:
            return 1.0
        if m in LINE2_MACHINES:
            return 1.0 / LINE2_DIVISOR
        return 1.0

    df["BoardsEquivalent"] = df.apply(lambda r: float(r["n"]) * weight(r["Machine"]), axis=1)
    out = df.groupby("Board")["BoardsEquivalent"].sum().reset_index().rename(columns={"BoardsEquivalent": "BoardsRun"})
    return out

def machine_log_breakdown(conn, dt_start, dt_end, boards, mos, machines) -> pd.DataFrame:
    where, params = _build_where(dt_start, dt_end, boards, mos, machines, components=None)
    sql = "SELECT machine AS Machine, COUNT(*) AS LogFiles FROM logs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " GROUP BY machine ORDER BY LogFiles DESC"
    return pd.read_sql_query(sql, conn, params=params)

# =========================================================
# EVENTS QUERY
# =========================================================
def query_events(conn, dt_start, dt_end, boards, mos, machines, components, bom_lookup):
    where, params = _build_where(dt_start, dt_end, boards, mos, machines, components)
    sql = """
    SELECT
      component AS Component,
      description AS Description,
      location AS Location,
      board_name AS Board,
      mo AS MO,
      file_dt AS FileDateTime,
      machine AS Machine
    FROM events
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY file_dt DESC"
    df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        df["UnitCost"] = []
        df["Cost"] = []
        return df

    df["UnitCost"] = df["Component"].map(lambda c: float(bom_lookup.get(str(c).strip(), 0.0)))
    df["Cost"] = df["UnitCost"]
    return df

# =========================================================
# DERIVED VIEWS
# =========================================================
def make_summary(events_df):
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Description","Machine","Spits","UnitCost","TotalCost"])
    return (
        events_df.groupby("Component")
        .agg(
            Description=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            Machine=("Machine", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            Spits=("Component", "count"),
            UnitCost=("UnitCost", "max"),
            TotalCost=("Cost", "sum"),
        )
        .reset_index()
        .sort_values("TotalCost", ascending=False)
    )

def make_repeated_locations(events_df):
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Location","Board","Machine","Spits","TotalCost"])
    return (
        events_df.groupby(["Component","Location","Board","Machine"])
        .agg(Spits=("Component","count"), TotalCost=("Cost","sum"))
        .reset_index()
        .query("Spits > 1")
        .sort_values(["TotalCost","Spits"], ascending=[False, False])
    )

def make_missing_costs(events_df):
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Spits (cost=0)"])
    return (
        events_df.loc[events_df["UnitCost"] == 0.0, "Component"]
        .value_counts()
        .reset_index()
        .rename(columns={"index":"Component", "Component":"Spits (cost=0)"})
    )

def make_board_loss(events_df, board_value):
    if events_df.empty:
        return pd.DataFrame(columns=["Board","TotalCost","Loss % of Board Value (period)"])
    out = events_df.groupby("Board")["Cost"].sum().reset_index(name="TotalCost")
    out["Loss % of Board Value (period)"] = (out["TotalCost"] / board_value * 100) if board_value else np.nan
    return out.sort_values("Loss % of Board Value (period)", ascending=False)

def make_board_loss_components(events_df, boards_run_by_board, board_value):
    if events_df.empty:
        return pd.DataFrame(columns=[
            "Board","BoardsRun","Component","Description","Spits","UnitCost","Cost",
            "Period % of Board Value","Avg % of Board Value per Board","% of Board’s Total Loss"
        ])

    comp_loss = (
        events_df.groupby(["Board","Component"])
        .agg(
            Description=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            Spits=("Component","count"),
            UnitCost=("UnitCost","max"),
            Cost=("Cost","sum")
        )
        .reset_index()
    )

    comp_loss = comp_loss.merge(boards_run_by_board, on="Board", how="left")
    comp_loss["BoardsRun"] = comp_loss["BoardsRun"].fillna(0.0)

    comp_loss["Period % of Board Value"] = ((comp_loss["Cost"] / board_value) * 100) if board_value else np.nan
    comp_loss["Avg % of Board Value per Board"] = np.where(
        (board_value > 0) & (comp_loss["BoardsRun"] > 0),
        (comp_loss["Cost"] / (board_value * comp_loss["BoardsRun"])) * 100,
        np.nan
    )

    board_total = comp_loss.groupby("Board")["Cost"].transform("sum")
    comp_loss["% of Board’s Total Loss"] = np.where(board_total > 0, (comp_loss["Cost"] / board_total) * 100, 0.0)

    return comp_loss.sort_values(["Board", "Cost"], ascending=[True, False])

# =========================================================
# APP UI
# =========================================================
conn = db_connect()
db_init(conn)

st.title("SMT Spit Analytics (Full)")

# ---- View uploaded files and BOMs
with st.expander("📚 View uploaded Logs and Master BOM versions"):
    cA, cB = st.columns(2)
    with cA:
        st.subheader("BOM versions uploaded")
        boms_df = list_boms(conn)
        if boms_df.empty:
            st.info("No BOM versions stored yet.")
        else:
            st.dataframe(boms_df, use_container_width=True, height=240)

    with cB:
        st.subheader("Log files ingested (latest 200)")
        logs_df = pd.read_sql_query(
            "SELECT filename, file_dt, machine, board_name, mo, ingested_at FROM logs ORDER BY ingested_at DESC LIMIT 200",
            conn
        )
        if logs_df.empty:
            st.info("No logs ingested yet.")
        else:
            st.dataframe(logs_df, use_container_width=True, height=240)

with st.expander("📦 Data Store (upload once, reused later)", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Master BOM (stored as new version)")
        bom_up = st.file_uploader(
            "Master BOM Excel: component in column A, cost in column J (all sheets read).",
            type=["xls", "xlsx"],
            key="bom_up"
        )
        bom_name = st.text_input("BOM name/label (e.g. Master BOM Jan-2026)", value="", key="bom_name")
        if st.button("Save Master BOM to Database", type="secondary"):
            if not bom_up:
                st.warning("Upload a Master BOM file first.")
            else:
                label = bom_name.strip() if bom_name.strip() else bom_up.name
                n = ingest_master_bom(conn, bom_up.getvalue(), label)
                if n == 0:
                    st.error("No BOM items were loaded. Check: component in column A and cost in column J.")
                else:
                    st.success(f"Master BOM stored. Components loaded: {n}")

    with col2:
        st.subheader("Ingest Log Files")
        logs_up = st.file_uploader(
            "Upload SMT log files (CSV/XLS/XLSX).",
            type=["csv", "xls", "xlsx"],
            accept_multiple_files=True,
            key="logs_up"
        )
        if st.button("Ingest Logs into Database", type="secondary"):
            if not logs_up:
                st.warning("Upload log files first.")
            else:
                ins_files, ins_events, skipped = ingest_logs(conn, logs_up)
                st.success(f"Ingest complete. New files: {ins_files} | New spit events: {ins_events}")
                if skipped:
                    st.dataframe(pd.DataFrame(skipped, columns=["File", "Reason"]), use_container_width=True)

# ---- Filters
st.subheader("🔎 Filters (combine as needed)")

today = date.today()
default_start = datetime.combine(today, time(0, 0, 0))
default_end = datetime.now().replace(microsecond=0)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    start_date = st.date_input("Start date", value=default_start.date(), key="start_date")
    start_time = st.time_input("Start time", value=default_start.time(), key="start_time")
with c2:
    end_date = st.date_input("End date", value=default_end.date(), key="end_date")
    end_time = st.time_input("End time", value=default_end.time(), key="end_time")
with c3:
    board_value = st.number_input("Board value (for % loss)", min_value=0.0, value=0.0, step=1.0, key="board_value")
with c4:
    st.write("")
    st.write("")
    run_query = st.button("Run Query", type="primary")

dt_start = datetime.combine(start_date, start_time)
dt_end = datetime.combine(end_date, end_time)

# Filter option lists
boards_all = [r[0] for r in conn.execute(
    "SELECT DISTINCT board_name FROM logs WHERE board_name IS NOT NULL AND board_name <> '' ORDER BY board_name"
).fetchall()]
mos_all = [r[0] for r in conn.execute(
    "SELECT DISTINCT mo FROM logs WHERE mo IS NOT NULL AND mo <> '' ORDER BY mo"
).fetchall()]
machines_all = [r[0] for r in conn.execute(
    "SELECT DISTINCT machine FROM logs WHERE machine IS NOT NULL AND machine <> '' ORDER BY machine"
).fetchall()]
components_all = [r[0] for r in conn.execute(
    "SELECT DISTINCT component FROM events WHERE component IS NOT NULL AND component <> '' ORDER BY component"
).fetchall()]

# BOM selector
boms_df = list_boms(conn)
bom_labels, bom_id_by_label = [], {}
if not boms_df.empty:
    for _, r in boms_df.iterrows():
        label = f'{int(r["bom_id"])} | {r["bom_name"]} | {r["uploaded_at"]}'
        bom_labels.append(label)
        bom_id_by_label[label] = int(r["bom_id"])

f0, f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1, 1])
with f0:
    selected_boms_labels = st.multiselect(
        "Master BOM version(s) to use for analysis (blank = latest version per component)",
        options=bom_labels,
        default=[],
        key="selected_boms"
    )
with f1:
    boards_sel = st.multiselect("Board Name", boards_all, default=[], key="boards_sel")
with f2:
    mos_sel = st.multiselect("MO", mos_all, default=[], key="mos_sel")
with f3:
    machines_sel = st.multiselect("Machine", machines_all, default=[], key="machines_sel")
with f4:
    components_sel = st.multiselect("Component (optional)", components_all, default=[], key="components_sel")

selected_bom_ids = [bom_id_by_label[x] for x in selected_boms_labels] if selected_boms_labels else []

# Persist results
if "has_results" not in st.session_state:
    st.session_state.has_results = False

if run_query:
    bom_lookup = get_bom_lookup(conn, selected_bom_ids if selected_bom_ids else None)

    events_df = query_events(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel, components_sel, bom_lookup=bom_lookup)

    total_boards_est = estimate_total_boards(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel)

    boards_in_results = sorted([b for b in events_df["Board"].dropna().astype(str).unique()])
    boards_run_by_board = estimate_boards_by_board(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel, boards_limit=boards_in_results)

    m_breakdown = machine_log_breakdown(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel)

    st.session_state.events_df = events_df
    st.session_state.total_boards_est = float(total_boards_est)
    st.session_state.boards_run_by_board = boards_run_by_board
    st.session_state.machine_breakdown = m_breakdown
    st.session_state.has_results = True

# View selector
view = st.selectbox(
    "Select View",
    [
        "Summary",
        "Spit Events",
        "Pareto (Cost)",
        "Repeated Locations",
        "Yield Loss",
        "Missing BOM Costs",
        "Board Loss %",
        "Board Loss Components"
    ],
    index=0
)

if not st.session_state.has_results:
    st.info("Upload BOM + ingest logs (once), then set filters and click **Run Query**.")
    st.stop()

events_df = st.session_state.events_df.copy()
total_boards_est = float(st.session_state.total_boards_est)
boards_run_by_board = st.session_state.boards_run_by_board.copy()
machine_breakdown_df = st.session_state.machine_breakdown.copy()

# Views
if view == "Summary":
    st.dataframe(make_summary(events_df), use_container_width=True)

elif view == "Spit Events":
    st.dataframe(events_df, use_container_width=True)

elif view == "Pareto (Cost)":
    if events_df.empty:
        st.info("No events in this selection.")
    else:
        st.bar_chart(events_df.groupby("Component")["Cost"].sum().sort_values(ascending=False))

elif view == "Repeated Locations":
    st.dataframe(make_repeated_locations(events_df), use_container_width=True)

elif view == "Yield Loss":
    total_cost = float(events_df["Cost"].sum()) if not events_df.empty else 0.0
    st.metric("Estimated Boards Run", round(total_boards_est, 2))
    st.metric("Total Cost Loss", round(total_cost, 2))
    st.metric("Avg Cost / Board (Estimated)", round(total_cost / total_boards_est, 2) if total_boards_est else 0.0)

    st.subheader("Machine log breakdown (for board estimation)")
    st.dataframe(machine_breakdown_df, use_container_width=True)

    st.subheader("Boards Run breakdown by board (Estimated)")
    if boards_run_by_board.empty:
        st.info("No board breakdown for this selection.")
    else:
        st.dataframe(boards_run_by_board.sort_values("BoardsRun", ascending=False), use_container_width=True)

elif view == "Missing BOM Costs":
    st.dataframe(make_missing_costs(events_df), use_container_width=True)

elif view == "Board Loss %":
    st.dataframe(make_board_loss(events_df, board_value), use_container_width=True)

elif view == "Board Loss Components":
    out = make_board_loss_components(events_df, boards_run_by_board, board_value)
    st.dataframe(out, use_container_width=True)

