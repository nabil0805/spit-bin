import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import sqlite3
import hashlib
from datetime import datetime, date, time, timedelta
from pandas.errors import EmptyDataError

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="SMT Spit Analytics (DB Mode)", layout="wide")

DB_PATH = "smt_spit.db"
REJECT_CODES = {2, 3, 4, 5, 7}

# Filename example:
# 20260106091251-IIN2-053-2.csv
# YYYYMMDDHHMMSS-MACHINE(.ext)
FILENAME_RE = re.compile(r"^(?P<dt>\d{14})-(?P<machine>.+?)(?:\.[A-Za-z0-9]+)?$")

# =========================================================
# DB HELPERS
# =========================================================
def db_connect():
    # check_same_thread False is safe for Streamlit single-process usage
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS bom (
        component TEXT PRIMARY KEY,
        unit_cost REAL NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        file_hash TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_dt TEXT,          -- ISO datetime
        machine TEXT,
        board_name TEXT,
        mo TEXT,
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
    s = re.sub(r"[^\d\.\-]", "", s)  # strip currency symbols etc.
    try:
        return float(s)
    except:
        return np.nan

def extract_component_from_bom_cell(text):
    if pd.isna(text):
        return None
    m = re.search(r"\[(.*?)\]", str(text))
    return m.group(1).strip() if m else None

def parse_dt_machine_from_filename(filename: str):
    """
    Returns (dt_iso, machine) from filename like:
    20260106091251-IIN2-053-2.csv
    """
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

def read_header_board_mo(file_bytes: bytes):
    """
    Reads row1 for:
      Board Name: B1 (col 1)
      MO: D1 (col 3)
    (We rely on filename for dt + machine per your instruction.)
    """
    try:
        try:
            header = pd.read_excel(io.BytesIO(file_bytes), nrows=1, header=None)
        except Exception:
            header = safe_read_csv(file_bytes, nrows=1, header=None)
    except Exception:
        return None, None

    try:
        board = str(header.iloc[0, 1]).strip()
        mo = str(header.iloc[0, 3]).strip()
        return board, mo
    except Exception:
        return None, None

def read_body_df(file_bytes: bytes, filename: str):
    """
    Reads data skipping first 2 rows, keeping first 12 cols.
    """
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
# BOM INGEST (STORE IN DB)
# =========================================================
def ingest_bom(conn: sqlite3.Connection, bom_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(bom_bytes))
    rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        for i in range(len(df)):
            comp = extract_component_from_bom_cell(df.iat[i, 0]) if df.shape[1] > 0 else None
            if not comp:
                continue
            raw_cost = df.iat[i, 10] if df.shape[1] > 10 else None
            cost = parse_cost(raw_cost)
            if pd.isna(cost):
                continue
            rows.append((str(comp).strip(), float(cost)))

    if not rows:
        return 0

    now = datetime.now().isoformat(sep=" ")
    # Upsert into bom
    conn.executemany(
        """
        INSERT INTO bom(component, unit_cost, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(component) DO UPDATE SET
            unit_cost=excluded.unit_cost,
            updated_at=excluded.updated_at
        """,
        [(c, cost, now) for c, cost in rows]
    )
    conn.commit()
    return len(set([r[0] for r in rows]))

def get_bom_lookup(conn: sqlite3.Connection) -> dict:
    cur = conn.execute("SELECT component, unit_cost FROM bom")
    return {r[0]: float(r[1]) for r in cur.fetchall()}

# =========================================================
# LOG INGEST (STORE IN DB)
# =========================================================
def ingest_logs(conn: sqlite3.Connection, uploads):
    bom_lookup = get_bom_lookup(conn)

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

        # Already ingested?
        exists = conn.execute("SELECT 1 FROM logs WHERE file_hash=?", (file_hash,)).fetchone()
        if exists:
            skipped.append((filename, "Already ingested (same hash)"))
            continue

        dt_iso, machine = parse_dt_machine_from_filename(filename)
        board, mo = read_header_board_mo(b)

        df = read_body_df(b, filename)
        if df is None:
            skipped.append((filename, "No readable data after skiprows=2"))
            continue

        # Insert log
        conn.execute(
            "INSERT INTO logs(file_hash, filename, file_dt, machine, board_name, mo, ingested_at) VALUES (?,?,?,?,?,?,?)",
            (file_hash, filename, dt_iso, machine, board, mo, datetime.now().isoformat(sep=" "))
        )
        inserted_files += 1

        # Insert events
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
# QUERY (FILTERS)
# =========================================================
def query_events(conn: sqlite3.Connection,
                 dt_start: datetime | None,
                 dt_end: datetime | None,
                 boards: list[str],
                 mos: list[str],
                 machines: list[str],
                 components: list[str]) -> pd.DataFrame:

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
    if components:
        where.append("component IN (%s)" % ",".join(["?"] * len(components)))
        params.extend(components)

    sql = """
    SELECT
      component AS Component,
      description AS Description,
      location AS Location,
      board_name AS Board,
      mo AS MO,
      file_dt AS FileDateTime,
      machine AS Machine,
      unit_cost AS UnitCost,
      cost AS Cost
    FROM events
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY file_dt DESC"

    df = pd.read_sql_query(sql, conn, params=params)
    return df

def query_file_counts(conn: sqlite3.Connection,
                      dt_start: datetime | None,
                      dt_end: datetime | None,
                      boards: list[str],
                      mos: list[str],
                      machines: list[str]) -> int:
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

    sql = "SELECT COUNT(*) FROM logs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return int(conn.execute(sql, params).fetchone()[0])

# =========================================================
# DERIVED VIEWS
# =========================================================
def make_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Description","Machine","Spits","UnitCost","TotalCost"])
    return (
        events_df.groupby("Component")
        .agg(
            Description=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            Machine=("Machine", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            Spits=("Component", "count"),
            UnitCost=("UnitCost", "max"),
            TotalCost=("Cost", "sum")
        )
        .reset_index()
        .sort_values("TotalCost", ascending=False)
    )

def make_repeated_locations(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Location","Board","Machine","Spits","TotalCost"])
    return (
        events_df.groupby(["Component","Location","Board","Machine"])
        .agg(Spits=("Component","count"), TotalCost=("Cost","sum"))
        .reset_index()
        .query("Spits > 1")
        .sort_values(["TotalCost","Spits"], ascending=[False, False])
    )

def make_missing_costs(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["Component","Spits (cost=0)"])
    return (
        events_df.loc[events_df["UnitCost"] == 0.0, "Component"]
        .value_counts()
        .reset_index()
        .rename(columns={"index":"Component", "Component":"Spits (cost=0)"})
    )

# =========================================================
# UI
# =========================================================
conn = db_connect()
db_init(conn)

st.title("SMT Spit Analytics (Database Mode)")

# ---------- Admin: BOM + Log ingest ----------
with st.expander("📦 Data Store (upload once, reuse later)", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload / Update BOM")
        bom_up = st.file_uploader("BOM Excel (all sheets read)", type=["xls", "xlsx"], key="bom_up")
        if st.button("Save BOM to Database", type="secondary"):
            if not bom_up:
                st.warning("Upload a BOM file first.")
            else:
                n = ingest_bom(conn, bom_up.getvalue())
                st.success(f"BOM stored/updated. Components loaded: {n}")

    with col2:
        st.subheader("Ingest Log Files")
        logs_up = st.file_uploader("Upload log files (CSV/XLS/XLSX)", type=["csv","xls","xlsx"], accept_multiple_files=True, key="logs_up")
        if st.button("Ingest Logs into Database", type="secondary"):
            if not logs_up:
                st.warning("Upload log files first.")
            else:
                ins_files, ins_events, skipped = ingest_logs(conn, logs_up)
                st.success(f"Ingest complete. New files: {ins_files} | New spit events: {ins_events}")
                if skipped:
                    st.dataframe(pd.DataFrame(skipped, columns=["File", "Reason"]), use_container_width=True)

# ---------- Filter panel ----------
st.subheader("🔎 Filters (combine as needed)")

# Default date/time range: today 00:00 to now
today = date.today()
default_start = datetime.combine(today, time(0, 0, 0))
default_end = datetime.now()

c1, c2, c3, c4 = st.columns([1,1,1,1])

with c1:
    start_date = st.date_input("Start date", value=default_start.date(), key="start_date")
    start_time = st.time_input("Start time", value=default_start.time(), key="start_time")
with c2:
    end_date = st.date_input("End date", value=default_end.date(), key="end_date")
    end_time = st.time_input("End time", value=default_end.time().replace(microsecond=0), key="end_time")
with c3:
    board_value = st.number_input("Board value (for % loss)", min_value=0.0, value=0.0, step=1.0, key="board_value")
with c4:
    st.write("")
    st.write("")
    run_query = st.button("Run Query", type="primary")

dt_start = datetime.combine(start_date, start_time)
dt_end = datetime.combine(end_date, end_time)

# Populate filter options from DB (fast)
boards_all = [r[0] for r in conn.execute("SELECT DISTINCT board_name FROM logs WHERE board_name IS NOT NULL AND board_name <> '' ORDER BY board_name").fetchall()]
mos_all = [r[0] for r in conn.execute("SELECT DISTINCT mo FROM logs WHERE mo IS NOT NULL AND mo <> '' ORDER BY mo").fetchall()]
machines_all = [r[0] for r in conn.execute("SELECT DISTINCT machine FROM logs WHERE machine IS NOT NULL AND machine <> '' ORDER BY machine").fetchall()]
components_all = [r[0] for r in conn.execute("SELECT DISTINCT component FROM events WHERE component IS NOT NULL AND component <> '' ORDER BY component").fetchall()]

f1, f2, f3, f4 = st.columns(4)
with f1:
    boards_sel = st.multiselect("Board Name", boards_all, default=[], key="boards_sel")
with f2:
    mos_sel = st.multiselect("MO", mos_all, default=[], key="mos_sel")
with f3:
    machines_sel = st.multiselect("Machine", machines_all, default=[], key="machines_sel")
with f4:
    components_sel = st.multiselect("Component (optional)", components_all, default=[], key="components_sel")

# ---------- Keep results in session_state so dropdown works ----------
if "has_results" not in st.session_state:
    st.session_state.has_results = False

if run_query:
    events_df = query_events(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel, components_sel)
    boards_count = query_file_counts(conn, dt_start, dt_end, boards_sel, mos_sel, machines_sel)

    st.session_state.events_df = events_df
    st.session_state.boards_count = boards_count
    st.session_state.has_results = True

# ---------- View selector ----------
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
    st.info("Ingest logs + BOM once (above), then set filters and click **Run Query**.")
    st.stop()

events_df = st.session_state.events_df.copy()
boards_count = int(st.session_state.boards_count)

# ---------- VIEWS ----------
if view == "Summary":
    summary = make_summary(events_df)
    st.dataframe(summary, use_container_width=True)

elif view == "Spit Events":
    st.dataframe(events_df, use_container_width=True)

elif view == "Pareto (Cost)":
    if events_df.empty:
        st.info("No events in this selection.")
    else:
        pareto = events_df.groupby("Component")["Cost"].sum().sort_values(ascending=False)
        st.bar_chart(pareto)

elif view == "Repeated Locations":
    rep = make_repeated_locations(events_df)
    st.dataframe(rep, use_container_width=True)

elif view == "Yield Loss":
    total_cost = float(events_df["Cost"].sum()) if not events_df.empty else 0.0
    st.metric("Boards Run (files)", boards_count)
    st.metric("Total Cost Loss", round(total_cost, 2))
    st.metric("Avg Cost / Board", round(total_cost / boards_count, 2) if boards_count else 0.0)

elif view == "Missing BOM Costs":
    missing = make_missing_costs(events_df)
    st.dataframe(missing, use_container_width=True)

elif view == "Board Loss %":
    if events_df.empty:
        st.info("No events in this selection.")
    else:
        board_loss = events_df.groupby("Board")["Cost"].sum().reset_index()
        board_loss["Loss % of Board Value"] = (board_loss["Cost"] / board_value * 100) if board_value else np.nan
        st.dataframe(board_loss.sort_values("Loss % of Board Value", ascending=False), use_container_width=True)

elif view == "Board Loss Components":
    if events_df.empty:
        st.info("No events in this selection.")
    else:
        comp_loss = (
            events_df.groupby(["Board", "Component"])
            .agg(Spits=("Component", "count"), Cost=("Cost", "sum"))
            .reset_index()
        )
        comp_loss["Loss % of Board Value"] = (comp_loss["Cost"] / board_value * 100) if board_value else np.nan
        comp_loss = comp_loss.sort_values(["Board", "Cost"], ascending=[True, False])
        st.dataframe(comp_loss, use_container_width=True)


