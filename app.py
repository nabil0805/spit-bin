import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from pandas.errors import EmptyDataError

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="SMT Spit Analytics", layout="wide")
REJECT_CODES = {2, 3, 4, 5, 7}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
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

def extract_component(text):
    if pd.isna(text):
        return None
    m = re.search(r"\[(.*?)\]", str(text))
    return m.group(1).strip() if m else None

def safe_read_csv(bytes_data, **kwargs):
    """
    Robust CSV read:
    - try utf-8 then latin-1
    - lets EmptyDataError bubble up so caller can handle it
    """
    try:
        return pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1", **kwargs)

# ---------------------------------------------------------
# LOAD BOM (ALL SHEETS)
# ---------------------------------------------------------
def load_bom(bom_bytes):
    xls = pd.ExcelFile(io.BytesIO(bom_bytes))
    records = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        for i in range(len(df)):
            comp = extract_component(df.iat[i, 0])
            if not comp:
                continue
            cost = parse_cost(df.iat[i, 10]) if df.shape[1] > 10 else np.nan
            if not pd.isna(cost):
                records.append({"Component": str(comp).strip(), "Unit Cost": float(cost)})

    if not records:
        return {}

    bom_df = pd.DataFrame(records)
    return (
        bom_df.drop_duplicates("Component", keep="last")
        .set_index("Component")["Unit Cost"]
        .to_dict()
    )

# ---------------------------------------------------------
# READ LOG FILES (HARDENED AGAINST EMPTY/ODD FILES)
# ---------------------------------------------------------
def read_logs(log_files, cost_lookup):
    events = []
    files = []
    skipped = []

    for up in log_files:
        name = up.name
        data = up.getvalue()

        if data is None or len(data) == 0:
            skipped.append((name, "Empty file (0 bytes)"))
            continue

        # -------- HEADER (row 1) --------
        try:
            try:
                header = pd.read_excel(io.BytesIO(data), nrows=1, header=None)
            except Exception:
                header = safe_read_csv(data, nrows=1, header=None)
        except EmptyDataError:
            skipped.append((name, "Empty header"))
            continue
        except Exception as e:
            skipped.append((name, f"Header read failed: {type(e).__name__}"))
            continue

        # validate header has required cells (B1, D1, I1, L1)
        try:
            board = str(header.iloc[0, 1]).strip()     # B1
            mo = str(header.iloc[0, 3]).strip()        # D1
            date = str(header.iloc[0, 8]).strip()      # I1
            machine = str(header.iloc[0, 11]).strip()  # L1
        except Exception:
            skipped.append((name, "Header missing required cells (B1/D1/I1/L1)"))
            continue

        # -------- BODY (skip first 2 rows, first 12 cols) --------
        try:
            try:
                df = pd.read_excel(io.BytesIO(data), skiprows=2, header=None, usecols=range(12))
            except Exception:
                df = safe_read_csv(data, skiprows=2, header=None, usecols=range(12))
        except EmptyDataError:
            skipped.append((name, "No data after skipping first 2 rows"))
            continue
        except Exception as e:
            skipped.append((name, f"Data read failed: {type(e).__name__}"))
            continue

        if df is None or df.empty:
            skipped.append((name, "Dataframe empty after read"))
            continue

        # ensure 12 cols exist even if pandas returned fewer for some reason
        try:
            df = df.iloc[:, :12]
        except Exception:
            skipped.append((name, "Could not slice first 12 columns"))
            continue

        if df.shape[1] < 12:
            skipped.append((name, f"Only {df.shape[1]} columns found (need 12)"))
            continue

        df.columns = list("ABCDEFGHIJKL")

        spit_count = 0
        for _, r in df.iterrows():
            try:
                if int(r["L"]) in REJECT_CODES:
                    comp = str(r["B"]).strip()
                    cost = float(cost_lookup.get(comp, 0.0))
                    spit_count += 1
                    events.append({
                        "Component": comp,
                        "Description": str(r["C"]).strip(),
                        "Location": str(r["D"]).strip(),
                        "Board": board,
                        "MO": mo,
                        "Date": date,
                        "Machine": machine,
                        "Unit Cost": cost,
                        "Cost": cost
                    })
            except Exception:
                # ignore bad rows
                continue

        files.append({"File": name, "Board": board, "MO": mo, "Spits": spit_count})

    return pd.DataFrame(events), pd.DataFrame(files), skipped

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("SMT Spit Analytics Dashboard")

st.subheader("Inputs")
log_files = st.file_uploader(
    "Upload SMT log files (CSV / XLS / XLSX) — each file = one board",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

bom_file = st.file_uploader(
    "Upload BOM Excel (all sheets read; Component in [ ] in column A, Cost in column K)",
    type=["xls", "xlsx"]
)

board_value = st.number_input(
    "Board value (used for % loss calculations)",
    min_value=0.0,
    value=0.0
)

run = st.button("Run Analysis", type="primary")

if run and log_files and bom_file:
    cost_lookup = load_bom(bom_file.getvalue())
    events_df, files_df, skipped = read_logs(log_files, cost_lookup)

    if skipped:
        with st.expander(f"Skipped files ({len(skipped)})", expanded=True):
            st.dataframe(pd.DataFrame(skipped, columns=["File", "Reason"]), use_container_width=True)

    if len(files_df) == 0:
        st.error("No valid log files were processed. Please check the skipped files list.")
        st.stop()

    st.success(f"Done. Boards processed: {len(files_df)} | Spit events: {len(events_df)}")

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
        ]
    )

    if view == "Summary":
        if len(events_df) == 0:
            st.info("No spit events found in the selected logs.")
        else:
            summary = (
                events_df.groupby("Component")
                .agg(
                    Description=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
                    Machine=("Machine", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
                    Spits=("Component", "count"),
                    Unit_Cost=("Unit Cost", "max"),
                    Total_Cost=("Cost", "sum")
                )
                .reset_index()
                .sort_values("Total_Cost", ascending=False)
            )
            st.dataframe(summary, use_container_width=True)

    elif view == "Spit Events":
        st.dataframe(events_df, use_container_width=True)

    elif view == "Pareto (Cost)":
        if len(events_df) == 0:
            st.info("No spit events found.")
        else:
            pareto = events_df.groupby("Component")["Cost"].sum().sort_values(ascending=False)
            st.bar_chart(pareto)

    elif view == "Repeated Locations":
        if len(events_df) == 0:
            st.info("No spit events found.")
        else:
            rep = (
                events_df.groupby(["Component", "Location", "Machine"])
                .size()
                .reset_index(name="Spits")
                .query("Spits > 1")
                .sort_values("Spits", ascending=False)
            )
            st.dataframe(rep, use_container_width=True)

    elif view == "Yield Loss":
        total_boards = len(files_df)
        total_cost = float(events_df["Cost"].sum()) if len(events_df) else 0.0
        st.metric("Total Boards Run", total_boards)
        st.metric("Total Cost Loss", round(total_cost, 2))
        st.metric("Avg Cost / Board", round(total_cost / total_boards, 2) if total_boards else 0)

    elif view == "Missing BOM Costs":
        if len(events_df) == 0:
            st.info("No spit events found.")
        else:
            missing = (
                events_df.loc[events_df["Unit Cost"] == 0.0, "Component"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Component", "Component": "Spits (cost=0)"})
            )
            st.dataframe(missing, use_container_width=True)

    elif view == "Board Loss %":
        if len(events_df) == 0:
            st.info("No spit events found.")
        else:
            board_loss = events_df.groupby("Board")["Cost"].sum().reset_index()
            board_loss["Loss % of Board Value"] = (
                (board_loss["Cost"] / board_value) * 100 if board_value else np.nan
            )
            st.dataframe(board_loss, use_container_width=True)

    elif view == "Board Loss Components":
        if len(events_df) == 0:
            st.info("No spit events found.")
        else:
            comp_loss = (
                events_df.groupby(["Board", "Component"])
                .agg(Spits=("Component", "count"), Cost=("Cost", "sum"))
                .reset_index()
                .sort_values(["Board", "Cost"], ascending=[True, False])
            )
            comp_loss["Loss % of Board Value"] = (
                (comp_loss["Cost"] / board_value) * 100 if board_value else np.nan
            )
            st.dataframe(comp_loss, use_container_width=True)

else:
    st.info("Upload log files + BOM, enter board value, then click Run Analysis.")

