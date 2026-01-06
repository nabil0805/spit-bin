import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt

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
    if isinstance(v, (int, float)):
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
                records.append({"Component": comp, "Unit Cost": cost})

    bom_df = pd.DataFrame(records)
    return (
        bom_df
        .drop_duplicates("Component", keep="last")
        .set_index("Component")["Unit Cost"]
        .to_dict()
    )

# ---------------------------------------------------------
# READ LOG FILES
# ---------------------------------------------------------
def read_logs(log_files, cost_lookup):
    events = []
    files = []

    for f in log_files:
        name = f.name
        data = f.getvalue()

        # Header
        try:
            header = pd.read_excel(io.BytesIO(data), nrows=1, header=None)
        except:
            header = pd.read_csv(io.BytesIO(data), nrows=1, header=None, encoding="latin-1")

        board = str(header.iloc[0, 1]).strip()
        mo = str(header.iloc[0, 3]).strip()
        date = str(header.iloc[0, 8]).strip()
        machine = str(header.iloc[0, 11]).strip()

        # Body
        try:
            df = pd.read_excel(io.BytesIO(data), skiprows=2, header=None, usecols=range(12))
        except:
            df = pd.read_csv(io.BytesIO(data), skiprows=2, header=None, encoding="latin-1", usecols=range(12))

        df.columns = list("ABCDEFGHIJKL")
        spit_count = 0

        for _, r in df.iterrows():
            try:
                if int(r["L"]) in REJECT_CODES:
                    c = str(r["B"]).strip()
                    cost = cost_lookup.get(c, 0.0)
                    spit_count += 1
                    events.append({
                        "Component": c,
                        "Description": str(r["C"]),
                        "Location": str(r["D"]),
                        "Board": board,
                        "MO": mo,
                        "Date": date,
                        "Machine": machine,
                        "Unit Cost": cost,
                        "Cost": cost
                    })
            except:
                pass

        files.append({"Board": board, "MO": mo, "Spits": spit_count})

    return pd.DataFrame(events), pd.DataFrame(files)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("SMT Spit Analytics Dashboard")

st.subheader("Inputs")
log_files = st.file_uploader(
    "Upload SMT log files (CSV / XLS / XLSX)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

bom_file = st.file_uploader(
    "Upload BOM Excel (all sheets read, cost from column K)",
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
    events_df, files_df = read_logs(log_files, cost_lookup)

    st.success("Analysis complete")

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

    # -----------------------------------------------------
    if view == "Summary":
        summary = (
            events_df.groupby("Component")
            .agg(
                Description=("Description", "first"),
                Machine=("Machine", "first"),
                Spits=("Component", "count"),
                Unit_Cost=("Unit Cost", "max"),
                Total_Cost=("Cost", "sum")
            )
            .sort_values("Total_Cost", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)

    # -----------------------------------------------------
    elif view == "Spit Events":
        st.dataframe(events_df, use_container_width=True)

    # -----------------------------------------------------
    elif view == "Pareto (Cost)":
        pareto = events_df.groupby("Component")["Cost"].sum().sort_values(ascending=False)
        st.bar_chart(pareto)

    # -----------------------------------------------------
    elif view == "Repeated Locations":
        rep = (
            events_df
            .groupby(["Component", "Location", "Machine"])
            .size()
            .reset_index(name="Spits")
            .query("Spits > 1")
        )
        st.dataframe(rep, use_container_width=True)

    # -----------------------------------------------------
    elif view == "Yield Loss":
        total_boards = len(log_files)
        total_cost = events_df["Cost"].sum()
        st.metric("Total Boards Run", total_boards)
        st.metric("Total Cost Loss", round(total_cost, 2))
        st.metric("Avg Cost / Board", round(total_cost / total_boards, 2) if total_boards else 0)

    # -----------------------------------------------------
    elif view == "Missing BOM Costs":
        missing = events_df[events_df["Unit Cost"] == 0]["Component"].value_counts()
        st.dataframe(missing)

    # -----------------------------------------------------
    elif view == "Board Loss %":
        board_loss = events_df.groupby("Board")["Cost"].sum().reset_index()
        board_loss["Loss % of Board Value"] = (
            board_loss["Cost"] / board_value * 100
            if board_value else 0
        )
        st.dataframe(board_loss, use_container_width=True)

    # -----------------------------------------------------
    elif view == "Board Loss Components":
        comp_loss = (
            events_df
            .groupby(["Board", "Component"])
            .agg(
                Spits=("Component", "count"),
                Cost=("Cost", "sum")
            )
            .reset_index()
        )
        comp_loss["Loss % of Board Value"] = (
            comp_loss["Cost"] / board_value * 100
            if board_value else 0
        )
        st.dataframe(comp_loss, use_container_width=True)

else:
    st.info("Upload log files + BOM, enter board value, then click Run Analysis.")
