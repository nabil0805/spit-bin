import streamlit as st
import pandas as pd
import sqlite3
import io
import os
import re
from datetime import datetime
from openai import OpenAI

st.set_page_config(page_title="SMT Spit Analysis", layout="wide")

DB_FILE = "smt_database.db"


# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS logs(
        filename TEXT,
        board TEXT,
        mo TEXT,
        log_time TEXT,
        machine TEXT,
        component TEXT,
        description TEXT,
        location TEXT,
        feeder TEXT,
        slot TEXT,
        reject_code TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS bom(
        component TEXT PRIMARY KEY,
        cost REAL
    )
    """)

    conn.commit()


init_db()


# ---------------------------------------------------
# LOG INGESTION
# ---------------------------------------------------

def ingest_logs(files):

    conn = get_conn()
    cur = conn.cursor()

    rows = []

    for file in files:

        data = file.read()

        try:
            df = pd.read_csv(
                io.BytesIO(data),
                skiprows=2,
                header=None,
                encoding="latin-1",
                engine="python"
            )
        except:
            continue

        try:
            board = df.iloc[0,1]
            mo = df.iloc[0,3]
            log_time = df.iloc[0,8]
            machine = df.iloc[0,11]
        except:
            continue

        for _, r in df.iterrows():

            try:

                component = str(r[1])
                desc = str(r[2])
                location = str(r[3])
                feeder = str(r[7])
                slot = str(r[8])
                code = str(r[11])

                rows.append([
                    file.name,
                    board,
                    mo,
                    log_time,
                    machine,
                    component,
                    desc,
                    location,
                    feeder,
                    slot,
                    code
                ])

            except:
                continue

    cur.executemany("""
    INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()


# ---------------------------------------------------
# BOM INGESTION
# ---------------------------------------------------

def ingest_bom(file):

    df = pd.read_excel(file)

    rows = []

    for _, r in df.iterrows():

        component = str(r[0]).strip()

        try:
            cost = float(r[9])
        except:
            continue

        rows.append((component, cost))

    conn = get_conn()
    cur = conn.cursor()

    cur.executemany("""
    INSERT OR REPLACE INTO bom VALUES (?,?)
    """, rows)

    conn.commit()


# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

def load_data():

    conn = get_conn()

    logs = pd.read_sql("SELECT * FROM logs", conn)
    bom = pd.read_sql("SELECT * FROM bom", conn)

    return logs, bom


# ---------------------------------------------------
# FILTERS
# ---------------------------------------------------

def apply_filters(df, board, machine, mo, start, end):

    if board:
        df = df[df.board == board]

    if machine:
        df = df[df.machine == machine]

    if mo:
        df = df[df.mo == mo]

    if start:
        df = df[df.log_time >= str(start)]

    if end:
        df = df[df.log_time <= str(end)]

    return df


# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------

def build_summary(df, bom):

    rejects = df[df.reject_code != "0"]

    summary = rejects.groupby(["component","board"]).size().reset_index(name="spits")

    summary = summary.merge(bom, on="component", how="left")

    summary["total_cost"] = summary["spits"] * summary["cost"]

    # reject code breakdown

    code_map = (
        rejects
        .groupby(["component","board","reject_code"])
        .size()
        .reset_index(name="count")
    )

    def code_text(comp,board):

        rows = code_map[
            (code_map.component==comp) &
            (code_map.board==board)
        ]

        parts=[]

        for _,r in rows.iterrows():
            parts.append(f"{r['count']}x C{r['reject_code']}")

        return ", ".join(parts)

    summary["reject_codes"] = summary.apply(
        lambda r: code_text(r.component,r.board),
        axis=1
    )

    # ------------------------------------------------
    # TOTAL PLACEMENT COST (ALL SUCCESSFUL PARTS)
    # ------------------------------------------------

    success = df[df.reject_code=="0"]

    success = success.merge(bom,on="component",how="left")

    placement_cost = success.groupby("board")["cost"].sum()

    summary["total_placement_cost"] = summary["board"].map(placement_cost)

    summary["loss_percent"] = (
        summary["total_cost"] /
        summary["total_placement_cost"]
    ) * 100

    return summary


# ---------------------------------------------------
# SPIT EVENTS
# ---------------------------------------------------

def spit_events(df, bom):

    rejects = df[df.reject_code!="0"]

    rejects = rejects.merge(bom,on="component",how="left")

    return rejects


# ---------------------------------------------------
# PARETO
# ---------------------------------------------------

def pareto(df):

    rejects = df[df.reject_code!="0"]

    p = rejects.groupby("component").size().reset_index(name="spits")

    p = p.sort_values("spits",ascending=False)

    return p


# ---------------------------------------------------
# REPEATED SPITS
# ---------------------------------------------------

def repeated_spits(df):

    rejects = df[df.reject_code!="0"]

    r = (
        rejects
        .groupby(["board","location","component"])
        .size()
        .reset_index(name="count")
    )

    r = r[r["count"]>1]

    return r


# ---------------------------------------------------
# YIELD LOSS
# ---------------------------------------------------

def yield_loss(df, bom):

    s = build_summary(df,bom)

    loss = s.groupby("board")["total_cost"].sum().reset_index()

    boards = df.filename.nunique()

    machines_line2 = [
        "IINEO682",
        "IIN2-053-2",
        "IIN2-053-1"
    ]

    if df.machine.isin(machines_line2).any():
        boards = int(boards/3)

    loss["boards_run"] = boards

    loss["loss_per_board"] = loss["total_cost"] / boards

    return loss


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title("SMT Spit Analysis System")

logs = st.file_uploader("Upload Log Files", accept_multiple_files=True)

bom = st.file_uploader("Upload Master BOM")

if logs and st.button("Ingest Logs"):
    ingest_logs(logs)
    st.success("Logs ingested")

if bom and st.button("Ingest BOM"):
    ingest_bom(bom)
    st.success("BOM ingested")

df, bom_df = load_data()

if df.empty:
    st.stop()

st.sidebar.header("Filters")

board = st.sidebar.selectbox("Board", [""] + sorted(df.board.unique().tolist()))

machine = st.sidebar.selectbox("Machine", [""] + sorted(df.machine.unique().tolist()))

mo = st.sidebar.selectbox("MO", [""] + sorted(df.mo.unique().tolist()))

start = st.sidebar.date_input("Start Date", None)

end = st.sidebar.date_input("End Date", None)

df = apply_filters(df, board, machine, mo, start, end)

tabs = st.tabs([
"Summary",
"Spit Events",
"Pareto",
"Repeated Spits",
"Yield Loss"
])


with tabs[0]:

    s = build_summary(df,bom_df)

    st.dataframe(s)


with tabs[1]:

    e = spit_events(df,bom_df)

    st.dataframe(e)


with tabs[2]:

    p = pareto(df)

    st.bar_chart(p.set_index("component"))


with tabs[3]:

    r = repeated_spits(df)

    st.dataframe(r)


with tabs[4]:

    y = yield_loss(df,bom_df)

    st.dataframe(y)
