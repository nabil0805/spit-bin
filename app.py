import streamlit as st
import pandas as pd
import sqlite3
import os
import io
import re
from datetime import datetime
from openai import OpenAI

st.set_page_config(page_title="SMT Spit Analysis",layout="wide")

DB="smt_logs.db"

def get_conn():
    return sqlite3.connect(DB,check_same_thread=False)

conn=get_conn()

def init_db():
    c=conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        board TEXT,
        mo TEXT,
        date TEXT,
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

def ingest_logs(files):

    rows=[]

    for f in files:

        data=f.read()

        try:
            df=pd.read_csv(io.BytesIO(data),skiprows=2,header=None,encoding="latin-1")
        except:
            continue

        board=df.iloc[0,1]
        mo=df.iloc[0,3]
        date=df.iloc[0,8]
        machine=df.iloc[0,11]

        for _,r in df.iterrows():

            component=str(r[1])
            desc=str(r[2])
            loc=str(r[3])
            feeder=str(r[7])
            slot=str(r[8])
            code=str(r[11])

            rows.append([
                f.name,
                board,
                mo,
                date,
                machine,
                component,
                desc,
                loc,
                feeder,
                slot,
                code
            ])

    c=conn.cursor()

    c.executemany("""
    INSERT INTO logs(
    filename,board,mo,date,machine,
    component,description,location,
    feeder,slot,reject_code
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,rows)

    conn.commit()

def ingest_bom(file):

    df=pd.read_excel(file)

    rows=[]

    for _,r in df.iterrows():

        comp=str(r[0]).strip()
        cost=r[9]

        if pd.isna(comp): 
            continue

        rows.append((comp,cost))

    c=conn.cursor()

    c.executemany("""
    INSERT OR REPLACE INTO bom(component,cost)
    VALUES (?,?)
    """,rows)

    conn.commit()

def get_data():

    df=pd.read_sql("SELECT * FROM logs",conn)

    bom=pd.read_sql("SELECT * FROM bom",conn)

    return df,bom

def apply_filters(df,board,machine,mo,start,end):

    if board:
        df=df[df.board==board]

    if machine:
        df=df[df.machine==machine]

    if mo:
        df=df[df.mo==mo]

    if start:
        df=df[df.date>=str(start)]

    if end:
        df=df[df.date<=str(end)]

    return df

def summary(df,bom):

    rejects=df[df.reject_code!="0"]

    grp=rejects.groupby(["component","board"]).size().reset_index(name="spits")

    grp=grp.merge(bom,on="component",how="left")

    grp["total_cost"]=grp["spits"]*grp["cost"]

    codes=rejects.groupby(["component","board","reject_code"]).size()

    code_map={}

    for (c,b,code),cnt in codes.items():

        k=(c,b)

        txt=f"{cnt}x C{code}"

        code_map.setdefault(k,[]).append(txt)

    grp["reject_codes"]=grp.apply(
        lambda r:", ".join(code_map.get((r.component,r.board),[])),axis=1
    )

    placement=df[df.reject_code=="0"]

    placement=placement.merge(bom,on="component",how="left")

    board_totals=placement.groupby("board")["cost"].sum()

    grp["total_placement_cost"]=grp["board"].map(board_totals)

    grp["loss_percent"]=grp["total_cost"]/grp["total_placement_cost"]*100

    return grp

def spit_events(df,bom):

    rej=df[df.reject_code!="0"]

    rej=rej.merge(bom,on="component",how="left")

    return rej

def pareto(df):

    p=df.groupby("component").size().reset_index(name="spits")

    p=p.sort_values("spits",ascending=False)

    return p

def repeated_spits(df):

    r=df[df.reject_code!="0"]

    rep=r.groupby(["board","location","component"]).size()

    rep=rep.reset_index(name="count")

    rep=rep[rep["count"]>1]

    return rep

def yield_loss(df,bom):

    s=summary(df,bom)

    y=s.groupby("board")["total_cost"].sum().reset_index()

    boards=df.filename.nunique()

    machines_line2=["IINEO682","IIN2-053-2","IIN2-053-1"]

    if df.machine.isin(machines_line2).any():

        boards=int(boards/3)

    y["boards_run"]=boards

    y["loss_per_board"]=y["total_cost"]/boards

    return y

def chatbot(df):

    key=st.text_input("OpenAI API key",type="password")

    if not key:
        return

    client=OpenAI(api_key=key)

    q=st.text_input("Ask about the data")

    if not q:
        return

    data=df.to_csv(index=False)

    prompt=f"""
You are SMT manufacturing analyst.

Dataset:

{data}

Question:
{q}

Answer clearly.
"""

    resp=client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    st.write(resp.output_text)

st.title("SMT Spit Analysis Tool")

log_files=st.file_uploader("Upload Log Files",accept_multiple_files=True)

bom_file=st.file_uploader("Upload Master BOM")

if log_files and st.button("Ingest Logs"):
    ingest_logs(log_files)
    st.success("Logs ingested")

if bom_file and st.button("Ingest BOM"):
    ingest_bom(bom_file)
    st.success("BOM ingested")

df,bom=get_data()

if df.empty:
    st.stop()

st.sidebar.header("Filters")

board=st.sidebar.selectbox("Board",[""]+sorted(df.board.unique().tolist()))

machine=st.sidebar.selectbox("Machine",[""]+sorted(df.machine.unique().tolist()))

mo=st.sidebar.selectbox("MO",[""]+sorted(df.mo.unique().tolist()))

start=st.sidebar.date_input("Start Date",None)

end=st.sidebar.date_input("End Date",None)

df=apply_filters(df,board,machine,mo,start,end)

tabs=st.tabs([
"Summary",
"Spit Events",
"Pareto",
"Repeated Spits",
"Yield Loss",
"Chatbot"
])

with tabs[0]:

    s=summary(df,bom)

    st.dataframe(s)

with tabs[1]:

    e=spit_events(df,bom)

    st.dataframe(e)

with tabs[2]:

    p=pareto(df)

    st.bar_chart(p.set_index("component"))

with tabs[3]:

    r=repeated_spits(df)

    st.dataframe(r)

with tabs[4]:

    y=yield_loss(df,bom)

    st.dataframe(y)

with tabs[5]:

    chatbot(df)
