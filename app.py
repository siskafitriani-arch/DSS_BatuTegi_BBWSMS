import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
import os
import io
import json

st.set_page_config(page_title="DSS Bendungan Batutegi", layout="wide")

# ======================
# FILE PATH
# ======================
MODEL_FILE = "model_lstm_batutegi.keras"
SCALER_X_FILE = "scaler_X.pkl"
SCALER_Y_FILE = "scaler_y.pkl"
DATASET_FILE = "dataset_lstm_batutegi_ready.xlsx"
VALIDASI_FILE = "hasil_prediksi_lstm.xlsx"
METRICS_FILE = "metrics_lstm.xlsx"
HISTORY_FILE = "history_loss.csv"

# ======================
# LOAD MODEL & SCALERS
# ======================
def load_model_resources(model_file=MODEL_FILE, scaler_x_file=SCALER_X_FILE, scaler_y_file=SCALER_Y_FILE):
    try:
        model = load_model(model_file)
    except Exception as e:
        raise FileNotFoundError(f"Gagal memuat model: {e}")
    try:
        scaler_X = joblib.load(scaler_x_file)
    except Exception as e:
        raise FileNotFoundError(f"Gagal memuat scaler X: {e}")
    try:
        scaler_y = joblib.load(scaler_y_file)
    except Exception as e:
        raise FileNotFoundError(f"Gagal memuat scaler Y: {e}")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_resources()

# ======================
# LOAD DATASET
# ======================
@st.cache_data
def load_dataset():
    hist = pd.read_excel(DATASET_FILE)
    hist.columns = [str(c).strip().lower() for c in hist.columns]
    hist["tanggal"] = pd.to_datetime(hist["tanggal"], errors="coerce")
    hist = hist.sort_values("tanggal").reset_index(drop=True)

    validasi = pd.read_excel(VALIDASI_FILE)
    validasi.columns = [str(c).strip().lower() for c in validasi.columns]

    try:
        metrics = pd.read_excel(METRICS_FILE)
        metrics.columns = [str(c).strip().lower() for c in metrics.columns]
    except:
        metrics = pd.DataFrame()

    try:
        history = pd.read_csv(HISTORY_FILE)
    except:
        history = None

    return hist, validasi, metrics, history

hist_df, validasi_df, metrics_df, history_df = load_dataset()

# ======================
# FEATURE COLUMNS
# ======================
feature_cols = [
    "pos_hujan_1","pos_hujan_2",
    "pos_hujan_1_lag_1","pos_hujan_1_lag_2","pos_hujan_1_lag_3",
    "pos_hujan_1_lag_4","pos_hujan_1_lag_5","pos_hujan_1_lag_6","pos_hujan_1_lag_7",
    "pos_hujan_2_lag_1","pos_hujan_2_lag_2","pos_hujan_2_lag_3",
    "pos_hujan_2_lag_4","pos_hujan_2_lag_5","pos_hujan_2_lag_6","pos_hujan_2_lag_7",
    "bulan","time_index"
]

def clean_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

hist_df = clean_numeric(hist_df, feature_cols)
hist_df = hist_df.dropna(subset=feature_cols+["tanggal"]).reset_index(drop=True)

# ======================
# DSS FUNCTIONS
# ======================
def kategori_risiko(nilai):
    if nilai >= 50:
        return "Tinggi"
    elif nilai >= 20:
        return "Sedang"
    else:
        return "Rendah"

def rekomendasi_dss(kategori):
    if kategori=="Tinggi":
        return "SIAGA - Monitoring intensif inflow dan kesiapsiagaan operasional."
    elif kategori=="Sedang":
        return "WASPADA - Pantau kondisi DAS dan potensi kenaikan inflow."
    else:
        return "NORMAL - Operasi normal dan pemantauan rutin."

# ======================
# PREDIKSI 15 HARI
# ======================
def prediksi_15_hari_lstm(tanggal_awal,pos_hujan_1_input,pos_hujan_2_input):
    tanggal_awal = pd.to_datetime(tanggal_awal)
    data_sebelum = hist_df[hist_df["tanggal"] <= tanggal_awal].copy()
    base_df = data_sebelum.tail(7) if len(data_sebelum)>=7 else hist_df.tail(7)
    r1_series = list(base_df["pos_hujan_1"].values)+[pos_hujan_1_input]
    r2_series = list(base_df["pos_hujan_2"].values)+[pos_hujan_2_input]
    current_time_index = int(base_df["time_index"].iloc[-1])
    hasil=[]
    for i in range(1,16):
        tanggal_pred = tanggal_awal + pd.Timedelta(days=i)
        row_input={"pos_hujan_1":r1_series[-1],"pos_hujan_2":r2_series[-1],
                   "bulan":tanggal_pred.month,"time_index":current_time_index}
        for lag in range(1,8):
            row_input[f"pos_hujan_1_lag_{lag}"] = r1_series[-lag]
            row_input[f"pos_hujan_2_lag_{lag}"] = r2_series[-lag]
        X_input = pd.DataFrame([row_input])[feature_cols].values
        X_scaled = scaler_X.transform(X_input).reshape((1,1,len(feature_cols)))
        pred_scaled = model.predict(X_scaled, verbose=0)
        pred = max(float(scaler_y.inverse_transform(pred_scaled)[0][0]),0)
        kategori = kategori_risiko(pred)
        hasil.append({"tanggal_prediksi":tanggal_pred,"hari_ke":f"H+{i}",
                      "pos_hujan_1_input":pos_hujan_1_input,"pos_hujan_2_input":pos_hujan_2_input,
                      "prediksi_hujan_lstm":pred,"kategori_risiko":kategori,
                      "rekomendasi":rekomendasi_dss(kategori)})
        r1_series.append(pred)
        r2_series.append(r2_series[-1])
        current_time_index +=1
    return pd.DataFrame(hasil)

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.title("DSS Bendungan Batutegi")
menu = st.sidebar.radio("Menu", ["Overview","Validasi Model","Prediksi Interaktif","Rekomendasi DSS","Data"])
pos_hujan_1_input = st.sidebar.number_input("Curah Hujan Pos 1", value=float(hist_df["pos_hujan_1"].iloc[-1]))
pos_hujan_2_input = st.sidebar.number_input("Curah Hujan Pos 2", value=float(hist_df["pos_hujan_2"].iloc[-1]))
tanggal_input = st.sidebar.date_input("Tanggal awal prediksi", value=hist_df["tanggal"].max())
pred15_df = prediksi_15_hari_lstm(tanggal_input,pos_hujan_1_input,pos_hujan_2_input)
max_pred = pred15_df["prediksi_hujan_lstm"].max()
status = kategori_risiko(max_pred)

# ======================
# PAGES
# ======================
if menu=="Overview":
    st.title("Overview DSS Batutegi")
    st.metric("Prediksi Maks 15 Hari",f"{max_pred:.2f} mm")
    st.metric("Status DSS", status)
    fig = px.line(pred15_df,x="hari_ke",y="prediksi_hujan_lstm",markers=True,title="Prediksi Curah Hujan 15 Hari")
    st.plotly_chart(fig,use_container_width=True)

elif menu=="Validasi Model":
    st.title("Validasi Model LSTM")
    st.dataframe(validasi_df,use_container_width=True)
    # Grafik Aktual vs Prediksi
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(validasi_df['aktual'], label='Data Aktual')
    ax.plot(validasi_df['prediksi'], label='Hasil Prediksi')
    ax.set_title("Grafik Aktual vs Prediksi")
    ax.set_xlabel("Data Uji")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    # Grafik 100 data pertama
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(validasi_df['aktual'][:100], label='Data Aktual')
    ax2.plot(validasi_df['prediksi'][:100], label='Hasil Prediksi')
    ax2.set_title("Grafik Aktual vs Prediksi (100 Data Pertama)")
    ax2.set_xlabel("Data Uji")
    ax2.set_ylabel("Curah Hujan (mm)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    # Grafik Loss Training jika ada
    if history_df is not None:
        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.plot(history_df['loss'], label='Training Loss')
        if 'val_loss' in history_df.columns:
            ax3.plot(history_df['val_loss'], label='Validation Loss')
        ax3.set_title("Grafik Loss Training dan Validation")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

elif menu=="Prediksi Interaktif":
    st.title("Prediksi Interaktif")
    st.dataframe(pred15_df,use_container_width=True)
    fig = px.bar(pred15_df,x="hari_ke",y="prediksi_hujan_lstm",
                 color="kategori_risiko",text="prediksi_hujan_lstm",
                 title="Prediksi Curah Hujan 15 Hari")
    st.plotly_chart(fig,use_container_width=True)

elif menu=="Rekomendasi DSS":
    st.title("Rekomendasi DSS")
    for _, row in pred15_df.iterrows():
        if row["kategori_risiko"]=="Tinggi":
            st.error(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")
        elif row["kategori_risiko"]=="Sedang":
            st.warning(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")
        else:
            st.success(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")

elif menu=="Data":
    st.title("Data Dashboard")
    st.subheader("Data Historis")
    st.dataframe(hist_df,use_container_width=True)
    st.subheader("Prediksi 15 Hari")
    st.dataframe(pred15_df,use_container_width=True)
    st.subheader("Validasi Model")
    st.dataframe(validasi_df,use_container_width=True)
    st.subheader("Metrik Model")
    st.dataframe(metrics_df,use_container_width=True)
