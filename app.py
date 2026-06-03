import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from datetime import timedelta, datetime
import os
import io
import json

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="DSS Realtime Batutegi", layout="wide")

# --- FUNGSI LOAD MODEL & HISTORY (CACHING) ---
@st.cache_resource
def load_model_resources():
    """
    Memuat model, scaler, dan history training.
    """
try:
    model = keras.models.load_model('model_lstm_batutegi.keras')

    # Load scaler (jika pakai pickle)
import pickle
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
        
        # Load History Training (Opsional)
        history_data = None
        if os.path.exists("history_training.json"):
            with open("history_training.json", "r") as f:
                history_data = json.load(f)
                
        return model, scaler_X, scaler_y, history_data
    except Exception as e:
        st.error(f"❌ Gagal memuat resource: {e}")
        st.info("Pastikan file 'model_lstm_batutegi.keras', 'scaler_X.pkl', 'scaler_y.pkl', dan opsional 'history_training.json' ada di folder proyek.")
        st.stop()

# --- LOGIKA PREDIKSI 15 HARI (BERBASIS INPUT MANUAL) ---
def prediksi_15_hari_manual(input_hari_ini, history_7_hari, model, scaler_X, scaler_y):
    """
    input_hari_ini: dict {'rainfall_1': val, 'rainfall_2': val} -> Data REALTIME hari ini
    history_7_hari: list of dicts [{'rainfall_1': val, 'rainfall_2': val}, ...] 
                    Urutan: [t-1, t-2, ..., t-7] (Kemarin sampai 7 hari lalu)
    """
    
    # Fitur yang dibutuhkan model (Harus sama persis urutan & namanya saat training)
    feature_cols = [
       "rainfall_1", "rainfall_2",
       "rainfall_1_lag_1", "rainfall_1_lag_2", "rainfall_1_lag_3",
       "rainfall_1_lag_4", "rainfall_1_lag_5", "rainfall_1_lag_6", "rainfall_1_lag_7",
       "rainfall_2_lag_1", "rainfall_2_lag_2", "rainfall_2_lag_3",
       "rainfall_2_lag_4", "rainfall_2_lag_5", "rainfall_2_lag_6", "rainfall_2_lag_7",
       "bulan", "time_index"
    ]
    
    hasil_prediksi = []
    current_date = datetime.now() + timedelta(days=1) # Prediksi mulai dari BESOK
    
    # Ambil nilai history untuk inisialisasi lag
    lags_1_init = [h['rainfall_1'] for h in history_7_hari]
    lags_2_init = [h['rainfall_2'] for h in history_7_hari]
    
    # Virtual History untuk iterasi recursive
    v_lags_1 = lags_1_init.copy()
    v_lags_2 = lags_2_init.copy()
    
    # Nilai input saat ini
    curr_h1 = input_hari_ini['rainfall_1']
    curr_h2 = input_hari_ini['rainfall_2']
    
    # Time index placeholder
    base_time_index = 1000 

    for i in range(1, 16):
        row_data = [
            curr_h1, curr_h2, 
            v_lags_1[0], v_lags_1[1], v_lags_1[2], v_lags_1[3], v_lags_1[4], v_lags_1[5], v_lags_1[6],
            v_lags_2[0], v_lags_2[1], v_lags_2[2], v_lags_2[3], v_lags_2[4], v_lags_2[5], v_lags_2[6],
            current_date.month, base_time_index + i
        ]
        
        df_input = pd.DataFrame([row_data], columns=feature_cols)
        
        # Scaling
        X_scaled = scaler_X.transform(df_input)
        X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        
        # Predict
        pred_scaled = model.predict(X_scaled, verbose=0)
        pred_val = scaler_y.inverse_transform(pred_scaled)[0][0]
        pred_val = max(pred_val, 0) 
        
        hasil_prediksi.append({
            "tanggal": current_date,
            "hari_ke": f"H+{i}",
            "prediksi_hujan_lstm": pred_val
        })
        
        # Update Virtual History
        v_lags_1.insert(0, curr_h1)
        v_lags_1.pop()
        
        v_lags_2.insert(0, curr_h2)
        v_lags_2.pop()
        
        curr_h1 = pred_val
        
        current_date += timedelta(days=1)
        
    return pd.DataFrame(hasil_prediksi)

# --- LOGIKA DSS (NERACA AIR) ---
def dss_irigasi_dan_pemeliharaan(df_prediksi, kapasitas_waduk_m3, volume_saat_ini_m3):
    LUAS_DAERAH_TANGKAPAN_KM2 = 50  
    KOEFISIEN_ALIRAN_MASUK = 0.3    
    DEBIT_IRIGASI_RATA_RATA_M3_HARI = 5000 
    DEBIT_EVAPORASI_M3_HARI = 1000  
    
    rekomendasi = []
    volume_air = volume_saat_ini_m3
    
    for index, row in df_prediksi.iterrows():
        tanggal = row['tanggal']
        hujan_mm = row['prediksi_hujan_lstm']
        
        # Hitung Neraca Air
        input_air_m3 = hujan_mm * (LUAS_DAERAH_TANGKAPAN_KM2 * 1_000_000) * KOEFISIEN_ALIRAN_MASUK / 1000
        output_air_m3 = DEBIT_IRIGASI_RATA_RATA_M3_HARI + DEBIT_EVAPORASI_M3_HARI
        
        volume_air = volume_air + input_air_m3 - output_air_m3
        
        # Batasan Fisik
        if volume_air > kapasitas_waduk_m3: volume_air = kapasitas_waduk_m3
        elif volume_air < 0: volume_air = 0
            
        persentase_kapasitas = (volume_air / kapasitas_waduk_m3) * 100
        
        # Logika Status
        if persentase_kapasitas > 60:
            status_irigasi = "AMAN - Pasokan Penuh"
            rek_irigasi = "Buka pintu air normal."
        elif persentase_kapasitas > 30:
            status_irigasi = "WASPADA - Hemat Air"
            rek_irigasi = "Kurangi debit 20%."
        else:
            status_irigasi = "KRITIS - Darurat"
            rek_irigasi = "Rotasi ketat."
            
        if hujan_mm > 20:
            status_maint = "DITUNDA"
            alasan_maint = "Hujan tinggi (>20mm)."
        elif persentase_kapasitas < 20:
            status_maint = "DITUNDA"
            alasan_maint = "Air kritis."
        else:
            status_maint = "DIREKOMENDASIKAN"
            alasan_maint = "Cuaca & kondisi aman."
            
        rekomendasi.append({
            'tanggal': tanggal,
            'prediksi_hujan_mm': round(hujan_mm, 2),
            'estimasi_volume_m3': round(volume_air, 2),
            'persentase_kapasitas': round(persentase_kapasitas, 2),
            'status_irigasi': status_irigasi,
            'rekomendasi_irigasi': rek_irigasi,
            'status_maintenance': status_maint,
            'alasan_maintenance': alasan_maint
        })
        
    return pd.DataFrame(rekomendasi)

# --- UI STREAMLIT ---
st.title("🌊 DSS Realtime Bendungan Batutegi")
st.markdown("Sistem pendukung keputusan berbasis data **Titik Tangkapan Air Lokal**.")

# SIDEBAR: INPUT DATA
st.sidebar.header("📝 Input Data Curah Hujan")

# 1. Data Realtime Hari Ini
st.sidebar.subheader("Hari Ini (Realtime)")
h1_today = st.sidebar.number_input("Curah Hujan Titik 1 (mm)", value=0.0, step=0.1, format="%.2f")
h2_today = st.sidebar.number_input("Curah Hujan Titik 2 (mm)", value=0.0, step=0.1, format="%.2f")

# 2. Data Historis
st.sidebar.subheader("Data 7 Hari Terakhir (Historis)")
st.sidebar.caption("Model LSTM butuh data masa lalu. Isi dengan data riil.")

history_data = []
for i in range(7):
    st.sidebar.text(f"H-{i+1} (Hari ke-{i+1} lalu)")
    col1, col2 = st.sidebar.columns(2)
    h1_hist = col1.number_input(f"Titik 1", value=5.0, step=0.1, key=f"h1_hist_{i}")
    h2_hist = col2.number_input(f"Titik 2", value=5.0, step=0.1, key=f"h2_hist_{i}")
    history_data.append({'rainfall_1': h1_hist, 'rainfall_2': h2_hist})

# Parameter Waduk
st.sidebar.header("⚙️ Parameter Operasional")
kapasitas_waduk = st.sidebar.number_input("Kapasitas Total (m³)", value=1000000, step=10000)
volume_awal = st.sidebar.number_input("Volume Air Saat Ini (m³)", value=600000, step=10000)

# TOMBOL PROSES
if st.button("🚀 Jalankan Prediksi & Analisis DSS"):
    with st.spinner('Menghitung prediksi berbasis input lokal...'):
        # Load Model & History
        model, scaler_X, scaler_y, train_history = load_model_resources()
        
        # Siapkan Input
        input_realtime = {'rainfall_1': h1_today, 'rainfall_2': h2_today}
        
        # 1. Prediksi
        df_prediksi = prediksi_15_hari_manual(input_realtime, history_data, model, scaler_X, scaler_y)
        
        # 2. DSS
        df_dss = dss_irigasi_dan_pemeliharaan(df_prediksi, kapasitas_waduk, volume_awal)
        
        # --- TAMPILAN DASHBOARD ---
        
        # Metrik Utama
        col1, col2, col3 = st.columns(3)
        besok = df_dss.iloc[0]
        
        col1.metric("Prediksi Hujan Besok", f"{besok['prediksi_hujan_mm']:.2f} mm")
        col2.metric("Estimasi Volume Besok", f"{besok['estimasi_volume_m3']:,.0f} m³")
        col3.metric("Status Irigasi", besok['status_irigasi'])
        
        # Grafik 1: Proyeksi Hujan vs Volume
        st.subheader("📈 Proyeksi 15 Hari Kedepan")
        fig_proj = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_proj.add_trace(
            go.Bar(x=df_dss['tanggal'], y=df_dss['prediksi_hujan_mm'], name="Curah Hujan (mm)", marker_color='royalblue'),
            secondary_y=False
        )
        
        fig_proj.add_trace(
            go.Scatter(x=df_dss['tanggal'], y=df_dss['persentase_kapasitas'], name="% Kapasitas", line=dict(color='red', width=3)),
            secondary_y=True
        )
        
        fig_proj.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Batas Aman", secondary_y=True)
        fig_proj.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Batas Waspada", secondary_y=True)
        
        fig_proj.update_layout(title_text="Proyeksi: Hujan vs Volume Waduk", hovermode="x unified")
        st.plotly_chart(fig_proj, use_container_width=True)

        # Grafik 2: Training Loss (Jika data tersedia)
        if train_history:
            st.subheader("📉 Performa Model (Training History)")
            st.caption("Grafik ini menunjukkan bagaimana model belajar mengurangi error selama pelatihan.")
            
            epochs = range(1, len(train_history['loss']) + 1)
            
            fig_loss = go.Figure()
            
            # Training Loss
            fig_loss.add_trace(go.Scatter(
                x=list(epochs), 
                y=train_history['loss'], 
                mode='lines+markers', 
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            
            # Validation Loss (jika ada)
            if 'val_loss' in train_history:
                fig_loss.add_trace(go.Scatter(
                    x=list(epochs), 
                    y=train_history['val_loss'], 
                    mode='lines+markers', 
                    name='Validation Loss',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
            fig_loss.update_layout(
                title="Evolution of Loss During Training",
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE/MAE)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.warning("⚠️ File `history_training.json` tidak ditemukan. Grafik performa model tidak dapat ditampilkan.")
            st.info("Tips: Simpan history training saat fitting model menggunakan `json.dump(history.history, file)` untuk melihat grafik ini.")
        
        # Tabel Rekomendasi
        st.subheader("📋 Tabel Rekomendasi Operasional")
        df_tampilan = df_dss[['tanggal', 'prediksi_hujan_mm', 'status_irigasi', 'status_maintenance', 'alasan_maintenance']]
        df_tampilan.columns = ['Tanggal', 'Hujan (mm)', 'Status Irigasi', 'Maintenance', 'Alasan']
        st.dataframe(df_tampilan, use_container_width=True)
        
        # Download Excel
        @st.cache_data
        def convert_df_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='DSS_Result')
            return output.getvalue()
            
        excel_data = convert_df_to_excel(df_dss)
        st.download_button(
            label="📥 Download Hasil (.xlsx)",
            data=excel_data,
            file_name=f'DSS_Batutegi_{datetime.now().strftime("%Y%m%d")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

else:
    st.info("👈 Silakan isi data curah hujan dari titik tangkapan air Anda di sidebar, lalu klik tombol **Jalankan Prediksi**.")
