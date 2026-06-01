import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from datetime import timedelta

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="DSS Waduk Batutegi", layout="wide")

# --- FUNGSI LOAD MODEL (CACHING) ---
@st.cache_resource
def load_dss_resources():
    """
    Memuat model dan scaler sekali saja saat aplikasi dimulai.
    Menggunakan cache agar tidak reload setiap kali user berinteraksi.
    """
    try:
        model = load_model("model_lstm_batutegi.keras")
        scaler_X = joblib.load("scaler_X.pkl")
        scaler_y = joblib.load("scaler_y.pkl")
        df_last = pd.read_excel("dataset_lstm_batutegi_ready.xlsx")
        return model, scaler_X, scaler_y, df_last
    except Exception as e:
        st.error(f"Gagal memuat file model/data: {e}")
        st.stop()

# --- LOGIKA PREDIKSI (SAMA SEPERTI SCRIPT ASLI) ---
def prediksi_15_hari(df, model, scaler_X, scaler_y):
    df_pred = df.copy()
    hasil_prediksi = []
    
    feature_cols = [
       "pos_hujan_1", "pos_hujan_2",
       "pos_hujan_1_lag_1", "pos_hujan_1_lag_2", "pos_hujan_1_lag_3",
       "pos_hujan_1_lag_4", "pos_hujan_1_lag_5", "pos_hujan_1_lag_6", "pos_hujan_1_lag_7",
       "pos_hujan_2_lag_1", "pos_hujan_2_lag_2", "pos_hujan_2_lag_3",
       "pos_hujan_2_lag_4", "pos_hujan_2_lag_5", "pos_hujan_2_lag_6", "pos_hujan_2_lag_7",
       "bulan", "time_index"
    ]
    
    last_date = pd.to_datetime(df_pred["tanggal"].iloc[-1])
    
    for i in range(1, 16):
        last_row = df_pred.iloc[-1].copy()
        
        X_input = last_row[feature_cols].values.reshape(1, -1)
        X_scaled = scaler_X.transform(X_input)
        X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        
        pred_scaled = model.predict(X_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        
        # Curah hujan tidak boleh negatif
        pred = max(pred, 0)
        
        next_date = last_date + timedelta(days=i)
        
        hasil_prediksi.append({
            "tanggal": next_date,
            "hari_ke": f"H+{i}",
            "prediksi_hujan_lstm": pred
        })
        
        # Buat baris baru untuk iterasi berikutnya
        new_row = last_row.copy()
        new_row["tanggal"] = next_date
        new_row["pos_hujan_1"] = pred
        new_row["pos_hujan_2"] = last_row["pos_hujan_2"] 
        new_row["bulan"] = next_date.month
        new_row["time_index"] = last_row["time_index"] + 1
        
        # Update lag rainfall_1
        for lag in range(7, 1, -1):
            new_row[f"pos_hujan_1_lag_{lag}"] = last_row[f"pos_hujan_1_lag_{lag-1}"]
        new_row["pos_hujan_1_lag_1"] = pred
        
        # Update lag rainfall_2
        for lag in range(7, 1, -1):
            new_row[f"pos_hujan_2_lag_{lag}"] = last_row[f"pos_hujan_2_lag_{lag-1}"]
        new_row["pos_hujan_2_lag_1"] = last_row["pos_hujan_2"]
        
        df_pred = pd.concat([df_pred, pd.DataFrame([new_row])], ignore_index=True)
        
    return pd.DataFrame(hasil_prediksi)

# --- LOGIKA DSS (SAMA SEPERTI SCRIPT ASLI) ---
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
        
        input_air_m3 = hujan_mm * (LUAS_DAERAH_TANGKAPAN_KM2 * 1_000_000) * KOEFISIEN_ALIRAN_MASUK / 1000
        output_air_m3 = DEBIT_IRIGASI_RATA_RATA_M3_HARI + DEBIT_EVAPORASI_M3_HARI
        
        volume_air = volume_air + input_air_m3 - output_air_m3
        
        if volume_air > kapasitas_waduk_m3:
            volume_air = kapasitas_waduk_m3
        elif volume_air < 0:
            volume_air = 0
            
        persentase_kapasitas = (volume_air / kapasitas_waduk_m3) * 100
        
        if persentase_kapasitas > 60:
            status_irigasi = "AMAN - Pasokan Penuh"
            rekomendasi_irigasi = "Buka pintu air sesuai jadwal normal."
        elif persentase_kapasitas > 30:
            status_irigasi = "WASPADA - Hemat Air"
            rekomendasi_irigasi = "Kurangi debit 20%. Prioritaskan tanaman kritis."
        else:
            status_irigasi = "KRITIS - Darurat"
            rekomendasi_irigasi = "Rotasi ketat. Hanya untuk tanaman masa generatif."
            
        if hujan_mm > 20:
            status_maintenance = "DITUNDA"
            alasan_maintenance = "Curah hujan tinggi (>20mm). Risiko banjir & longsor."
        elif persentase_kapasitas < 20:
            status_maintenance = "DITUNDA"
            alasan_maintenance = "Status air kritis. Fokus pada distribusi air."
        else:
            status_maintenance = "DIREKOMENDASIKAN"
            alasan_maintenance = "Cuaca mendukung & kondisi air aman."
            
        rekomendasi.append({
            'tanggal': tanggal,
            'prediksi_hujan_mm': round(hujan_mm, 2),
            'estimasi_volume_m3': round(volume_air, 2),
            'persentase_kapasitas': round(persentase_kapasitas, 2),
            'status_irigasi': status_irigasi,
            'rekomendasi_irigasi': rekomendasi_irigasi,
            'status_maintenance': status_maintenance,
            'alasan_maintenance': alasan_maintenance
        })
        
    return pd.DataFrame(rekomendasi)

# --- UI STREAMLIT ---
st.title("🌊 Dashboard DSS Waduk Batutegi")
st.markdown("Sistem Pendukung Keputusan untuk Prediksi Hujan dan Manajemen Irigasi.")

# Sidebar untuk Input Parameter
st.sidebar.header("⚙️ Parameter Operasional")
kapasitas_waduk = st.sidebar.number_input("Kapasitas Waduk (m³)", value=1000000, step=10000)
volume_awal = st.sidebar.number_input("Volume Air Saat Ini (m³)", value=600000, step=10000)

# Tombol Trigger Prediksi
if st.button("🔄 Jalankan Prediksi & Analisis DSS"):
    with st.spinner('Memuat model dan menghitung prediksi...'):
        # Load resources
        model, scaler_X, scaler_y, df_last = load_dss_resources()
        
        # 1. Jalankan Prediksi
        hasil_15hari = prediksi_15_hari(df_last, model, scaler_X, scaler_y)
        
        # 2. Jalankan DSS
        df_dss = dss_irigasi_dan_pemeliharaan(hasil_15hari, kapasitas_waduk, volume_awal)
        
        # --- TAMPILAN HASIL ---
        
        # Kolom Metrik Utama (Hari Ini/Besok)
        col1, col2, col3 = st.columns(3)
        hari_ini = df_dss.iloc[0]
        
        col1.metric("Prediksi Hujan Besok", f"{hari_ini['prediksi_hujan_mm']:.2f} mm")
        col2.metric("Estimasi Volume Akhir", f"{hari_ini['estimasi_volume_m3']:,.0f} m³")
        col3.metric("Status Irigasi", hari_ini['status_irigasi'])

        # Grafik Interaktif dengan Plotly
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar Chart untuk Hujan
        fig.add_trace(
            go.Bar(x=df_dss['tanggal'], y=df_dss['prediksi_hujan_mm'], name="Curah Hujan (mm)", marker_color='royalblue'),
            secondary_y=False
        )
        
        # Line Chart untuk Volume %
        fig.add_trace(
            go.Scatter(x=df_dss['tanggal'], y=df_dss['persentase_kapasitas'], name="% Kapasitas Waduk", line=dict(color='red', width=3)),
            secondary_y=True
        )
        
        # Garis Batas
        fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Batas Aman (60%)", secondary_y=True)
        fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Batas Waspada (30%)", secondary_y=True)
        
        fig.update_layout(
            title_text="Prediksi Hujan vs Proyeksi Volume Waduk (15 Hari)",
            xaxis_title="Tanggal",
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Curah Hujan (mm)", secondary_y=False)
        fig.update_yaxes(title_text="Kapasitas Waduk (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel Rekomendasi Detail
        st.subheader("📋 Tabel Rekomendasi Operasional")
        
        # Format tampilan tabel agar lebih bersih
        df_tampilan = df_dss[['tanggal', 'prediksi_hujan_mm', 'status_irigasi', 'rekomendasi_irigasi', 'status_maintenance', 'alasan_maintenance']]
        df_tampilan.columns = ['Tanggal', 'Hujan (mm)', 'Status Irigasi', 'Rekomendasi Irigasi', 'Status Maintenance', 'Alasan']
        
        # Fungsi untuk mewarnai baris berdasarkan status (opsional, sederhana saja)
        st.dataframe(df_tampilan, use_container_width=True)
        
        # Tombol Download Excel
        @st.cache_data
        def convert_df_to_excel(df):
            return df.to_excel(index=False)
            
        excel_data = convert_df_to_excel(df_dss)
        st.download_button(
            label="📥 Download Hasil DSS (.xlsx)",
            data=excel_data,
            file_name='rekomendasi_dss_batutegi.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

else:
    st.info("👈 Silakan klik tombol **'Jalankan Prediksi & Analisis DSS'** di atas untuk melihat hasil.")
