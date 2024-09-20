import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set seed untuk reproducibility
np.random.seed(42)

st.title("Analisis dan Prediksi Reach untuk Perusahaan")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

# Fungsi untuk menangani NaN dan nilai tak terbatas pada prediksi
def safe_expm1(predictions):
    # Clip prediksi ke rentang untuk menghindari overflow
    clipped_predictions = np.clip(predictions, -700, 700)  # np.expm1(700) adalah angka yang sangat besar
    return np.expm1(clipped_predictions)

if uploaded_file:
    # Muat file Excel dan sheet
    df = pd.read_excel(uploaded_file, sheet_name='Analisis partai Perindo')

    # Konversi 'Tanggal' ke format datetime dan jadikan index untuk pengelompokan bulanan
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    
    # Kelompokkan berdasarkan bulan dan agregat data reach berdasarkan gender dan rentang usia
    df_monthly = df.resample('M', on='Tanggal').sum()

    # Hapus outlier menggunakan Z-score
    df_monthly = df_monthly[(np.abs(stats.zscore(df_monthly.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

    # Tangani nilai yang hilang dengan forward filling
    df_monthly = df_monthly.fillna(method='ffill')

    # Tentukan rentang usia secara global
    age_ranges = ['Range Usia 18-24', 'Range Usia 25-34', 'Range Usia 35-44', 'Range Usia 45-54', 'Range Usia 55-64', 'Range Usia 65+']

    # Tambahkan fitur musiman
    df_monthly['Month_Index'] = np.arange(len(df_monthly))
    df_monthly['Month_Sin'] = np.sin(2 * np.pi * df_monthly['Month_Index'] / 12)
    df_monthly['Month_Cos'] = np.cos(2 * np.pi * df_monthly['Month_Index'] / 12)

    # Opsi di sidebar
    st.sidebar.title("Opsi")
    analysis_type = st.sidebar.selectbox(
        "Pilih jenis analisis",
        ["Lihat Data", "Visualisasi Data Historis", "Prediksi Reach menggunakan Random Forest"]
    )

    if analysis_type == "Lihat Data":
        # Tampilkan data yang diunggah
        st.subheader("Data yang Diunggah")
        st.dataframe(df)  # Tampilkan data asli yang diunggah dari Excel
        st.subheader("Data yang Digunakan untuk Analisis")
        st.dataframe(df_monthly)  # Tampilkan DataFrame yang digunakan untuk analisis

    elif analysis_type == "Visualisasi Data Historis":
        # Visualisasi Reach Aktual berdasarkan Gender
        st.subheader("Reach Bulanan Berdasarkan Gender (Historis)")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=df_monthly[['Gender (L)', 'Gender (P)']], ax=ax, palette={'Gender (L)': 'blue', 'Gender (P)': 'red'})
        ax.set_title('Reach Bulanan Berdasarkan Gender (Historis)')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Reach')
        ax.legend(title='Gender')
        st.pyplot(fig)

        # Kesimpulan singkat untuk gender
        mean_gender_reach = df_monthly[['Gender (L)', 'Gender (P)']].mean()
        max_reach_gender = mean_gender_reach.idxmax()
        st.write(f"Kesimpulan: Gender dengan rata-rata reach bulanan tertinggi adalah '{max_reach_gender}'. Ini menunjukkan bahwa gender tersebut memiliki dampak yang lebih konsisten dalam engagement.")

        # Visualisasi Reach Aktual berdasarkan Rentang Usia
        st.subheader("Reach Bulanan Berdasarkan Rentang Usia (Historis)")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=df_monthly[age_ranges], ax=ax, palette='tab10')
        ax.set_title('Reach Bulanan Berdasarkan Rentang Usia (Historis)')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Reach')
        ax.legend(title='Rentang Usia')
        st.pyplot(fig)

        # Kesimpulan singkat untuk rentang usia
        mean_age_reach = df_monthly[age_ranges].mean()
        max_reach_age = mean_age_reach.idxmax()
        st.write(f"Kesimpulan: Rentang usia dengan rata-rata reach bulanan tertinggi adalah '{max_reach_age}'. Ini menunjukkan bahwa kelompok usia tersebut menunjukkan engagement yang lebih tinggi terhadap konten.")

        # Visualisasi Dampak Hot News terhadap Reach
        st.subheader("Dampak Hot News terhadap Reach")
        # Agregat reach untuk setiap hot news
        hot_news_reach = {}
        for idx, row in df.iterrows():
            news_list = row['Hot News'].split(' | ')
            reach = row['Reach']  # Total reach untuk entri tersebut
            for news in news_list:
                if news in hot_news_reach:
                    hot_news_reach[news] += reach / len(news_list)
                else:
                    hot_news_reach[news] = reach / len(news_list)

        # Konversi ke DataFrame
        hot_news_reach_df = pd.DataFrame(list(hot_news_reach.items()), columns=['Hot News', 'Reach'])
        hot_news_reach_df = hot_news_reach_df.sort_values(by='Reach', ascending=False)

        # Plot Reach Hot News
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=hot_news_reach_df, y='Hot News', x='Reach', ax=ax, palette='viridis')
        ax.set_title('Dampak Hot News terhadap Reach')
        ax.set_xlabel('Reach')
        ax.set_ylabel('Hot News')
        st.pyplot(fig)

        # Kesimpulan singkat untuk hot news
        top_hot_news = hot_news_reach_df.iloc[0]
        st.write(f"Kesimpulan: Hot News yang memiliki dampak terbesar terhadap reach adalah '{top_hot_news['Hot News']}' dengan total reach tertinggi. Ini menunjukkan bahwa topik tersebut memiliki daya tarik yang signifikan terhadap audiens.")

    elif analysis_type == "Prediksi Reach menggunakan Random Forest":
        # 1. Pembersihan dan Normalisasi Data
        # Siapkan variabel lag untuk model
        df_monthly['Gender_L_Lag1'] = df_monthly['Gender (L)'].shift(1)
        df_monthly['Gender_P_Lag1'] = df_monthly['Gender (P)'].shift(1)
        for age_range in age_ranges:
            df_monthly[f'{age_range}_Lag1'] = df_monthly[age_range].shift(1)
        df_monthly.dropna(inplace=True)  # Hapus nilai NaN akibat lag

        # Pisahkan fitur dan target untuk gender
        X_gender = df_monthly[['Gender_L_Lag1', 'Gender_P_Lag1', 'Month_Sin', 'Month_Cos']]
        y_gender = df_monthly[['Gender (L)', 'Gender (P)']]

        # Pisahkan fitur dan target untuk rentang usia
        X_age = df_monthly[[f'{age_range}_Lag1' for age_range in age_ranges] + ['Month_Sin', 'Month_Cos']]
        y_age = df_monthly[age_ranges]

        # Normalisasi fitur
        scaler_features_gender = StandardScaler()
        X_gender_scaled = scaler_features_gender.fit_transform(X_gender)

        scaler_features_age = StandardScaler()
        X_age_scaled = scaler_features_age.fit_transform(X_age)

        # 2. Pelatihan Model
        # Latih Random Forest Regressor untuk gender
        model_gender = RandomForestRegressor(n_estimators=100, random_state=42)
        model_gender.fit(X_gender_scaled, y_gender)  # Latih pada target asli tanpa transformasi log

        # Latih Random Forest Regressor untuk rentang usia
        model_age = RandomForestRegressor(n_estimators=100, random_state=42)
        model_age.fit(X_age_scaled, y_age)  # Latih pada target asli tanpa transformasi log

        # 3. Prediksi
        # Prediksi 12 bulan ke depan secara iteratif untuk gender
        future_dates = pd.date_range(start=df_monthly.index[-1], periods=13, freq='M')[1:]
        predictions_gender = []

        # Gunakan nilai lag yang diketahui terakhir untuk inisialisasi prediksi
        last_known_values_lag = X_gender_scaled[-1, :].reshape(1, -1)  # Gunakan semua fitur
        for i in range(12):
            # Perbarui fitur musiman
            month_index = len(df_monthly) + i
            month_sin = np.sin(2 * np.pi * month_index / 12)
            month_cos = np.cos(2 * np.pi * month_index / 12)

            # Gabungkan nilai lag terakhir dengan fitur musiman
            next_input = np.concatenate([last_known_values_lag.flatten()[:-2], [month_sin, month_cos]])  # Sertakan nilai lag dan musiman
            next_input_scaled = scaler_features_gender.transform([next_input])  # Skala input dengan scaler utama
            next_prediction = model_gender.predict(next_input_scaled)

            predictions_gender.append(next_prediction.flatten())

            # Perbarui last_known_values_lag dengan nilai yang baru diprediksi
            last_known_values_lag = next_input_scaled

        # Buat DataFrame untuk prediksi gender
        df_future_gender = pd.DataFrame(predictions_gender, columns=['Gender (L)', 'Gender (P)'], index=future_dates)

        # Prediksi 12 bulan ke depan secara iteratif untuk rentang usia
        predictions_age = []
        last_known_values_lag = X_age_scaled[-1, :].reshape(1, -1)  # Gunakan semua fitur
        for i in range(12):
            # Perbarui fitur musiman
            month_index = len(df_monthly) + i
            month_sin = np.sin(2 * np.pi * month_index / 12)
            month_cos = np.cos(2 * np.pi * month_index / 12)

            # Gabungkan nilai lag terakhir dengan fitur musiman
            next_input = np.concatenate([last_known_values_lag.flatten()[:-2], [month_sin, month_cos]])  # Sertakan nilai lag dan musiman
            next_input_scaled = scaler_features_age.transform([next_input])  # Skala input dengan scaler utama
            next_prediction = model_age.predict(next_input_scaled)

            predictions_age.append(next_prediction.flatten())

            # Perbarui last_known_values_lag dengan nilai yang baru diprediksi
            last_known_values_lag = next_input_scaled

        # Buat DataFrame untuk prediksi rentang usia
        df_future_age = pd.DataFrame(predictions_age, columns=age_ranges, index=future_dates)

        # 4. Visualisasi Prediksi dan kesimpulan
        # Visualisasi Prediksi Reach Berdasarkan Gender menggunakan Random Forest
        st.subheader("Prediksi Reach Bulanan Berdasarkan Gender untuk 12 Bulan Mendatang")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=df_future_gender, ax=ax, palette={'Gender (L)': 'blue', 'Gender (P)': 'red'})
        ax.set_title('Prediksi Reach Bulanan Berdasarkan Gender untuk 12 Bulan Mendatang')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Reach')
        ax.legend(title='Gender')
        st.pyplot(fig)

        # Kesimpulan berdasarkan prediksi gender
        max_reach_gender = df_future_gender.mean().idxmax()
        st.write(f"Kesimpulan: Berdasarkan data prediksi, Gender '{max_reach_gender}' diperkirakan memiliki reach tertinggi dalam 12 bulan mendatang.")

        # Visualisasi Prediksi Reach Berdasarkan Rentang Usia menggunakan Random Forest
        st.subheader("Prediksi Reach Bulanan Berdasarkan Rentang Usia untuk 12 Bulan Mendatang")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=df_future_age, ax=ax, palette='tab10')
        ax.set_title('Prediksi Reach Bulanan Berdasarkan Rentang Usia untuk 12 Bulan Mendatang')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Reach')
        ax.legend(title='Rentang Usia')
        st.pyplot(fig)

        # Kesimpulan berdasarkan prediksi rentang usia
        max_reach_age = df_future_age.mean().idxmax()
        st.write(f"Kesimpulan: Rentang usia '{max_reach_age}' diperkirakan memiliki reach tertinggi dalam 12 bulan mendatang.")

        # 5. Evaluasi Model
        # Hitung Mean Squared Error untuk gender
        y_pred_gender = model_gender.predict(X_gender_scaled)
        mse_gender = mean_squared_error(y_gender, y_pred_gender)

        # Hitung Mean Squared Error untuk rentang usia
        y_pred_age = model_age.predict(X_age_scaled)
        mse_age = mean_squared_error(y_age, y_pred_age)

        # Tampilkan MSE
        st.write("Mean Squared Error (Reach Berdasarkan Gender):", mse_gender)
        st.write("Mean Squared Error (Reach Berdasarkan Rentang Usia):", mse_age) 