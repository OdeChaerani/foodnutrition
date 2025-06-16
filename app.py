# Project: Klasifikasi Makanan Berdasarkan Makronutrien + UI & Rekomendasi Menu
# Dataset: Indonesian Food and Drink Nutrition Dataset (Kaggle)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# Load dataset
# Gunakan file yang telah diunggah: nutrition.csv
df = pd.read_csv("nutrition.csv")

# Tampilkan kolom-kolom yang tersedia (untuk debug awal)
# st.write("Kolom tersedia:", df.columns.tolist())

# Sesuaikan nama kolom berdasarkan file sebenarnya
# Rename kolom jika perlu untuk konsistensi
rename_cols = {
    'Name': 'name',
    'Calories': 'calories',
    'Protein(g)': 'protein (g)',
    'Fat(g)': 'fat (g)',
    'Carbohydrate(g)': 'carbohydrate (g)'
}
df.rename(columns=rename_cols, inplace=True)

# Buat label berdasarkan makronutrien dominan
def label_makro(row):
    makro = {
        'protein': row['protein (g)'],
        'fat': row['fat (g)'],
        'carbohydrate': row['carbohydrate (g)']
    }
    return max(makro, key=makro.get)

df['label'] = df.apply(label_makro, axis=1)

# Pilih fitur dan label
X = df[['calories', 'protein (g)', 'fat (g)', 'carbohydrate (g)']]
y = df['label']

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model klasifikasi
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)

# STREAMLIT UI
st.title("Klasifikasi dan Rekomendasi Menu Makanan Indonesia")

st.subheader("1. Prediksi Tipe Makanan Berdasarkan Nilai Gizi")
cal = st.number_input("Kalori", min_value=0.0)
pro = st.number_input("Protein (g)", min_value=0.0)
fat = st.number_input("Lemak (g)", min_value=0.0)
carb = st.number_input("Karbohidrat (g)", min_value=0.0)

if st.button("Prediksi Tipe Makronutrien Dominan"):
    input_scaled = scaler.transform([[cal, pro, fat, carb]])
    pred = model.predict(input_scaled)[0]
    st.success(f"Makronutrien dominan: {pred.title()}")

st.subheader("2. Rekomendasi Menu Berdasarkan Kebutuhan Kalori Harian")
target_cal = st.slider("Pilih target kalori (kcal)", 100, 2000, step=100)

if st.button("Tampilkan Rekomendasi"):
    df_sorted = df.copy()
    df_sorted['selisih_kalori'] = abs(df_sorted['calories'] - target_cal)
    rekomendasi = df_sorted.sort_values('selisih_kalori').head(5)
    st.write("Menu dengan kalori mendekati target:")
    st.dataframe(rekomendasi[['name', 'calories', 'protein (g)', 'fat (g)', 'carbohydrate (g)', 'label']])

# Optional: visualisasi offline (jika tidak dalam mode Streamlit)
if __name__ == '__main__':
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualisasi distribusi
    sns.pairplot(df, hue="label", vars=['calories', 'protein (g)', 'fat (g)', 'carbohydrate (g)'])
    plt.suptitle("Distribusi Makanan Berdasarkan Makronutrien Dominan", y=1.02)
    plt.show()