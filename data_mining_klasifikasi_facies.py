import pandas as pd
import numpy as np

# =========================================================
# 1. DATA LOADING
# =========================================================
df = pd.read_csv("facies_vectors.csv")
print("Pandas versi:", pd.__version__)
print("Jumlah baris & kolom:", df.shape)
print(df.head())

# =========================================================
# 2. EKSPLORASI DATA
# =========================================================
print("\n=== INFO DATASET ===")
df.info()

# Target (Facies) berupa kode angka 1-9, tetapi sebenarnya adalah KATEGORI
# kelas batuan, jadi dipisahkan dari kolom numerikal fitur.
target = "Facies"

# 2.1 Bagi jenis data: kategorikal & numerikal
kolom_kategorikal = df.select_dtypes(exclude="number").columns.tolist()
kolom_numerik = [k for k in df.select_dtypes(include="number").columns if k != target]
print("\nKolom numerikal (fitur)  :", kolom_numerik)
print("Kolom kategorikal (teks) :", kolom_kategorikal)
print("Kolom target             :", target, "(kode kelas 1-9)")

# 2.2 Tipe data kolom kategorikal (object / str, bukan float / void)
print("\nTipe data kolom kategorikal:")
print(df[kolom_kategorikal].dtypes)

# 2.3 Jumlah data duplikat & null
print("\nJumlah data duplikat:", df.duplicated().sum())
print("\nJumlah nilai kosong (null) per kolom:")
print(df.isnull().sum())

# Statistik deskriptif & sebaran target
print("\n=== STATISTIK DESKRIPTIF ===")
print(df[kolom_numerik].describe().T)
print("\nSebaran target (Facies):")
print(df[target].value_counts().sort_index())

# =========================================================
# 3. DATA CLEANING
# =========================================================
# 3.1 Hapus data duplikat
sebelum = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplikat dihapus: {sebelum - len(df)} baris (sisa {len(df)} baris)")

# 3.2 Handle null data numerik -> isi dengan median
for kolom in kolom_numerik:
    jumlah_null = df[kolom].isnull().sum()
    if jumlah_null > 0:
        df[kolom] = df[kolom].fillna(df[kolom].median())
        print(f"Null di {kolom}: {jumlah_null} diisi median")

# 3.3 Handle null data kategorikal -> isi modus (kalau ada), lalu encode
for kolom in kolom_kategorikal:
    if df[kolom].isnull().sum() > 0:
        df[kolom] = df[kolom].fillna(df[kolom].mode()[0])

# Well Name hanya identitas sumur, bukan sifat batuan -> tidak dipakai sebagai fitur
df = df.drop(columns="Well Name")

# Formation (nama lapisan batuan) di-encode menjadi angka
peta_formation = {nama: i for i, nama in enumerate(sorted(df["Formation"].unique()))}
df["Formation"] = df["Formation"].map(peta_formation)
print("\nEncoding Formation:", peta_formation)

# 3.4 Handle outlier dengan metode IQR (nilai di luar batas dibatasi ke batas)
for kolom in ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]:
    q1, q3 = df[kolom].quantile(0.25), df[kolom].quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    jumlah_outlier = ((df[kolom] < bawah) | (df[kolom] > atas)).sum()
    df[kolom] = df[kolom].clip(lower=bawah, upper=atas)
    print(f"Outlier di {kolom}: {jumlah_outlier} (dibatasi ke {bawah:.2f} - {atas:.2f})")

print("\nSisa nilai kosong setelah cleaning:", int(df.isnull().sum().sum()))
print(df.head())

# =========================================================
# 4. MODELING: DECISION TREE
# =========================================================
fitur = ["Formation", "Depth", "GR", "ILD_log10", "DeltaPHI",
         "PHIND", "PE", "NM_M", "RELPOS"]
X = df[fitur]
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=6, random_state=123)
tree_model = tree_model.fit(X_train, y_train)

# =========================================================
# 5. EVALUASI
# =========================================================
from sklearn.metrics import accuracy_score, classification_report
y_pred = tree_model.predict(X_test)
acc_score = round(accuracy_score(y_test, y_pred), 3)
print("\nAccuracy:", acc_score)
print(classification_report(y_test, y_pred, zero_division=0))

# =========================================================
# 6. PREDIKSI DATA BARU
# =========================================================
# Contoh data log sumur baru pada kedalaman 2900 ft
data_baru = pd.DataFrame(
    [[peta_formation["B1 SH"], 2900, 80, 0.7, 6, 12, 3.5, 1, 0.8]],
    columns=fitur)
print("Prediksi Facies untuk data baru:", tree_model.predict(data_baru)[0])

# =========================================================
# 7. EXPORT POHON KEPUTUSAN (.dot)
# =========================================================
from sklearn.tree import export_graphviz
export_graphviz(
    tree_model,
    out_file="facies_tree.dot",
    feature_names=fitur,
    class_names=[str(k) for k in sorted(y.unique())],
    rounded=True,
    filled=True)
print("File facies_tree.dot berhasil dibuat")
