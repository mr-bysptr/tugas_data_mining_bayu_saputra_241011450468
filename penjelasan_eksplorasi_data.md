# Klasifikasi Fasies Batuan dari Data Well Log (Lapangan Gas Hugoton dan Panoma, Kansas)

## 1. Sumber dan Tujuan Dataset

Dataset yang dipakai adalah data well log (facies_vectors.csv) dari kontes machine learning SEG 2016 (https://github.com/seg/2016-ml-contest). Data ini berasal dari latihan kelas University of Kansas pada lapangan gas Hugoton dan Panoma, mengacu pada Dubois et al. (2007). Dataset berisi 4.149 baris dan 11 kolom, mencakup 10 sumur, dengan satu baris untuk setiap interval kedalaman setengah kaki.

Dalam industri migas, jenis batuan (fasies) perlu diketahui untuk menentukan lapisan yang menyimpan hidrokarbon. Cara manual dengan inti batuan (core) mahal dan lama, sedangkan data log sumur sudah tersedia di setiap pemboran. Karena itu, dataset ini dipakai untuk melatih model Decision Tree agar dapat memprediksi jenis batuan dari hasil pengukuran log.

Target data (label): Facies, terdiri dari 9 kelas.
Jenis masalah: klasifikasi.

Daftar kelas target:
1 = Nonmarine sandstone (SS)
2 = Nonmarine coarse siltstone (CSiS)
3 = Nonmarine fine siltstone (FSiS)
4 = Marine siltstone and shale (SiSh)
5 = Mudstone (MS)
6 = Wackestone (WS)
7 = Dolomite (D)
8 = Packstone-grainstone (PS)
9 = Phylloid-algal bafflestone (BS)

## 2. Deskripsi Kolom

- Facies (target): kode kelas batuan 1-9
- Formation (kategorikal): nama lapisan batuan, 14 nilai (contoh A1 SH, B1 LM)
- Well Name (kategorikal): nama sumur
- Depth (numerik): kedalaman
- GR (numerik): log gamma ray
- ILD_log10 (numerik): resistivitas dalam skala log10
- DeltaPHI (numerik): selisih porositas neutron-density
- PHIND (numerik): rata-rata porositas neutron-density
- PE (numerik): log photoelectric effect
- NM_M (numerik): indikator nonmarine (1) atau marine (2)
- RELPOS (numerik): posisi relatif lapisan

## 3. Data Loading

Data dibaca dengan pd.read_csv("facies_vectors.csv"). Selain CSV, pandas juga dapat membaca format lain seperti read_excel, read_json, read_html, read_xml, dan read_sql. Setelah dimuat, head() dipakai untuk melihat 5 baris awal dan shape untuk melihat jumlah baris dan kolom.

## 4. Eksplorasi Data

1. Pembagian jenis data
Data numerikal (fitur): Depth, GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS.
Data kategorikal: Formation dan Well Name.
Facies disimpan sebagai angka 1-9, tetapi sebenarnya adalah label kelas (kategori), sehingga dipisahkan sebagai target dan tidak dihitung sebagai fitur numerikal.
Pembagian dilakukan dengan select_dtypes(include="number") dan select_dtypes(exclude="number").

2. Tipe data untuk data kategorikal
Data kategorikal (teks) disimpan sebagai tipe object, bukan float maupun void. Float dipakai untuk angka desimal. Pada pandas versi 3.x, kolom teks ditampilkan sebagai tipe str, yang fungsinya sama dengan object pada versi sebelumnya. Karena itu pemisahan dilakukan dengan exclude="number" agar tetap aman di kedua versi.

3. Jumlah data duplikat dan null
Duplikat dicek dengan df.duplicated().sum(), hasilnya 1 baris. Nilai kosong dicek dengan df.isnull().sum(), hasilnya hanya kolom PE yang memiliki nilai kosong, yaitu 917 baris. Kolom lain, termasuk kolom teks, tidak memiliki nilai kosong.

Statistik deskriptif (describe()) menunjukkan sebaran yang tidak merata pada beberapa kolom, misalnya GR yang nilai maksimumnya jauh dari kuartil ketiga. Sebaran target dengan value_counts() menunjukkan kelas 2 paling banyak (940 baris) dan kelas 7 paling sedikit (141 baris).

## 5. Data Cleaning

1. Duplikat dihapus dengan drop_duplicates(), tersisa 4.148 baris.
2. Nilai kosong pada kolom PE (917 baris) diisi dengan median kolom, karena median tidak terpengaruh nilai ekstrem.
3. Tidak ada nilai kosong pada kolom teks. Jika ada, nilainya akan diisi modus. Kolom Formation di-encode menjadi angka 0-13 secara alfabet. Kolom Well Name dibuang karena hanya identitas sumur dan tidak menggambarkan sifat batuan.
4. Outlier ditangani dengan metode IQR. Nilai di bawah Q1 - 1,5 x IQR atau di atas Q3 + 1,5 x IQR dibatasi ke batas tersebut (capping). Jumlah outlier: GR 89, ILD_log10 63, DeltaPHI 161, PHIND 170, PE 250.

## 6. Pelatihan dan Evaluasi

Data dibagi 80 persen untuk latih dan 20 persen untuk uji (train_test_split, random_state=123). Model yang dipakai adalah DecisionTreeClassifier dengan max_depth=6 agar pohon tidak terlalu rumit. Akurasi dihitung dengan accuracy_score dan hasilnya sekitar 0,60 (60 persen). Akurasi ini wajar untuk 9 kelas pada data nyata, karena beberapa jenis batuan memiliki karakteristik log yang mirip (misalnya kelas 5 dan 6), sehingga sering tertukar. Model kemudian dipakai memprediksi satu data log baru, dan pohon keputusan diekspor ke file facies_tree.dot menggunakan export_graphviz, lalu dikonversi ke PNG lewat Graphviz.

## 7. Catatan

Pembagian data latih dan uji dilakukan secara acak per baris, sehingga baris yang berdekatan pada sumur yang sama bisa berada di kedua bagian. Hal ini dapat membuat akurasi sedikit lebih tinggi dibanding pengujian pada sumur yang benar-benar baru.
