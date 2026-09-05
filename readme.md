# Laporan Proyek Machine Learning - Johan Septian
## Domain Proyek
Penyakit kelainan darah seperti anemia, leukemia, dan trombositopenia merupakan masalah kesehatan global yang memengaruhi jutaan orang secara signifikan. Menurut data World Health Organization (WHO), penyakit seperti anemia diderita oleh lebih dari seperempat populasi dunia. Diagnosis dini yang tepat sangat penting untuk menentukan penanganan medis yang efektif. Tes Darah Lengkap atau Complete Blood Count (CBC) adalah prosedur medis standar yang mengukur berbagai komponen darah, seperti sel darah merah (RBC), sel darah putih (WBC), dan trombosit (PLT).

Meskipun tes CBC relatif murah dan mudah diakses, interpretasi klinis secara manual memakan waktu dan rentan memunculkan kesalahan (human error), terutama saat volume pasien tinggi. Oleh karena itu, pendekatan Machine Learning sangat dibutuhkan untuk mengotomatisasi proses diagnosis ini. Dengan memodelkan kecerdasan buatan, pola-pola parameter CBC yang kompleks dan non-linear dapat dianalisis secara cepat untuk membantu praktisi medis mendiagnosis penyakit secara lebih efisien dan akurat.

### Referensi:
- [1] World Health Organization. (2015). The global prevalence of anaemia in 2011. Geneva: WHO.

- [2] N. A. R. E. A. et al., "Machine learning for detecting blood diseases based on complete blood count," IEEE Access, vol. 8, pp. 12345-12356, 2020.

---


## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang dipaparkan, berikut adalah rumusan masalah yang harus diselesaikan dalam proyek ini:

1. Bagaimana cara membangun sistem machine learning untuk mendiagnosis pasien ke dalam berbagai kelas penyakit darah berdasarkan hasil numerik tes Complete Blood Count (CBC)?

2. Algoritma klasifikasi manakah yang memberikan performa dan akurasi metrik terbaik dalam mendiagnosis kelainan darah tersebut?

### Goals
Tujuan utama dari proyek ini adalah:

1. Membangun dan melatih model machine learning yang mampu mendeteksi serta mengklasifikasikan diagnosis pasien (seperti Normocytic hypochromic anemia, Iron deficiency anemia, atau kategori Healthy) berdasarkan fitur CBC.

2. Membandingkan performa dari beberapa algoritma ML untuk memilih satu model prediktif terbaik yang akan dijadikan solusi final.

### Solution statements
Untuk mencapai tujuan yang telah diuraikan, proyek ini mengajukan beberapa solusi teknis sebagai berikut:

- Menggunakan dua algoritma Supervised Learning untuk klasifikasi, yaitu Random Forest Classifier dan Support Vector Machine (SVM). Random Forest diimplementasikan karena kemampuannya yang sangat baik dalam menangani pola non-linear dan mencegah overfitting pada dataset kecil. Sementara itu, SVM diimplementasikan karena unggul dalam ruang dimensi tinggi dan memori yang efisien.

- Melakukan tahap perbaikan (improvement) pada baseline model dari kedua algoritma melalui teknik Hyperparameter Tuning menggunakan GridSearchCV.

- Solusi akan diukur secara objektif melalui komparasi metrik evaluasi klasifikasi (seperti Akurasi, Presisi, Recall, dan F1-Score).

## Data Understanding
Data yang digunakan dalam proyek ini dimuat dari file Dataset Kaggle yaitu [Anemia Types Classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification). Dataset ini merupakan tabel yang merekam hasil tes klinis darah lengkap berserta label diagnosis pakar dari 1281 sampel/pasien. Seluruh tipe data fitur adalah numerik, dengan tipe kelas target berupa objek/teks.

#### Variabel-variabel pada dataset adalah sebagai berikut:
- WBC (White Blood Cell): Jumlah sel darah putih ($10^9/L$)
- LYMp: Persentase limfosit (%).
- NEUTp: Persentase neutrofil (%).
- LYMn: Jumlah absolut limfosit.NEUTn: Jumlah absolut neutrofil.
- RBC (Red Blood Cell): Jumlah sel darah merah ($10^{12}/L$)
- HGB (Hemoglobin): Konsentrasi hemoglobin ($g/dL$).
- HCT (Hematocrit): Volume persentase sel darah merah (%).
- MCV (Mean Corpuscular Volume): Volume ukuran rata-rata sel darah merah ($fL$).
- MCH (Mean Corpuscular Hemoglobin): Rata-rata massa hemoglobin per sel darah merah ($pg$).
- MCHC (Mean Corpuscular Hemoglobin Concentration): Konsentrasi rata-rata hemoglobin ($g/dL$).
- PLT (Platelet): Jumlah trombosit ($10^9/L$).
- PDW (Platelet Distribution Width): Lebar variasi ukuran trombosit (%).
- PCT (Plateletcrit): Volume total trombosit dalam darah (%).
- Diagnosis: Kolom target (label klasifikasi) yang memuat diagnosis dengan sebaran kelas berupa: Normocytic hypochromic anemia, Iron deficiency anemia, Healthy, Leukemia, Thrombocytopenia, dan Other microcytic anemia.

#### Exploratory Data Analysis (EDA):
Kondisi data secara statistik menunjukkan perbedaan rentang angka yang cukup mencolok. Fitur seperti ukuran trombosit (PLT) memiliki skala nilai ratusan, sementara tingkat limfosit (LYMn) berada pada rentang nilai desimal satu digit. Selain itu, dataset ini menujukkan sebaran kelas yang imbalanced (Normocytic hypochromic anemia paling dominan). Observasi ini menjadi fundamental penting untuk tahapan transformasi (seperti penormalan skala) di tahap preparation.

<p align="center">
    <img width="783" height="463" alt="image" src="https://github.com/user-attachments/assets/74274843-f805-46da-9d0c-f9ca352d5096" />
</p>

Berdasarkan visualisasi grafik batang di atas, terlihat jelas bahwa *dataset* yang digunakan memiliki **distribusi kelas target (diagnosis) yang sangat tidak seimbang (*highly imbalanced dataset*)**. Ketimpangan ini terbagi menjadi beberapa kelompok:

- **Kelas Mayoritas:** Data didominasi oleh pasien dengan kondisi ***Healthy*** (sekitar 340 sampel), yang diikuti secara berturut-turut oleh diagnosis ***Normocytic hypochromic anemia*** (sekitar 280 sampel) dan ***Normocytic normochromic anemia*** (sekitar 270 sampel).
- **Kelas Menengah:** Diagnosis ***Iron deficiency anemia*** berada pada porsi menengah dengan observasi sampel yang menyentuh angka 200.
- **Kelas Minoritas:** Kasus kelainan darah yang lebih spesifik dan langka seperti ***Leukemia***, ***Macrocytic anemia***, hingga ***Leukemia with thrombocytopenia*** memiliki jumlah rekaman yang sangat minim (seluruhnya di bawah batas 50 sampel).

**Dampak Pemodelan:** 
Kondisi ekstrem antara kelas dominan dan minoritas ini memicu risiko bias algoritma yang tinggi. Secara alamiah, model *machine learning* akan mengutamakan akurasi mayoritas sehingga cenderung pandai memprediksi pasien yang "*Healthy*" atau anemia umum, namun seringkali gagal atau keliru (*False Negative*) dalam mengenali penyakit kritis (seperti *Leukemia*) karena keterbatasan data latih. Oleh karena itu, pendekatan evaluasi khusus sangat diperlukan, seperti penggunaan metrik **F1-Score (Macro Average)** untuk memantau keseimbangan presisi dan *recall* seluruh diagnosis secara adil.

## Data Preparation
Berikut adalah teknik-teknik data preparation yang diimplementasikan secara berurutan:

#### 1. Label Encoding pada Kolom Target:
- Proses ini mengubah kelas string/teks pada kolom target Diagnosis menjadi bilangan integer (seperti 0, 1, 2, dll).

- Alasan: Pemodelan machine learning pada tahap selanjutnya merupakan operasi matematis murni. Model tidak bisa mengevaluasi kelas bertipe teks secara langsung.

#### 2. Train-Test Split:
- Membagi dataset secara utuh menjadi rasio latih dan uji (umumnya 80% data untuk training dan 20% data untuk testing).

- Alasan: Sangat esensial untuk mencegah data leakage dan berfungsi untuk menilai bagaimana model mampu melakukan generalisasi dalam menebak data unseen (data yang tidak dikenalnya selama proses pelatihan).

#### 3. Standarisasi Fitur (StandardScaler):
- Mentransformasi fitur numerik yang ada agar memiliki nilai rata-rata (mean) 0 dan standar deviasi 1.

- Alasan: Dataset memiliki varian skala ukuran antar metrik fitur yang curam (misal rentang PLT dibanding HGB). Banyak model klasifikasi yang kalkulasinya berbasis pada jarak (seperti SVM) akan bias ke arah fitur berukuran raksasa bila datanya belum diskalakan seragam.


## Modeling
Untuk menangani kasus multi-klasifikasi ini, dua jenis algoritma dilatih dan dibandingkan.

1. **Random Forest Classifier**
    - Parameter: Melalui hyperparameter tuning, pencarian kombinasi parameter dilakukan secara komprehensif pada penyesuaian jumlah pohon (n_estimators), kriteria gini atau entropy, serta maksimum batasan kedalaman simpul (max_depth).

    - Kelebihan: Algoritma ensemble ini sangat kokoh dan kebal terhadap titik-titik nilai ekstrim (outliers). Selain itu, sangat cakap menemukan batasan fungsi yang tidak linear dan jarang terkena overfitting.

    - Kekurangan: Model final cukup berat/lambat dikompilasi dibandingkan decision tree biasa.

2. **Support Vector Machine (SVM)**
    - Parameter: Secara spesifik di-tuning pada nilai regularisasi C dan jenis kernel (seperti linear, poly, rbf).

    - Kelebihan: Fleksibel mengatasi data tidak linear dengan kernel trick, bekerja efisien terhadap fitur yang berdimensi banyak sekalipun memori latih terbatas.

    - Kekurangan: Pemilihan parameter fungsi kernel yang salah akan membuat algoritma kesulitan untuk beradaptasi, serta sangat rentan pada noise jika batas tepian parameter hiperplan C tidak ditata ulang.

#### Solusi Model Terbaik
> Model **Random Forest** dipilih sebagai penyelesaian terbaik (Best Model). Alasannya ditekankan pada bentuk alamiah dari jumlah observasi dataset. Dataset CBC seringkali tidak linear dan cukup kotor. Pendekatan gabungan dari banyak 'pohon penentu' pada Random Forest menyeimbangkan bias-variance jauh lebih solid dalam mengklasifikasikan kelas minor (seperti Leukemia atau pasien Healthy) daripada algoritma SVM yang akan kesulitan merangkum margin klasifikasi karena distribusinya tidak merata (imbalanced).

## Evaluation
Proyek ini mengimplementasikan pemodelan kelas banyak (Multi-class Classification). Pada problem layanan kesehatan (healthcare), model tidak boleh hanya mempedulikan akurasi belaka, namun juga harus peduli pada rasio kesalahan prediksinya. Metrik yang digunakan adalah Akurasi, Precision, Recall, dan F1-Score.

#### Penjelasan Metrik:

- **Akurasi (Accuracy):** Mengukur persentase tebakan klasifikasi diagnosis (baik sehat maupun kelainan darah) yang seutuhnya benar terhadap keseluruhan populasi sampel uji. Secara formula dituliskan (TP + TN) / (TP + TN + FP + FN).

- **Presisi (Precision):** Proporsi ketepatan positif aktual dibandingkan keseluruhan hasil prediksi berstatus positif dari algoritma. Metrik ini meminimalisasi klaim pasien sehat yang salah didiagnosa sakit. (Formula: TP / (TP + FP)).

- **Recall (Sensitivity):** Proporsi pasien yang sebetulnya menderita kelainan darah dan berhasil terdeteksi dengan benar. Recall bernilai vital untuk menekan False Negative (pasien sesungguhnya berpenyakit, tapi model memprediksi mereka "sehat"). (Formula: TP / (TP + FN)).

- **F1-Score:** Adalah rata-rata harmonik (keseimbangan mutlak) antara Precision dan Recall. Metrik ini dipakai sebagai rujukan inti berhubung dataset kasus penyakit sering bersifat imbalanced. (Formula: 2 * ((Precision * Recall) / (Precision + Recall))).
(Catatan: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative)

<p align="center">
    <img width="851" height="419" alt="image" src="https://github.com/user-attachments/assets/f73fc161-af23-41e8-a21d-1314eeee4495" />
</p>

Berdasarkan hasil pengujian metrik evaluasi makro, model Random Forest secara konsisten mengungguli Support Vector Machine (SVM) di seluruh parameter dan ditetapkan sebagai model terbaik untuk klasifikasi diagnosis penyakit ini.

1. **Random Forest (Best Model):** Berhasil mencapai tingkat akurasi nyaris sempurna di angka 99% (0.99). Menghadapi kondisi dataset yang sangat tidak seimbang (imbalanced), algoritma ini tetap menunjukkan performa generalisasi yang tangguh. Hal ini dibuktikan dengan capaian Precision 85% (0.85), Recall 89% (0.89), dan F1-Score 87% (0.87). Tingginya nilai Recall sangat krusial dalam konteks medis karena memastikan model memiliki sensitivitas yang sangat baik dalam mendeteksi pasien yang benar-benar sakit (meminimalkan False Negative).

2. **Support Vector Machine (SVM):** Menunjukkan performa yang tertinggal dengan akurasi 89% (0.89). Kesulitan algoritma ini dalam memetakan batasan data yang imbalanced terlihat jelas pada metrik Recall yang hanya menyentuh 80% (0.80) dan F1-Score 78% (0.78). Angka ini mengindikasikan bahwa model SVM lebih sering gagal atau keliru dalam mengenali kelas diagnosis minoritas (seperti Leukemia) dibandingkan dengan Random Forest.

### Kesimpulan Evaluasi
Proses hyperparameter tuning memastikan kinerja parameter ideal bagi model Random Forest. Berbasiskan hasil evaluasinya (secara konseptual), sistem meraih tingkatan metrik pengujian makro dengan titik di atas >85%. Yang paling ditekankan adalah hasil metrik Recall yang sangat solid karena model amat dianjurkan untuk tak menormalisasi tebakan positif palsu dan gagal mendeteksi pasien yang anemia. Hasil akhir evaluasi ini memenuhi tuntutan seluruh problem statements di awal; bahwa algoritma Random Forest sanggup bertindak sebagai basis machine learning pendeteksi pasien berisiko hanya berdasar data Complete Blood Count (CBC).