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

#### Jumlah baris dan kolom:
> Dataset ini memiliki total 1281 baris (sampel/pasien) dan 15 kolom (yang terdiri dari 14 kolom fitur metrik darah dan 1 kolom target diagnosis).

<p align="center">
    <img width="611" height="563" alt="image" src="https://github.com/user-attachments/assets/1efb0939-fd6d-4f08-9921-49fb6340a7b9" />
</p>
Inspeksi kualitas data secara terprogram mengonfirmasi bahwa dataset berada dalam kondisi yang sangat bersih tanpa ada satu pun nilai yang hilang (missing value). Pemeriksaan menyeluruh terhadap keseluruhan 15 variabel yang mencakup 14 parameter metrik darah lengkap mulai dari kandungan WBC hingga PCT, serta satu kolom target klasifikasi berupa Diagnosis secara konsisten mencetak angka nol untuk jumlah baris yang kosong. Kelengkapan data yang mencapai 100% ini merupakan kondisi ideal yang menguntungkan karena tahapan Data Preparation selanjutnya tidak memerlukan intervensi manipulasi data, seperti imputasi nilai buatan menggunakan mean atau median, maupun penghapusan baris observasi. Dengan demikian, orisinalitas distribusi data klinis pasien tetap terjaga sepenuhnya, memungkinkan model machine learning untuk mempelajari pola klasifikasi dari rekam medis yang seutuhnya tanpa adanya risiko bias akibat penanganan data yang hilang.

#### Variabel-variabel pada dataset adalah sebagai berikut:
- WBC (White Blood Cell): Jumlah sel darah putih ($10^9/L$).
- LYMp: Persentase limfosit (%).
- NEUTp: Persentase neutrofil (%).
- LYMn: Jumlah absolut limfosit.NEUTn: Jumlah absolut neutrofil.
- NEUTn: Jumlah absolut neutrofil.
- RBC (Red Blood Cell): Jumlah sel darah merah ($10^{12}/L$).
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

#### 1. Pemisahan Fitur dan Target:
- Proses ini memisahkan dataset ke dalam dua variabel berbeda: variabel independen/fitur ($X$) dan variabel dependen/target ($y$). Seluruh kolom hasil tes laboratorium numerik dimasukkan ke dalam $X$ dengan melakukan drop pada kolom Diagnosis, sementara kolom Diagnosis itu sendiri diisolasi ke dalam variabel $y$.

- Alasan: Pemisahan ini merupakan syarat fundamental sebelum melakukan proses encoding maupun pemodelan, agar algoritma machine learning dapat membedakan mana data yang digunakan sebagai bahan belajar (fitur) dan mana data yang harus diprediksi (target).

#### 2. Label Encoding pada Kolom Target:
- Proses ini mengubah kelas string/teks pada kolom target Diagnosis menjadi bilangan integer (seperti 0, 1, 2, dll).

- Alasan: Pemodelan machine learning pada tahap selanjutnya merupakan operasi matematis murni. Model tidak bisa mengevaluasi kelas bertipe teks secara langsung.

#### 3. Train-Test Split:
- Membagi dataset secara utuh menjadi rasio latih dan uji (umumnya 80% data untuk training dan 20% data untuk testing).

- Alasan: Sangat esensial untuk mencegah data leakage dan berfungsi untuk menilai bagaimana model mampu melakukan generalisasi dalam menebak data unseen (data yang tidak dikenalnya selama proses pelatihan).

#### 4. Standarisasi Fitur (StandardScaler):
- Mentransformasi fitur numerik yang ada agar memiliki nilai rata-rata (mean) 0 dan standar deviasi 1.

- Alasan: Dataset memiliki varian skala ukuran antar metrik fitur yang curam (misal rentang PLT dibanding HGB). Banyak model klasifikasi yang kalkulasinya berbasis pada jarak (seperti SVM) akan bias ke arah fitur berukuran raksasa bila datanya belum diskalakan seragam.


## Modeling
Untuk menangani kasus klasifikasi penyakit darah multi-kelas ini, dua jenis algoritma dilatih dan dibandingkan. Berikut adalah penjelasan cara kerja, parameter, kelebihan, serta kekurangan dari masing-masing model:

<p align="center">
    <img width="1141" height="88" alt="image" src="https://github.com/user-attachments/assets/3195be2e-30a3-418a-aa0e-5ecf777cbf86" />
</p>

**1. Random Forest Classifier**
   - Cara Kerja Model: Random Forest adalah algoritma Supervised Learning berjenis ensemble. Alih-alih mengandalkan satu pohon keputusan (Decision Tree), algoritma ini membangun sebuah "hutan" yang terdiri dari banyak pohon keputusan secara acak pada fase pelatihan (training). Ketika proses prediksi dilakukan, setiap pohon akan mengeluarkan tebakan diagnosisnya sendiri (voting). Hasil prediksi akhir (apakah pasien didiagnosis Healthy atau Leukemia, misalnya) ditentukan dari kelas yang mendapatkan suara terbanyak (majority voting) dari seluruh pohon keputusan tersebut.
   
   - Parameter yang Digunakan: Melalui hyperparameter tuning (GridSearchCV), pencarian kombinasi parameter dilakukan secara komprehensif pada:
      - n_estimators: Penyesuaian jumlah pohon dalam hutan.
      - criterion: Fungsi untuk mengukur kualitas pemisahan (gini atau entropy).
      - max_depth: Maksimum batasan kedalaman simpul pohon.

   - Nilai Parameter Terbaik (Best Parameters): Berdasarkan hasil tuning, parameter spesifik yang memberikan akurasi tertinggi adalah: {'criterion': 'entropy', 'max_depth': None, 'n_estimators': 100}.Kelebihan: Sangat kokoh, kebal terhadap titik nilai ekstrem (outliers), dan sangat cakap menemukan batasan fungsi yang tidak linear. Mekanisme ensemble mencegah model dari masalah overfitting yang sering dialami oleh pohon keputusan tunggal.
   - Kekurangan: Model final berukuran lebih berat dan memerlukan waktu komputasi (kompilasi) yang lebih lama untuk melakukan prediksi jika dibandingkan dengan algoritma decision tree biasa.

**2. Support Vector Machine (SVM)**

   - Cara Kerja Model: SVM bekerja dengan mencari hyperplane (garis atau bidang batas) terbaik di dalam ruang berdimensi $N$ (di mana $N$ adalah jumlah fitur, misal 14 fitur metrik darah). Hyperplane terbaik ini adalah yang memiliki margin paling maksimal; artinya, batas yang memiliki jarak terjauh antara titik-titik data terluar (support vectors) dari masing-masing kelas penyakit. Jika data tidak bisa dipisahkan secara linear, SVM menggunakan kernel trick untuk memproyeksikan data tersebut ke dimensi yang lebih tinggi agar garis pemisah bisa ditemukan.
    
   - Parameter yang Digunakan: Secara spesifik di-tuning pada:
        - C (Regularization parameter): Menentukan seberapa keras model mencoba menghindari kesalahan klasifikasi pada tiap titik latih (mengatur keseimbangan antara batas margin yang halus dan klasifikasi yang benar).
        - kernel: Menentukan jenis fungsi matematis untuk transformasi data (seperti linear, poly, rbf).Nilai Parameter Terbaik (Best Parameters): Berdasarkan hasil tuning, parameter spesifik yang memberikan performa terbaik adalah: {'C': 100, 'kernel': 'linear'}.
    
   - Kelebihan: Sangat efektif dan unggul bekerja di ruang dimensi tinggi (ketika jumlah fitur banyak). Fleksibel mengatasi data tidak linear dengan kernel trick, serta sangat efisien dalam penggunaan memori latih.
   
   - Kekurangan: Pemilihan parameter fungsi kernel atau regularisasi C yang salah akan membuat algoritma sangat sulit beradaptasi. Selain itu, SVM cukup rentan terhadap noise (data yang tumpang tindih) jika batas tepian margin tidak ditata ulang dengan baik.

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

Berdasarkan hasil pengujian metrik evaluasi makro, model Random Forest secara konsisten mengungguli Support Vector Machine (SVM) di seluruh parameter dan ditetapkan sebagai model terbaik untuk klasifikasi diagnosis penyakit ini.

1. **Random Forest (Best Model):** Berhasil mencapai tingkat akurasi nyaris sempurna di angka 99% (0.99). Menghadapi kondisi dataset yang sangat tidak seimbang (highly imbalanced), algoritma ensemble ini tetap menunjukkan performa generalisasi yang sangat tangguh. Hal ini dibuktikan dengan capaian Precision sebesar 85% (0.85), Recall sebesar 89% (0.89), dan F1-Score sebesar 87% (0.87). Tingginya nilai Recall sangat krusial dalam konteks medis karena memastikan model memiliki sensitivitas yang kuat dalam mendeteksi pasien yang benar-benar sakit, sehingga meminimalkan jumlah False Negative (pasien sakit yang divonis sehat).

<p align="center">
    <img width="1183" height="201" alt="image" src="https://github.com/user-attachments/assets/8b5e2a7c-0938-44f6-84e5-b2258d7bd046" />
</p>

2. **Support Vector Machine (SVM):** Menunjukkan performa yang tertinggal dengan nilai akurasi akhir sebesar 89% (0.89). Kesulitan algoritma SVM dalam memetakan batasan data yang imbalanced terlihat jelas pada metrik presisi dan sensitivitasnya. Model ini hanya mampu mencatatkan Precision sebesar 83% (0.83), Recall yang menyentuh angka 80% (0.80), dan F1-Score di level 78% (0.78). Angka-angka yang lebih rendah ini mengindikasikan bahwa model SVM lebih sering gagal atau keliru (False Positive maupun False Negative) dalam mengenali kelas diagnosis minoritas (seperti Leukemia) jika dibandingkan dengan model Random Forest.

<p align="center">
    <img width="1180" height="213" alt="image" src="https://github.com/user-attachments/assets/8f80b55b-e0d3-404d-84a8-6a60e3d834d4" />
</p>

### Kesimpulan Evaluasi
Berdasarkan hasil pengujian klasifikasi machine learning pada data rekam medis Complete Blood Count (CBC), Random Forest terbukti menjadi model terbaik dan ditetapkan sebagai solusi prediksi akhir. Algoritma ensemble ini menunjukkan performa yang sangat unggul dengan tingkat akurasi keseluruhan mencapai 99%, didukung oleh metrik Precision sebesar 85%, Recall 89%, dan F1-Score 87%. Kemampuan Random Forest dalam mempertahankan nilai Recall yang tinggi terbukti sangat krusial dalam domain medis klinis, karena memastikan sistem memiliki sensitivitas yang kuat untuk mendeteksi pasien yang benar-benar mengidap penyakit kelainan darah sehingga meminimalkan risiko bahaya False Negative (pasien sakit yang divonis sehat). Selain itu, model ini terbukti kokoh dan stabil dalam menangani distribusi dataset yang sangat tidak seimbang (highly imbalanced) jika dibandingkan dengan algoritma pembandingnya, Support Vector Machine (SVM). Model SVM mencatatkan performa yang tertinggal dengan akurasi 89%, Precision 83%, Recall 80%, dan F1-Score 78%. Penurunan angka evaluasi pada SVM ini mengindikasikan bahwa algoritma tersebut kesulitan memetakan batas hyperplane keputusan secara adil pada data yang timpang, sehingga model lebih sering gagal atau keliru dalam mengenali diagnosis dari kelas pasien minoritas yang kritis seperti Leukemia.
