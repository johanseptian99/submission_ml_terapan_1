# Laporan Proyek Machine Learning - Johan Septian
## Domain Proyek
Penyakit kelainan darah seperti anemia, leukemia, dan trombositopenia merupakan masalah kesehatan global yang memengaruhi jutaan orang secara signifikan. Menurut World Health Organization (WHO), anemia merupakan masalah kesehatan masyarakat yang berdampak pada sebagian besar populasi dunia, terutama perempuan dan anak-anak [1]. Diagnosis dini yang tepat sangat penting untuk menentukan penanganan medis yang efektif. Tes Darah Lengkap atau Complete Blood Count (CBC) adalah prosedur medis standar yang mengukur berbagai komponen darah, seperti sel darah merah (RBC), sel darah putih (WBC), dan trombosit (PLT).

Meskipun tes CBC relatif murah dan mudah diakses, interpretasi klinis secara manual memakan waktu dan rentan memunculkan kesalahan (human error), terutama saat volume pasien tinggi. Penelitian sebelumnya menunjukkan bahwa machine learning dapat digunakan untuk mendeteksi penyakit darah berdasarkan parameter CBC [2]. Oleh karena itu, pendekatan machine learning digunakan dalam proyek ini untuk menganalisis pola parameter CBC secara cepat dan membantu proses klasifikasi diagnosis secara lebih efisien.

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

    - **Metrik keberhasilan:** masing-masing model dinilai pada data uji menggunakan accuracy, macro precision, macro recall, dan macro F1-score. Karena data tidak seimbang dan kasus kesehatan sensitif terhadap pasien sakit yang terlewat, model yang dipilih harus mempertahankan macro recall dan macro F1-score yang tinggi, bukan hanya accuracy. Sebagai target operasional proyek ini, model akhir diharapkan mencapai accuracy minimal 0,90, macro recall minimal 0,85, dan macro F1-score minimal 0,85.

- Melakukan tahap perbaikan (improvement) pada baseline model dari kedua algoritma melalui teknik Hyperparameter Tuning menggunakan GridSearchCV.

    - **Metrik keberhasilan:** konfigurasi hasil tuning dipilih berdasarkan nilai cross-validation accuracy tertinggi pada data latih, kemudian diverifikasi pada data uji. Tuning dianggap memberikan perbaikan apabila model hasil tuning memenuhi target evaluasi di atas dan tidak mengalami penurunan macro recall atau macro F1-score yang berarti dibandingkan model sebelum tuning.

- Solusi akan diukur secara objektif melalui komparasi metrik evaluasi klasifikasi. **Metrik keberhasilannya** adalah pemilihan model dengan kombinasi accuracy, macro precision, macro recall, dan macro F1-score terbaik pada data uji, dengan prioritas pada macro recall untuk mengurangi risiko false negative pada kelas diagnosis.

## Data Understanding
Data yang digunakan dalam proyek ini dimuat dari file Dataset Kaggle yaitu [Anemia Types Classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification). Dataset ini merupakan tabel yang merekam hasil tes klinis darah lengkap berserta label diagnosis pakar dari 1281 sampel/pasien. Seluruh tipe data fitur adalah numerik, dengan tipe kelas target berupa objek/teks.

#### Jumlah baris dan kolom:
> Dataset ini memiliki total 1281 baris (sampel/pasien) dan 15 kolom (yang terdiri dari 14 kolom fitur metrik darah dan 1 kolom target diagnosis).

<p align="center">
    <img width="611" height="563" alt="image" src="https://github.com/user-attachments/assets/1efb0939-fd6d-4f08-9921-49fb6340a7b9" />
</p>
Inspeksi kualitas data secara terprogram menunjukkan bahwa tidak terdapat nilai yang hilang (missing value) pada 15 kolom dataset. Namun, pemeriksaan pada dataset awal menemukan **49 baris duplikat**. Duplikasi ini berpotensi memberi bobot lebih besar pada observasi yang sama, memengaruhi distribusi kelas, dan membuat performa model terlihat lebih baik daripada kemampuan generalisasinya. Oleh karena itu, duplikat dihapus dengan `drop_duplicates()` pada tahap Data Preparation sehingga jumlah data berkurang dari 1.281 menjadi **1.232 baris unik**. Hasil pemeriksaan ini memastikan proses pemodelan menggunakan observasi yang tidak berulang, sedangkan tidak adanya missing value membuat imputasi tidak diperlukan.

#### Variabel-variabel pada dataset adalah sebagai berikut:
- WBC (White Blood Cell): Jumlah sel darah putih ($10^9/L$).
- LYMp: Persentase limfosit (%).
- NEUTp: Persentase neutrofil (%).
- LYMn: Jumlah absolut limfosit.
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
Kondisi data secara statistik menunjukkan perbedaan rentang angka yang cukup mencolok. Fitur seperti ukuran trombosit (PLT) memiliki skala nilai ratusan, sementara jumlah absolut limfosit (LYMn) berada pada rentang nilai desimal satu digit. Selain itu, dataset ini menunjukkan sebaran kelas yang imbalanced. Observasi ini menjadi dasar untuk melakukan standardisasi fitur pada tahap Data Preparation.

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

#### 1. Penghapusan Data Duplikat:
- Menghapus baris yang identik menggunakan `df.drop_duplicates()` dan menyimpan hasilnya sebagai dataset bersih. Dari 1.281 baris awal, sebanyak 49 baris duplikat dihapus sehingga tersisa 1.232 baris.

- Alasan: Setiap observasi yang berulang dapat memberi bobot berlebih pada pola tertentu, mengubah distribusi kelas, dan berpotensi menyebabkan evaluasi model terlalu optimistis.

#### 2. Pemisahan Fitur dan Target:
- Proses ini memisahkan dataset ke dalam dua variabel berbeda: variabel independen/fitur ($X$) dan variabel dependen/target ($y$). Seluruh kolom hasil tes laboratorium numerik dimasukkan ke dalam $X$ dengan melakukan drop pada kolom Diagnosis, sementara kolom Diagnosis itu sendiri diisolasi ke dalam variabel $y$.

- Alasan: Pemisahan ini merupakan syarat fundamental sebelum melakukan proses encoding maupun pemodelan, agar algoritma machine learning dapat membedakan mana data yang digunakan sebagai bahan belajar (fitur) dan mana data yang harus diprediksi (target).

#### 3. Label Encoding pada Kolom Target:
- Proses ini mengubah kelas string/teks pada kolom target Diagnosis menjadi bilangan integer (seperti 0, 1, 2, dll).

- Alasan: Pemodelan machine learning pada tahap selanjutnya merupakan operasi matematis murni. Model tidak bisa mengevaluasi kelas bertipe teks secara langsung.

#### 4. Train-Test Split:
- Membagi dataset secara utuh menjadi rasio latih dan uji (umumnya 80% data untuk training dan 20% data untuk testing).

- Alasan: Sangat esensial untuk mencegah data leakage dan berfungsi untuk menilai bagaimana model mampu melakukan generalisasi dalam menebak data unseen (data yang tidak dikenalnya selama proses pelatihan).

#### 5. Standarisasi Fitur (StandardScaler):
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

- Untuk klasifikasi multi-kelas, setiap kelas diagnosis diperlakukan secara bergantian sebagai kelas positif, sedangkan kelas lainnya menjadi kelas negatif. Dari perbandingan label sebenarnya dengan prediksi model pada confusion matrix, diperoleh **True Positive (TP)** ketika sampel kelas tersebut diprediksi benar sebagai kelas tersebut, **True Negative (TN)** ketika sampel dari kelas lain diprediksi bukan kelas tersebut, **False Positive (FP)** ketika sampel dari kelas lain keliru diprediksi sebagai kelas tersebut, dan **False Negative (FN)** ketika sampel kelas tersebut diprediksi sebagai kelas lain. Nilai metrik dihitung untuk setiap kelas, kemudian dirata-ratakan dengan `average='macro'` agar setiap diagnosis memiliki bobot yang sama.

- **Akurasi (Accuracy):** Mengukur proporsi seluruh prediksi yang benar terhadap seluruh sampel uji. Pada skema multi-kelas, bentuknya adalah jumlah elemen diagonal confusion matrix dibagi jumlah seluruh sampel, atau secara ekuivalen $(TP + TN) / (TP + TN + FP + FN)$ jika dilihat dari sudut pandang satu kelas.

- **Presisi (Precision):** Mengukur ketepatan prediksi positif. Dari seluruh sampel yang diprediksi sebagai suatu diagnosis, proporsi yang benar-benar termasuk diagnosis tersebut adalah $TP / (TP + FP)$. Presisi rendah berarti model menghasilkan lebih banyak false positive, misalnya pasien tanpa diagnosis tertentu ikut ditandai sebagai diagnosis tersebut.

- **Recall (Sensitivity):** Mengukur kemampuan model menemukan seluruh sampel yang benar-benar termasuk suatu diagnosis. Nilainya adalah $TP / (TP + FN)$. Dalam konteks kesehatan, false negative sangat berisiko karena pasien yang sebenarnya memiliki kelainan dapat diprediksi sebagai kelas lain atau Healthy. Oleh sebab itu, recall makro digunakan untuk memantau kemampuan model mendeteksi kelas minoritas secara lebih adil.

- **F1-Score:** Merupakan rata-rata harmonik antara precision dan recall, yaitu $2 \times (Precision \times Recall) / (Precision + Recall)$. F1-score tinggi hanya tercapai jika keduanya seimbang, sehingga metrik ini membantu menghindari model yang tampak baik hanya karena memiliki precision atau recall tinggi saja. Nilai `zero_division=0` digunakan pada implementasi untuk menangani kelas yang tidak memiliki prediksi positif tanpa menghasilkan kesalahan pembagian nol.

Dengan demikian, confusion matrix menjadi dasar perhitungan seluruh metrik: diagonalnya menunjukkan prediksi benar, sedangkan elemen di luar diagonal menunjukkan kesalahan antar-kelas. Evaluasi dilakukan pada data uji yang tidak digunakan untuk melatih model. Pendekatan ini penting agar performa yang dilaporkan merepresentasikan kemampuan generalisasi, terutama dalam mengurangi FN pada kelas penyakit.

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
