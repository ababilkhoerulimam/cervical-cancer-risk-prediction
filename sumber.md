# Sumber Data dan Dokumentasi Ekosistem Riset (Cervical Cancer Risk Prediction)

Dokumen ini mencatat inventaris resmi sumber data yang digunakan, dipetakan, dan dianalisis dalam proyek prediksi risiko kanker serviks berbasis faktor gaya hidup, demografi, dan anamnesis.

---

## 1. Dataset Primer (Data Riil Rumah Sakit)

### UCI Machine Learning Repository: Cervical Cancer (Risk Factors)
* **Pengumpul Data:** Kelman Fernandes, Jaime S. Cardoso, Jessica Fernandes (Universidade do Porto & Hospital Universitario de Caracas, Venezuela).
* **Tahun Pengumpulan:** 2017.
* **Tautan Resmi UCI:** [UCI Machine Learning Repository - Dataset ID 383](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
* **Mirror Kaggle:** [Kaggle - Cervical Cancer Risk Classification (loveall)](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)
* **Karakteristik Data:**
  * **Populasi:** Pasien rawat jalan ginekologi di Hospital Universitario de Caracas, Venezuela.
  * **Ukuran Asli:** 858 observasi, 36 atribut.
  * **Ukuran File Lokal di Repo (`cervical-cancer_csv.csv`):** 835 observasi (setelah pembersihan baris duplikat/inkonsisten oleh uploader awal).
  * **Target Klinis Definitif:** `Biopsy` (histopatologi jaringan serviks: 54 positif / 835 total = 6.47%).
  * **Target Diagnostik Menengah (Dibuang untuk mencegah kebocoran klinis):** `Hinselmann` (kolposkopi), `Schiller` (uji iodine Lugol), `Citology` (Pap smear).
  * **Fitur Gaya Hidup Utama:** `Age`, `Number of sexual partners`, `First sexual intercourse`, `Num of pregnancies`, `Smokes (years, packs/year)`, `Hormonal Contraceptives (years)`, `IUD (years)`, riwayat `STDs` (kondiloma, herpes, HIV, HPV, sifilis).
* **Sitasi Akademik:**
  > Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). Transfer learning with partial observability applied to cervical cancer screening. In *Iberian Conference on Pattern Recognition and Image Analysis* (pp. 243-250). Springer, Cham. DOI: [10.1007/978-3-319-58838-4_27](https://doi.org/10.1007/978-3-319-58838-4_27).

---

## 2. Dataset Sekunder Riil (Populasi & Survei Biomarker Nasional)

### CDC NHANES (National Health and Nutrition Examination Survey)
* **Lembaga Penerbit:** Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics (NCHS), Amerika Serikat.
* **Tipe Data:** Data riil manusia (pemeriksaan fisik langsung, uji laboratorium darah/urin/sitologi, dan kuesioner medis privat).
* **Tautan Resmi CDC:** [CDC NHANES Questionnaires, Datasets, and Related Documentation](https://wwwn.cdc.gov/nchs/nhanes/default.aspx)
* **Siklus Rekomendasi:** 2015-2016 (Siklus I) dan 2017-2018 (Siklus J).
* **Modul Terkait yang Dipadukan:**
  1. **Demografi (`DEMO_I` / `DEMO_J`):**
     * Variabel: `SEQN` (ID Responden), `RIDAGEYR` (Umur dalam tahun), `RIAGENDR` (Gender, filter subjek wanita = 2).
     * Unduhan: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt`
  2. **Kesehatan Reproduksi (`RHQ_I` / `RHQ_J`):**
     * Variabel: `RHQ160` (Jumlah riwayat kehamilan / Parity), `RHQ420` (Pernah mengonsumsi pil KB / Hormonal Contraceptives), `RHQ010` (Umur menarke).
     * Unduhan: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/RHQ_I.xpt`
  3. **Perilaku Seksual (`SXQ_I`):**
     * Variabel: `SXD021` (Umur pertama kali berhubungan seksual / First sexual intercourse), `SXQ700` / `SXQ800` (Jumlah total pasangan seksual seumur hidup).
     * Unduhan: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SXQ_I.xpt`
  4. **Riwayat Merokok (`SMQ_I` / `SMQ_J`):**
     * Variabel: `SMQ020` (Pernah merokok minimal 100 batang seumur hidup), `SMD030` (Umur mulai merokok rutin).
     * Unduhan: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.xpt`
  5. **Kondisi Medis & Diagnosa Kanker (`MCQ_I` / `MCQ_J`):**
     * Variabel: `MCQ220` (Pernah didiagnosa menderita kanker atau keganasan oleh dokter), `MCQ230A` (Jenis kanker pertama yang diderita: Kode `15` merepresentasikan *Cervical Cancer*).
     * Unduhan: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt`

---

## 3. Dataset Sekunder Komplementer (Skrining & Simulasi Klinis)

### A. Hugging Face: Electric Sheep Africa Cervical Cancer Screening
* **Penerbit:** Electric Sheep Africa (Metadata-backed open healthcare initiatives).
* **Tautan Repository:** [Hugging Face - electricsheepafrica/africa-synth-cancer-cervical-cancer-screening-all](https://huggingface.co/datasets/electricsheepafrica/africa-synth-cancer-cervical-cancer-screening-all)
* **Karakteristik:**
  * Ukuran: 30.000 baris (3 file CSV: opportunistic screening, limited access, organized programme masing-masing 10.000 baris).
  * Variabel: `age_years`, `hiv_status`, `parity`, `contraception`, `true_hpv_positive`, `histology_result` (Normal, CIN1, CIN2, CIN3, Invasive Cancer).
  * Catatan: Data berbasis pemodelan sintetis pedoman WHO LMIC. Sangat baik digunakan sebagai pre-training atau prior distribution, namun tidak boleh menggantikan validasi data riil.

### B. CDC BRFSS (Behavioral Risk Factor Surveillance System)
* **Penerbit:** CDC Division of Population Health.
* **Tautan Mirror:** [Kaggle - aemreusta/brfss-2020-survey-data](https://www.kaggle.com/datasets/aemreusta/brfss-2020-survey-data)
* **Karakteristik:** Survei telepon tahunan (>400.000 responden) berisi status merokok, riwayat Pap test (`HADPAP2`), riwayat tes HPV (`HHPVTEST`), dan diagnosis kanker serviks (`CNCRTYP1`).

---

## 4. Matriks Harmonisasi Antar-Dataset (Feature Alignment Table)

Tabel berikut menjadi pedoman standarisasi kolom saat menyatukan dataset UCI Caracas dan CDC NHANES (Opsi B):

| Fitur Standar (Harmonized) | Kolom Sumber UCI Caracas | Kolom Sumber CDC NHANES | Tipe Data & Rentang Nilai |
|---|---|---|---|
| `Age` | `Age` | `RIDAGEYR` (DEMO) | Numerik kontinu (13 - 85 tahun) |
| `Number_of_sexual_partners` | `Number of sexual partners` | `SXQ700` / `SXQ800` (SXQ) | Numerik diskret (1, 2, 3, ...) |
| `First_sexual_intercourse` | `First sexual intercourse` | `SXD021` (SXQ) | Numerik kontinu / umur (10 - 40 tahun) |
| `Num_of_pregnancies` | `Num of pregnancies` | `RHQ160` (RHQ) | Numerik diskret (0 - 15) |
| `Smokes` | `Smokes` | `SMQ020` (1 = Ya, 2 = Tidak -> 0) | Biner (0 / 1) |
| `Hormonal_Contraceptives` | `Hormonal Contraceptives` | `RHQ420` (1 = Ya, 2 = Tidak -> 0) | Biner (0 / 1) |
| `STDs_HIV` | `STDs:HIV` | Hasil Lab / Kuesioner Medis | Biner (0 / 1) |
| `Target_Biopsy` | `Biopsy` | `MCQ230A == 15` (MCQ) | Biner (0 = Negatif, 1 = Positif Kanker/CIN) |

---

## 5. Protokol Integritas & Anti-Kebocoran (Leakage Prevention Invariant)

1. **Eksklusi Tes Diagnostik Penentu:** Seluruh variabel tes diagnostik penentu seperti `Hinselmann`, `Schiller`, dan `Citology` dari data primer dilarang masuk ke dalam matriks fitur pelatihan untuk memastikan model murni mengevaluasi risiko sebelum tindakan klinis invasif.
2. **Isolasi Imputasi & Penskalaan:** Seluruh estimasi median missing value dan penskalaan z-score wajib dipasang strictly di dalam batas fold training (fold-isolated) tanpa pernah meminjam statistik validation fold atau holdout.
3. **Pemberian Bobot Riil vs Tambahan:** Jika dataset sekunder digabungkan, evaluasi performa akhir wajib dilaporkan baik secara agregat maupun secara terisolasi pada cohort pasien riil (UCI) untuk menjamin transparansi daya generalisasi klinis.
