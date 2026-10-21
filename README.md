# ANALISIS PJ1 (RET503)
**Pengembangan, Pelatihan, Tuning, Seleksi, dan Pengujian Model YOLO**

---

### Informasi Mahasiswa & Sistem

| Field | Isi |
|-------|-----|
| **Nama Mahasiswa** | *Nadia Esterliana Siahaan* |
| **NIM** | *4222401001* |
| **Mata Kuliah** | Praktik Terpadu / RET503 — Computer Vision and Deep Learning |
| **Model Dasar** | YOLO (`yolo26n` |
| **Dataset** | `dataset_v1` (31 kelas target) |
| **Lingkungan Uji** | *Asus TUF A16* |

---

## TAHAP 1–2: Persiapan Dataset & Verifikasi Kualitas Anotasi (Asesmen P3 / CLO-3)

### 1. Tahap 1: Pengumpulan & Standarisasi Dataset

- **Sumber data:** Dataset deteksi objek dari Roboflow Universe (`image-detection-qixfs` / setara), distandarisasi ke struktur baku YOLO (`dataset_v1`).
- **Struktur:**
  - `images/{train,val,test}/` — citra
  - `labels/{train,val,test}/` — file `.txt` dengan nama identik terhadap citra
  - `data.yaml` — path split + 31 nama kelas
- **Pembagian data** (reproduktif, proporsi **70% / 20% / 10%**):

| Split | Jumlah citra | Fungsi |
|-------|--------------|--------|
| **Train** | **972** | Memperbarui bobot model |
| **Val** | **278** | Validasi tiap epoch & seleksi model terbaik |
| **Test** | **140** | Evaluasi final generalisasi (tidak disentuh saat train/tuning) |
| **Total** | **1.390** | |

- **Konfigurasi `data.yaml`:** menunjuk direktori `images/train`, `images/val`, `images/test` serta indeks kelas 0–30 (simbol huruf, angka, dan panah berwarna: Blue, Green, Red, White, Yellow).

### 2. Tahap 2: Pemeriksaan & Validasi Kualitas Anotasi

Pemeriksaan dilakukan sebelum pelatihan:

| Pemeriksaan | Hasil |
|-------------|--------|
| Format baris | Setiap baris tepat 5 atribut: `class_id x_center y_center width height` |
| Normalisasi | Koordinat & ukuran kotak dalam rentang **[0, 1]** |
| Validasi kelas | `class_id` berada di **0 … 30** (31 kelas) |
| Integritas pasangan | Jumlah file gambar = jumlah file label per split (972 / 278 / 140) |
| Anomali | Tidak ditemukan baris rusak pada pemeriksaan otomatis |

Kelas (31): Blue 3, Blue 8, Blue D, Blue Left Arrow, Blue S, Blue X, Green 2, Green 7, Green C, Green H, Green Right Arrow, Green W, Red 4, Red 9, Red Arrow Down, Red E, Red T, Red Y, White 1, White 6, White Arrow Up, White B, White Bullseye, White G, White V, Yellow 5, Yellow A, Yellow Circle, Yellow F, Yellow U, Yellow Z.

---

## TAHAP 3: Pelatihan Model E1, E2, E3 (Asesmen P4 / CLO-4)

Tiga eksperimen dijalankan dengan kerangka yang sama (`epochs=30`, `batch=16`), berbeda hanya pada **bobot awal** atau **satu hyperparameter**:

| Eksperimen | Bobot awal | Perubahan | imgsz |
|------------|------------|-----------|-------|
| **E1_pretrained** | Pretrained COCO | Baseline | 416 |
| **E2_scratch** | Acak (tanpa pretrained) | Kontrol transfer learning | 416 |
| **E3_tuning** | Pretrained COCO | **Satu** perubahan: resolusi input | **640** |

### 1. Tabel perbandingan hasil (isi setelah train)

| Parameter / Metrik | E1 (Baseline) | E2 (Scratch) | E3 (Tuning imgsz=640) |
|--------------------|---------------|--------------|------------------------|
| Transfer learning | Ya (COCO) | Tidak | Ya (COCO) |
| **mAP@0.5 (val)** | *(isi)* | *(isi)* | *(isi)* |
| **mAP@0.5:0.95 (val)** | *(isi)* | *(isi)* | *(isi)* |
| **Precision (P)** | *(isi)* | *(isi)* | *(isi)* |
| **Recall (R)** | *(isi)* | *(isi)* | *(isi)* |
| **Waktu inferensi (ms/citra)** | *(isi)* | *(isi)* | *(isi)* |
| **Waktu pelatihan** | *(isi)* | *(isi)* | *(isi)* |

### 2. Grafik pelatihan

Grafik metrik lengkap tersimpan pada:

- `hasil_praktik/E1_pretrained/results.png`
- `hasil_praktik/E2_scratch/results.png`
- `hasil_praktik/E3_tuning/results.png`

**Pembahasan kurva (isi setelah melihat grafik):**

1. **Loss (`box_loss`, `cls_loss` train/val)**  
   - Apakah `cls_loss` turun tajam di epoch awal pada E1/E3 (efek pretrained)?  
   - Apakah kurva val mengikuti train tanpa divergence besar?

2. **mAP50 & mAP50-95**  
   - Kapan mAP50 mendekati jenuh?  
   - Apakah E3 (`imgsz=640`) menaikkan mAP50-95 dibanding E1 (lokalisasi lebih presisi)?

---

## TAHAP 4–6: Seleksi Model, Uji TEST, dan Analisis (Pj1 / CLO-4)

### 1. Tahap 4: Pemilihan model terbaik (berdasarkan VAL)

- **Kriteria:** **mAP50-95 tertinggi pada data validasi** (mengukur deteksi + ketepatan bounding box di berbagai ambang IoU).
- **Model terpilih:** **`*(E1 / E2 / E3)*`** dengan mAP50-95 val = ***(isi)***.

### 2. Tahap 5: Pengujian pada data TEST (unseen)

Model terbaik diuji **satu kali saja** pada 140 citra test (tidak dipakai saat train/tuning):

| Metrik | Nilai TEST |
|--------|------------|
| **mAP@0.5** | *(isi)* |
| **mAP@0.5:0.95** | *(isi)* |

**Evaluasi singkat:**  
Bandingkan mAP50-95 val vs test. Selisih kecil → generalisasi baik; selisih besar → indikasi overfitting atau distribusi test berbeda.

### 3. Tahap 6: Analisis mendalam

#### A. Pengaruh transfer learning (E1 pretrained vs E2 scratch)

- Bandingkan mAP50-95 E1 vs E2.  
- **Pembahasan:** E2 diinisialisasi acak; dengan ~972 citra train dan 30 epoch, representasi fitur belum maksimal. E1 mewarisi fitur tingkat rendah (tepi, sudut, kontras) dari COCO sehingga konvergensi lebih cepat dan akurat.

#### B. Pengaruh hyperparameter tuning (E1 vs E3, imgsz 416 → 640)

- Bandingkan mAP50-95 dan precision.  
- **Trade-off:** resolusi lebih tinggi biasanya memperbaiki lokalisasi objek kecil/detail simbol, tetapi menambah waktu latih dan latency inferensi.

#### C. Rekomendasi implementasi

| Kebutuhan | Model disarankan | Alasan |
|-----------|------------------|--------|
| **Edge / real-time (robot, kamera bergerak)** | *(biasanya E1 jika latency jauh lebih rendah)* | FPS lebih tinggi, akurasi tetap memadai |
| **Akurasi maksimal (inspeksi / server)** | *(model dengan mAP50-95 val tertinggi)* | Presisi lokalisasi terbaik |

---

## Berkas rekapitulasi & artefak

| No | Artefak | Lokasi |
|----|---------|--------|
| 1 | Analisis ini | `README.md` |
| 2 | Tabel metrik val | `hasil_praktik_val.csv` (jika ada) |
| 3 | Grafik E1 / E2 / E3 | `hasil_praktik/E*_*/results.png` |
| 4 | Bobot model terbaik | `hasil_praktik/<model_terpilih>/weights/best.pt` |
| 5 | Konfigurasi data | `dataset_v1/data.yaml` |

> **Catatan ukuran file:** Dataset penuh dan file `.pt` besar tidak selalu di-commit ke GitHub. Simpan di Google Drive dan cantumkan tautan di bawah.

- **Dataset (`dataset_v1`):** *(tautan Google Drive)*  
- **Checkpoint `best.pt`:** *(tautan Google Drive)*  

---

## Cara menjalankan ulang (ringkas)

```bash
# 1. Siapkan dataset_v1 (struktur images/labels + data.yaml)
# 2. Perbaiki path di data.yaml agar absolut di mesin Anda

from ultralytics import YOLO

DATA = "dataset_v1/data.yaml"

# E1
YOLO("yolo26n.pt").train(data=DATA, epochs=30, imgsz=416, batch=16,
                         project="hasil_praktik", name="E1_pretrained")

# E2
YOLO("yolo26n.yaml").train(data=DATA, epochs=30, imgsz=416, batch=16,
                           pretrained=False, project="hasil_praktik", name="E2_scratch")

# E3
YOLO("yolo26n.pt").train(data=DATA, epochs=30, imgsz=640, batch=16,
                         project="hasil_praktik", name="E3_tuning")

# Evaluasi test (ganti path ke best.pt terpilih)
YOLO("hasil_praktik/E3_tuning/weights/best.pt").val(data=DATA, split="test", imgsz=640)
```

Jika `yolo26n` tidak tersedia di lingkungan Anda, ganti menjadi `yolov8n.pt` / `yolov8n.yaml` sesuai arahan dosen.

---

## Referensi

- Materi RET503 — deteksi objek & transfer learning  
- Ultralytics YOLO documentation  
- Dataset sumber: Roboflow Universe (object detection rambu / `image-detection-qixfs` atau setara), lisensi mengikuti sumber asli (CC BY 4.0 bila berlaku)

---

*Lengkapi semua sel *(isi)* setelah eksperimen selesai. Jangan memilih model berdasarkan angka TEST — hanya VAL, lalu laporkan TEST sekali.*
