# API CONTRACT — SEKOLAH RAKYAT MVP
## Versi 3.8 — Production-Ready | Single Source of Truth

> **Status:** FINAL — Acuan wajib untuk Tim 6 BE, Tim 3 RAG, Tim 4 Game, Tim 5 Mentor, Tim 1 Emosi, Tim 6 FE
> **Tanggal:** 2026-05-23
> **Basis:** V3.7 + Slim Payload — FE hanya kirim identifier + input user
> **Ringkasan perubahan V3.8:** Penyederhanaan payload request pada 5 endpoint (POST /sesi, POST /siswa/:id/quiz/mc, POST /siswa/:id/quiz/essay, POST /mentor/pesan, POST /mentor/evaluasi). FE tidak lagi mengirim field metadata yang bisa di-lookup BE dari identifier yang sudah ada.
> **Addendum — Image Path:** Tambah field `image_path` (opsional, `string|null`) pada konten `bacaan`, `quiz_pg`, dan `quiz_essay`. Tim 3 RAG menyertakan path relatif gambar hasil ekstraksi dokumen. FE render langsung via `<img src={image_path}>` — browser resolve relatif terhadap domain FE.

---

## DAFTAR ISI

1. [Konvensi Global](#1-konvensi-global)
2. [Changelog V3.1 — Addendum Sesi & Konteks Mentor](#2-changelog-v31-addendum-sesi--konteks-mentor)
3. [Changelog V3.3 — Pemisahan Endpoint Quiz, Konten, Game, Mentor](#3-changelog-v33-pemisahan-endpoint-quiz-konten-game-mentor)
4. [Changelog V3.4 — Penyesuaian Struktur Konten](#4-changelog-v34-penyesuaian-struktur-konten)
5. [Changelog V3.5 — Fix Audit Contract](#5-changelog-v35-fix-audit-contract)
6. [Changelog V3.6 — Fix Audit Contract (Lanjutan)](#6-changelog-v36-fix-audit-contract-lanjutan)
7. [Changelog V3.7 — Integrasi Format Tim 2 & Revisi Pretest](#7-changelog-v37-integrasi-format-tim-2--revisi-pretest)
8. [Changelog V3.8 — Slim Payload Request](#8-changelog-v38-slim-payload-request)
9. [Changelog V3.9 — Revisi Payload POST /rag/rekomendasi & Image Path Konten](#9-changelog-v39-revisi-payload-post-ragrekomendasi--image-path-konten)
10. [Peta Domain Endpoint](#10-peta-domain-endpoint)
11. [AUTH — Tim 6 BE](#11-auth--tim-6-be)
12. [ADMIN — Tim 6 BE](#12-admin--tim-6-be)
13. [GURU — Tim 6 BE](#13-guru--tim-6-be)
14. [SISWA — Tim 6 BE](#14-siswa--tim-6-be)
15. [KONTEN — Tim 3 RAG + Tim 6 BE](#15-konten--tim-3-rag--tim-6-be)
16. [SESI — Tim 6 BE](#16-sesi--tim-6-be)
17. [PRETEST — Tim 3 RAG (generate) + Tim 6 BE (serve)](#17-pretest--tim-3-rag-generate--tim-6-be-serve)
18. [QUIZ — Tim 6 BE](#18-quiz--tim-6-be)
19. [RAG — Tim 3](#19-rag--tim-3)
20. [GAME — Tim 4 + Tim 6 BE](#20-game--tim-4--tim-6-be)
21. [EMOSI — Tim 1](#21-emosi--tim-1)
22. [MENTOR — Tim 5](#22-mentor--tim-5)
23. [LEADERBOARD — Tim 6 BE](#23-leaderboard--tim-6-be)
24. [NOTIFIKASI — Tim 6 BE](#24-notifikasi--tim-6-be)
25. [WebSocket Spec — Tim 6 BE](#25-websocket-spec--tim-6-be)
26. [Hirarki Kurikulum (Aturan Global)](#26-hirarki-kurikulum-aturan-global)
27. [Standard Response & Error](#27-standard-response--error)
28. [BOOKS — Tim 6 BE + Tim 3 (Ingestion)](#28-books--tim-6-be--tim-3-ingestion)

---

## 1. KONVENSI GLOBAL

### 1.1 Base URL
```
https://api.sekolahrakyat.id/v1
```
Semua path endpoint di dokumen ini relatif terhadap base URL ini.

### 1.2 Autentikasi
Semua endpoint (kecuali yang ditandai `[PUBLIC]`) wajib menyertakan header:
```
Authorization: Bearer <access_token>
```

### 1.3 Standard Response Envelope
**Semua response WAJIB menggunakan envelope berikut:**
```json
{
  "data": {},
  "meta": null,
  "error": null
}
```
- **Success:** `data` berisi payload, `error` = `null`
- **Error:** `data` = `null`, `error` berisi objek error standar
- **Paginated:** `meta` berisi objek pagination (lihat 1.5)

### 1.4 Standard Error Object
```json
{
  "data": null,
  "meta": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Deskripsi error yang human-readable.",
    "details": {}
  }
}
```

**Error Codes standar:**
| Code | HTTP Status | Keterangan |
|------|-------------|------------|
| `UNAUTHORIZED` | 401 | Token tidak valid / expired |
| `FORBIDDEN` | 403 | Role tidak punya akses resource ini |
| `NOT_FOUND` | 404 | Resource tidak ditemukan |
| `CONFLICT` | 409 | Data duplikat (email, NIP, NIS, dll) |
| `VALIDATION_ERROR` | 400 | Field wajib kosong, format tidak valid |
| `UNPROCESSABLE` | 422 | Data tidak valid secara logika bisnis |
| `RATE_LIMITED` | 429 | Terlalu banyak request (LLM/RAG) |
| `INTERNAL_ERROR` | 500 | Kesalahan server tidak terduga |

### 1.5 Pagination
Endpoint list yang mungkin besar WAJIB mendukung pagination:

**Query params:** `?page=1&limit=20`

**Meta pagination di response:**
```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "total_pages": 8
  },
  "error": null
}
```

### 1.6 Konvensi Naming
- **Bahasa:** Field names dalam **Bahasa Indonesia** untuk domain bisnis; `id`, `status`, `level`, `score`, `role`, `email` tetap Inggris (common terms)
- **Format:** `snake_case` untuk semua field
- **Timestamp:** ISO 8601 (`2026-05-01T09:00:00.000Z`) untuk semua field waktu — termasuk WebSocket
- **ID:** String (bukan number) — hindari integer overflow
- **NIP:** String 18 digit
- **Level input dari UI (request):** Kapital di awal → `"Low"` | `"Mid"` | `"High"`
- **Level response dari BE (semua response):** Lowercase → `"low"` | `"mid"` | `"high"`

> **Aturan level casing wajib dipatuhi semua tim.** BE mengembalikan `"low"/"mid"/"high"` di semua response. FE mengirim `"Low"/"Mid"/"High"` hanya di request generate/publish/submit. Pelanggaran menyebabkan bug silent di FE saat compare string.

### 1.7 HTTP Methods
| Method | Semantik |
|--------|----------|
| `GET` | Baca data, tidak mengubah state |
| `POST` | Buat resource baru / aksi yang tidak idempoten |
| `PUT` | Full replace resource |
| `PATCH` | Partial update resource |
| `DELETE` | Hapus resource |

### 1.8 Filtering, Sorting
```
GET /admin/siswa?kelas_id=x1&status=Aktif&sort=nama&order=asc
```
- `sort`: nama field yang digunakan untuk sort
- `order`: `asc` | `desc` (default: `asc`)
- Filter key = nama field resource

---

## 2. CHANGELOG V3.1 (Addendum Sesi & Konteks Mentor)

Perubahan V3.1 bersifat **additive dan backward-compatible**.

| Titik | Perubahan |
|-------|-----------|
| `POST /sesi` — Request | **Hapus** field `konteks_quiz` — tidak lagi diperlukan |
| `POST /mentor/pesan` — Request | **Tambah** field `konteks.publish_id` dan `konteks.bacaan` |
| `POST /sesi` — Timing | Dipanggil saat siswa **masuk chatbot** (bukan saat pesan pertama) |

---

## 3. CHANGELOG V3.3 (Pemisahan Endpoint Quiz, Konten, Game, Mentor)

Perubahan V3.3 bersifat **additive**.

| Titik | Perubahan |
|-------|-----------|
| `POST /siswa/:id/quiz` | **Deprecated** → dipecah menjadi `POST /siswa/:id/quiz/mc` dan `POST /siswa/:id/quiz/essay` |
| `POST /konten/generate` | **Tambah** field `konten_id` di response — identifier per konten per level untuk regenerate |
| `POST /konten/regenerate` | Deprecated — digabung ke `POST /konten/generate`. instruksi_revisi ada = regenerate, tidak ada = generate baru |
| `POST /game/regenerate` | **Baru** — iterative refinement per game menggunakan `game_id` |
| `POST /mentor/evaluasi` | **Baru** — evaluasi quiz CTA, system prompt Tim 5 terpisah dari chat normal |
| `POST /mentor/evaluasi/stream` | **Baru** — versi SSE dari evaluasi |
| WebSocket | **Tambah** event `essay_dinilai` — push agregasi setelah Tim 3 selesai nilai essay |

### 3.1 Ringkasan Perubahan URL

| V2 (Lama) | V3 (Baru) |
|-----------|-----------|
| `POST /summary/siswa/:id` | `POST /sesi/:id/summary` |
| `GET /content/siswa` | `GET /siswa/:id/konten` |
| `GET /content/progress/siswa` | `GET /siswa/:id/progress` |
| `GET /content/progress/guru` | `GET /kelas/:id/progress` |
| `GET /content/riwayat` | `GET /guru/:id/konten` |
| `POST /content/quiz/submit` | `POST /siswa/:id/quiz/mc` + `/essay` |
| `GET /content/quiz/history` | `GET /siswa/:id/quiz?elemen_id=` |
| `GET /content/pretest/status` | `GET /siswa/:id/pretest/status` |
| `POST /content/recommend` | `POST /rag/rekomendasi` |
| `POST /content/insight` | `POST /rag/insight` |
| `GET /guru/rekomendasi` | `GET /siswa/:id/notifikasi` |
| `POST /guru/rekomendasi` | `POST /notifikasi` |
| `GET /emotion/history` | `GET /sesi/:id/emosi` |
| `GET /mentor/chat/history` | `GET /sesi/:id/chat` |
| `POST /game/selesai` | `PATCH /game/:id/penyelesaian` |

---

## 4. CHANGELOG V3.4 (Penyesuaian Struktur Konten)

Perubahan V3.4 bersifat **additive pada struktur `content`** — tidak ada endpoint baru atau perubahan URL.

| # | Titik | Perubahan |
|---|-------|-----------|
| 1 | `POST /game/generate`, `POST /game/regenerate`, `GET /game/:id`, `POST /konten/publish` (item game) | **`html_url` → `html_string`** — Tim 4 kirim game sebagai HTML string penuh. FE render via `<iframe srcDoc={html_string}>` |
| 2 | `POST /konten/generate` (tipe `quiz_pg`), semua endpoint yang kembalikan/simpan `quiz_pg` | **Tambah field `penjelasan`** di setiap soal MC |
| 3 | `POST /konten/generate` (tipe `mindmap`), semua endpoint yang kembalikan/simpan `mindmap` | **Tambah field `penjelasan`** di setiap node mindmap |
| 4 | `POST /konten/generate` (tipe `bacaan` & `flashcard`), semua endpoint yang kembalikan/simpan tipe tersebut | **Tambah field `source`** — sumber buku/dokumen hasil retrieve RAG |
| 5 | `POST /konten/generate`, `POST /konten/publish`, semua endpoint terkait pretest | ~~Pretest di-generate bersamaan konten secara internal~~ → **Direvisi di V3.6** — FE kirim 1 request eksplisit `tipe: "pretest"`. Lihat Section 13. |
| 6 | `PATCH /game/:id/penyelesaian` | **Trigger selesai via `postMessage`** — game HTML kirim `{ type: 'game:selesai' }` ke parent FE |

---

## 5. CHANGELOG V3.5 (Fix Audit Contract)

| # | Titik | Perubahan |
|---|-------|-----------|
| 1 | Konvensi 1.6 | **Perjelas aturan level casing** — semua response BE wajib lowercase |
| 2 | Semua response endpoint | **Fix level casing** — ganti semua `"Low"/"Mid"/"High"` di response menjadi `"low"/"mid"/"high"` |
| 3 | `GET /siswa/:id/progress` | **Fix trailing comma JSON** di `nilai_agregasi_terakhir` |
| 4 | `GET /admin/mapel/:id`, `GET /admin/mapel/:mapel_id/elemen/:id` | **Fix envelope** — tambahkan standard response envelope |
| 5 | Struktur dokumen | **Fix penomoran section** |
| 6 | `POST /rag/insight` | **Fix kontradiksi request body** |
| 7 | `POST /konten/publish` | **Fix error 409** — hapus kalimat "Gunakan endpoint update" |
| 8 | `POST /konten/publish` — item game | **Fix konten_id game** — item game cukup gunakan `game_id` tanpa `konten_id` |

---

## 6. CHANGELOG V3.6 (Fix Audit Contract — Lanjutan)

Perubahan V3.6 bersifat **additive dan backward-compatible**.

| # | Titik | Perubahan |
|---|-------|-----------|
| 1 | `GET /siswa/:id/quiz` — response | **Tambah field `hasil_quiz_id`** di setiap item `riwayat[]` |
| 2 | `POST /konten/publish` | **Tambah catatan atomik** |
| 3 | `POST /konten/generate` — catatan Pretest | **Revisi mekanisme pretest** — FE kirim 1 request eksplisit `tipe: "pretest"` (total 14 panggilan paralel) |
| 4 | Section GURU | **Tambah `GET /kelas/:id/progress` sebagai section resmi** |
| 5 | Section WebSocket | **Tambah WebSocket Siswa** — endpoint `wss://.../ws/siswa` tersendiri |

---

## 7. CHANGELOG V3.7 (Integrasi Format Tim 2 & Revisi Pretest)

Perubahan V3.7 mencakup **integrasi penuh format data Tim 2 (SFT)** ke semua tipe konten dan **revisi algoritma penetapan level pretest**. Semua transformasi adalah tanggung jawab Tim 3 RAG — FE menerima data clean.

| # | Tipe/Endpoint | Perubahan | Breaking? |
|---|--------------|-----------|-----------|
| 1 | `bacaan` — semua endpoint | **Tambah field `judul`** di `content` (wajib, non-empty). Tim 3 map dari `judul_utama` Tim 2. FE gunakan sebagai header komponen bacaan. | Tidak |
| 2 | `quiz_pg` — semua endpoint | **Tambah field `stimulus`** di setiap item soal (wajib, 50–150 kata). Tim 3 map dari `stimulus` Tim 2. FE render di atas soal. | Tidak |
| 3 | `quiz_essay` — semua endpoint | **Tambah field `stimulus`** per soal (wajib). **Tambah field `penjelasan`** per soal (tidak tampil ke siswa, digunakan Tim 5 via `hasil_quiz_id` lookup). `rubrik` dipertahankan sebagai 1 string deskriptif per soal (Tim 3 normalize dari `rubric_points` array Tim 2). | Tidak |
| 4 | `mindmap` — semua endpoint | **Penegasan wajib**: Tim 3 RAG melakukan DFS flatten dari nested tree Tim 2 (`root→children→children`) menjadi flat array `nodes[]` dengan `parent_id`. FE dilarang melakukan rekursi flatten. | Tidak |
| 5 | `flashcard` — semua endpoint | **Penegasan mapping**: Tim 3 rename `front`→`depan`, `back`→`belakang` dari format Tim 2. Envelope `{cards[], source}` tidak berubah. | Tidak |
| 6 | `pretest` — semua endpoint | **Tambah field `tingkat_kesulitan`** (`"low"/"mid"/"high"`) per soal — Tim 3 map dari `LOTS→low`, `MOTS→mid`, `HOTS→high`. **Tambah field `stimulus`** per soal. **Hapus field `bobot` dan `distribusi`** — tidak diperlukan oleh algoritma baru. | Ya |
| 7 | `POST /pretest/submit` — Request | **Sederhanakan format `jawaban`**: dari `{pilihan, tingkat_kesulitan}` kembali ke flat string. BE lookup `tingkat_kesulitan` dari database soal via `sesi_pretest_id`. | Ya |
| 8 | `POST /pretest/submit` — Response | **Ganti field response**: hapus `nilai` (persentase), tambah `benar_per_tingkat` (breakdown per level) dan `total_benar`. | Ya |
| 9 | `POST /pretest/submit` — Logika Level | **Ganti algoritma**: dari threshold persentase ke **Lowest Failed Level** — dievaluasi berurutan Low→Mid→High; berhenti di level pertama yang gagal. | Ya |
| 10 | `POST /konten/publish` — item `bacaan`, `quiz_pg`, `quiz_essay` | **Update contoh payload**: sesuaikan dengan field baru (`judul`, `stimulus`, `rubrik`, `penjelasan`). | Tidak |

> **Catatan breaking changes (item 6–9):** Tim 6 FE wajib update adapter `mapPretestSoalV3` dan `PretestPage` submit handler sebelum `MOCK = false`. Tim 6 BE wajib update handler `POST /pretest/submit`.
>
> **Prinsip Decoupling (wajib dipatuhi Tim 3 RAG):** Semua transformasi dari format Tim 2 dilakukan saat ingest ke database BE — bukan saat runtime FE. FE menerima data clean dan tidak melakukan rekursi, string parsing, atau field mapping domain bisnis.

---

## 8. CHANGELOG V3.8 (Slim Payload Request)

Perubahan V3.8 menerapkan prinsip **"FE hanya kirim identifier + input user"** — field metadata yang bisa di-lookup BE dari identifier yang sudah ada di database tidak perlu dikirim ulang oleh FE.

**Prinsip:** `publish_id` sudah mengandung `mapel_id`, `elemen_id`, `materi_id`, `kelas_id`, `atp`. `sesi_id` sudah terikat ke siswa dan konteks topik. `hasil_quiz_id` sudah mengandung seluruh data quiz. BE wajib melakukan lookup dari identifier tersebut — tidak boleh mengembalikan error karena field turunan tidak dikirim.

| # | Endpoint | Field dihapus dari request FE | Alasan |
|---|----------|-------------------------------|--------|
| 1 | `POST /sesi` | `mapel_id`, `elemen_id`, `materi_id` | BE lookup dari `publish_id` |
| 2 | `POST /siswa/:id/quiz/mc` | `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id` | BE lookup dari `publish_id` |
| 3 | `POST /siswa/:id/quiz/essay` | `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id` | BE lookup dari `publish_id` |
| 4 | `POST /mentor/pesan` + `/stream` | `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id`, `atp` | BE lookup dari `sesi_id` → `publish_id` |
| 5 | `POST /mentor/evaluasi` + `/stream` | `sesi_id`, `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id`, `level`, `atp` | BE lookup dari `hasil_quiz_id` |

> **Breaking changes:** Semua perubahan ini bersifat **breaking** — BE wajib menerima payload slim dan tidak boleh return error 400 karena field yang dihapus tidak ada. FE wajib stop mengirim field yang dihapus.
>
> **`siswa_id` tetap dikirim eksplisit** di semua endpoint — untuk kebutuhan logging, audit trail, dan validasi silang.
>
> **`level` di `POST /mentor/pesan` tetap dikirim eksplisit** — karena level siswa bisa berubah mid-session (naik level setelah quiz agregasi ≥ KKM), dan Tim 5 membutuhkan konteks level aktif saat ini untuk menyesuaikan respons chatbot.
>
> **`konteks.emosi` dan `konteks.bacaan` di `POST /mentor/pesan` tetap dikirim** — keduanya adalah data real-time dari FE (kamera + konten yang sedang dirender) yang tidak tersimpan di database BE.
>
> **`POST /sesi/:id/summary` tidak berubah** — tetap kirim semua field secara eksplisit sesuai kesepakatan dengan BE.

---

## 9. CHANGELOG V3.9 — Revisi Payload POST /rag/rekomendasi & Image Path Konten

| # | Endpoint | Perubahan |
|---|----------|-----------|
| 1 | `POST /rag/rekomendasi` — Request | **Hapus** field `levels` |
| 2 | `POST /rag/rekomendasi` — Request | **Tambah** field `available_ids` — semua berisi `publish_id` |
| 3 | `GET /siswa/:id/progress` — Response | `sudah_selesai_ids` dan `sedang_dipelajari_ids` kini mengembalikan `publish_id` (bukan `elemen_id`) |
| 4 | `POST /konten/generate`, `GET /siswa/:id/konten`, `GET /guru/:id/konten`, `POST /konten/publish` — tipe `bacaan` | **Tambah field `image_path`** (opsional, `string\|null`) di `content`. Path relatif gambar hasil ekstraksi Tim 3 di-deploy ke folder static FE. | Tidak |
| 5 | `POST /konten/generate`, `GET /siswa/:id/konten`, `GET /guru/:id/konten`, `POST /konten/publish` — tipe `quiz_pg` | **Tambah field `image_path`** (opsional, `string\|null`) per soal di `content.soal[]`, disisipkan setelah `stimulus`. | Tidak |
| 6 | `POST /konten/generate`, `GET /siswa/:id/konten`, `GET /guru/:id/konten`, `POST /konten/publish` — tipe `quiz_essay` | **Tambah field `image_path`** (opsional, `string\|null`) per soal di `content.pertanyaan[]`, disisipkan setelah `stimulus`. | Tidak |

> **Prinsip:** Tim 3 RAG menerima universe lengkap (`available_ids`) + status aktual siswa (`completed_ids`, `in_progress_ids`) dalam satu format identifier yang konsisten. BE Tim 6 lookup semua konteks dari `publish_id`.
> **Mekanisme `image_path`:** Tim 3 RAG mengekstraksi gambar dari dokumen sumber saat proses ingest, kemudian men-deploy file gambar tersebut ke folder static container FE di VPS (`/assets/extracted/...`). Path yang disimpan di Qdrant sudah disesuaikan dengan lokasi file di VPS. Saat retrieve, path ikut keluar bersama konten. FE render langsung via `<img src={image_path}>` — browser resolve relatif terhadap domain FE tanpa transformasi. BE menyimpan dan mengembalikan string `image_path` apa adanya (pass-through murni).

---

## 10. PETA DOMAIN ENDPOINT

```
/auth          → Autentikasi & sesi
/admin         → Manajemen kurikulum, guru, siswa, kelas (role: admin)
/guru/:id      → Data & aksi guru
/siswa/:id     → Data & aksi siswa
/kelas/:id     → Data kelas (monitoring guru)
/konten        → Generate & publish konten (role: guru)
/sesi          → Sesi belajar siswa
/pretest       → Soal & submit pretest
/rag           → Semua permintaan ke Tim 3 RAG (insight, rekomendasi)
/game          → Generate & aksi game (Tim 4)
/emosi         → Deteksi emosi frame (Tim 1)
/mentor        → Chatbot mentor (Tim 5)
/leaderboard   → Gamifikasi ranking
/notifikasi    → Notifikasi guru → siswa (satu arah)
/ws            → WebSocket monitoring real-time
```

---

## 11. AUTH — Tim 6 BE

### `[PUBLIC]` POST /auth/login
Login siswa, guru, atau admin. **Hanya email + password** — tidak ada NIS/NIP login, tidak ada OAuth.

**Request:**
```json
{
  "email": "budi@sekolah.id",
  "password": "passwordku"
}
```

**Response 200:**
```json
{
  "data": {
    "access_token": "eyJhbGci...",
    "refresh_token": "eyJhbGci...",
    "user": {
      "id": "usr_001",
      "nama": "Budi Santoso",
      "email": "budi@sekolah.id",
      "role": "siswa",
      "avatar": null,
      "is_first_login": true,
      "nis": "1234567890",
      "nip": null,
      "kelas_id": "x1"
    }
  },
  "meta": null,
  "error": null
}
```

> - `nis`: hanya siswa, `null` untuk guru/admin
> - `nip`: hanya guru (string 18 digit), `null` untuk siswa/admin
> - `is_first_login: true` → FE wajib paksa alur aktivasi (ganti password + pilih 3 mapel)
> - `kelas_id`: hanya siswa, `null` untuk guru/admin

**Error 401:** `UNAUTHORIZED` — "Email atau password salah."

---

### `[PUBLIC]` POST /auth/refresh
Tukar refresh token dengan access token baru.

**Request:**
```json
{ "refresh_token": "eyJhbGci..." }
```

**Response 200:**
```json
{
  "data": {
    "access_token": "eyJhbGci...",
    "refresh_token": "eyJhbGci..."
  },
  "meta": null,
  "error": null
}
```

**Error 401:** refresh token expired → FE paksa logout penuh.

---

### POST /auth/logout
Blacklist token aktif di server.

**Response 200:**
```json
{ "data": { "logged_out": true }, "meta": null, "error": null }
```

---

### POST /auth/aktivasi
Aktivasi akun siswa saat **first login** — ganti password + simpan 3 mapel pilihan. Atomik: jika gagal, seluruh aktivasi dianggap belum selesai.

**Auth:** role `siswa` (token sementara dari login pertama)

**Request:**
```json
{
  "password_baru": "passwordBaru123!",
  "mapel_ids": ["mat", "bio", "fis"]
}
```

> - `password_baru`: min 8 karakter, kombinasi huruf + angka
> - `mapel_ids`: tepat 3 elemen
> - `user_id` diambil dari JWT token — tidak perlu dikirim di body

**Response 200:**
```json
{
  "data": {
    "access_token": "eyJhbGci...",
    "refresh_token": "eyJhbGci...",
    "user": {
      "id": "usr_001",
      "nama": "Budi Santoso",
      "email": "budi@sekolah.id",
      "role": "siswa",
      "avatar": null,
      "is_first_login": false,
      "nis": "1234567890",
      "nip": null,
      "kelas_id": "x1"
    },
    "mapel_terpilih": ["mat", "bio", "fis"]
  },
  "meta": null,
  "error": null
}
```

**Error 400:** "Harus memilih tepat 3 mata pelajaran."
**Error 400:** "Password minimal 8 karakter dan harus mengandung huruf dan angka."

---

### PATCH /auth/password
Ganti password user yang sedang login (dari halaman profil).

**Auth:** semua role

**Request:**
```json
{
  "password_lama": "passwordLama",
  "password_baru": "passwordBaru123!"
}
```

**Response 200:**
```json
{ "data": { "updated": true }, "meta": null, "error": null }
```

**Error 401:** "Password lama tidak sesuai."

---

### `[PUBLIC]` POST /auth/lupa-password
Kirim link reset ke email. Selalu 200 (tidak bocorkan apakah email terdaftar).

**Request:** `{ "email": "budi@sekolah.id" }`

**Response 200:**
```json
{ "data": { "sent": true }, "meta": null, "error": null }
```

---

### GET /auth/me
Validasi sesi + ambil profil terbaru. Dipanggil saat refresh halaman.

**Response 200:** `data` = objek `user` identik dengan field `user` di `/auth/login`.

---

### PUT /auth/avatar
Upload / ganti foto profil user aktif.

**Auth:** semua role
**Content-Type:** `multipart/form-data`
**Form Fields:** `file` (JPEG/PNG, maks 2 MB)

**Response 200:**
```json
{
  "data": { "avatar": "https://cdn.sekolahrakyat.id/avatars/usr_001.jpg" },
  "meta": null,
  "error": null
}
```

**Error 400:** "Format file tidak didukung atau ukuran melebihi 2 MB."

---

## 12. ADMIN — Tim 6 BE

> Semua endpoint section ini hanya untuk role **`admin`**. Response `403` jika role lain mengakses.

### 12.1 Mapel (Mata Pelajaran)

#### GET /admin/mapel
**Query Params (opsional):** `tingkat` (`X` | `XI` | `XII`)

**Response 200:**
```json
{
  "data": [
    {
      "id": "mat",
      "label": "Matematika",
      "icon": "📐",
      "fase": "Fase E (Kelas X)",
      "deskripsi_cp": "Pada akhir fase ini, peserta didik mampu berpikir kritis...",
      "jumlah_elemen": 4
    }
  ],
  "meta": null,
  "error": null
}
```

#### GET /admin/mapel/:id
**Response 200:**
```json
{
  "data": {
    "id": "mat",
    "label": "Matematika",
    "icon": "📐",
    "tingkat": "X",
    "fase": "Fase E (Kelas X)",
    "deskripsi_cp": "Pada akhir fase ini...",
    "elemen": [
      { "id": "bil_aljabar", "label": "Bilangan dan Aljabar" },
      { "id": "geometri", "label": "Geometri dan Pengukuran" }
    ]
  },
  "meta": null,
  "error": null
}
```

#### POST /admin/mapel
**Request:**
```json
{
  "label": "Matematika",
  "icon": "📐",
  "tingkat": "X",
  "fase": "Fase E (Kelas X)",
  "deskripsi_cp": "Pada akhir fase ini..."
}
```
> `id` tidak dikirim FE — BE generate via auto-increment database.
**Response 201:** `data` = `Mapel` yang dibuat.
**Error 409:** "ID mapel sudah digunakan."

#### PATCH /admin/mapel/:id
`id` tidak bisa diubah. **Request Body (partial):** `{ "label"?, "icon"?, "fase"?, "deskripsi_cp"? }`
**Response 200:** `data` = `Mapel` yang diperbarui.

#### DELETE /admin/mapel/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — elemen di bawah mapel ini dihapus cascade.

**Error 422:** "Mapel tidak dapat dihapus karena sudah memiliki konten yang dipublish ke siswa."

> Guard dilakukan di level mapel — BE cek apakah ada konten publish di seluruh elemen mapel ini.
> Jika tidak ada publish di elemen manapun, hapus mapel + semua elemen cascade.
> Guru tidak memiliki opsi unpublish — konten yang sudah dipublish bersifat permanen.

---

### 12.2 Elemen (Kurikulum)

> Elemen adalah level kurikulum langsung di bawah mapel. Hanya menyimpan `{ id, mapel_id, label }`.

#### GET /admin/mapel/:mapel_id/elemen
**Response 200:**
```json
{
  "data": [
    { "id": "bil_aljabar", "mapel_id": "mat", "label": "Bilangan dan Aljabar" },
    { "id": "geometri", "mapel_id": "mat", "label": "Geometri dan Pengukuran" }
  ],
  "meta": null,
  "error": null
}
```

#### GET /admin/mapel/:mapel_id/elemen/:id
**Response 200:**
```json
{
  "data": { "id": "bil_aljabar", "mapel_id": "mat", "label": "Bilangan dan Aljabar" },
  "meta": null,
  "error": null
}
```

#### POST /admin/mapel/:mapel_id/elemen
**Request Body:** `{ "label": "Bilangan dan Aljabar" }`
**Response 201:** `data` = elemen yang dibuat.
**Error 409:** "Label elemen sudah ada di mapel ini."

#### PATCH /admin/mapel/:mapel_id/elemen/:id
`id` dan `mapel_id` tidak bisa diubah. **Request Body:** `{ "label": "Nama Elemen Baru" }`
**Response 200:** `data` = elemen yang diperbarui.

#### DELETE /admin/mapel/:mapel_id/elemen/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

**Error 422:** "Elemen tidak dapat dihapus karena sudah memiliki konten yang dipublish ke siswa."

> Jika elemen sudah punya publish, otomatis mapelnya juga punya — guard konsisten di kedua level.

---

### 12.3 Kelas

#### GET /admin/kelas
**Query Params (opsional):** `tingkat` (`X` | `XI` | `XII`)

**Response 200:**
```json
{
  "data": [
    {
      "id": "x1",
      "nama": "X-1",
      "tingkat": "X",
      "tahun_ajaran": "2025/2026",
      "jumlah_siswa": 30,
      "wali_kelas_id": "g1",
      "mapel_guru_map": { "mat": "g1", "bio": "g2" }
    }
  ],
  "meta": null,
  "error": null
}
```

#### GET /admin/kelas/:id
**Response 200:** `data` = satu objek `Kelas`.

#### GET /admin/kelas/:id/siswa
**Response 200:**
```json
{
  "data": [
    { "id": "s1", "nama": "Budi Santoso", "nis": "1234567890", "email": "budi@sekolah.id", "status": "Aktif" }
  ],
  "meta": { "page": 1, "limit": 50, "total": 30, "total_pages": 1 },
  "error": null
}
```

#### POST /admin/kelas
**Request:** `{ "nama": "X-1", "tingkat": "X", "tahun_ajaran": "2025/2026", "wali_kelas_id": null }`
**Response 201:** `data` = `Kelas` yang dibuat.

#### PATCH /admin/kelas/:id
Body: field yang diubah saja (`nama`, `wali_kelas_id`, `tahun_ajaran`).
**Response 200:** `data` = `Kelas` yang diperbarui.

#### DELETE /admin/kelas/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — siswa dilepas dari kelas, tidak dihapus.

#### POST /admin/kelas/:id/mapel
**Request Body:** `{ "mapel_id": "mat", "guru_id": "g1" }`
**Response 201:** `data` = `Kelas` yang diperbarui.
**Error 409:** "Mapel sudah ada di kelas ini."

#### PATCH /admin/kelas/:id/mapel/:mapel_id
**Request:** `{ "guru_id": "g2" }`
**Response 200:** `{ "data": { "mapel_id": "mat", "guru_id": "g2" }, "meta": null, "error": null }`

#### DELETE /admin/kelas/:id/mapel/:mapel_id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

#### POST /admin/kelas/:id/siswa
**Request:** `{ "siswa_id": "s5" }`
**Response 201:** `data` = siswa yang diperbarui.
**Error 409:** "Siswa sudah ada di kelas ini."

#### DELETE /admin/kelas/:id/siswa/:siswa_id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

---

### 12.4 Guru

#### GET /admin/guru
**Query Params (opsional):** `sort=nama&order=asc`

**Response 200:**
```json
{
  "data": [
    {
      "id": "g1",
      "nama": "Ibu Sari",
      "nip": "199001012020012001",
      "email": "sari@sekolah.id",
      "avatar": null,
      "kelas_ids": ["x1", "x2"],
      "mapel_kelas_map": { "mat": ["x1", "x2"] }
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 5, "total_pages": 1 },
  "error": null
}
```

#### GET /admin/guru/:id
**Response 200:** `data` = satu objek `Guru`.

#### POST /admin/guru
**Request:**
```json
{
  "nama": "Pak Budi",
  "nip": "199501152019011001",
  "email": "budi.guru@sekolah.id",
}
```
> `mapel_kelas_map` tidak dikirim saat create — assignment mapel dan kelas dilakukan
> via `POST /admin/kelas/:id/mapel` dari halaman detail kelas.
> `status` default `"Aktif"` — opsional, BE set otomatis.
> `bergabung` / tanggal bergabung — otomatis dari BE, tidak dikirim FE.
> `avatar`, `_tempPassword` tidak boleh dikirim FE di request ini.

**Response 201:**
```json
{
  "data": {
    "id": "g1",
    "nama": "Pak Budi",
    "nip": "199501152019011001",
    "email": "budi.guru@sekolah.id",
    "avatar": null,
    "status": "Aktif",
    "temp_password": "g98pfTwU",
    "mapel_kelas_map": {}
  },
  "meta": null,
  "error": null
}
```

> `temp_password`: BE generate, dikembalikan **sekali** di response create — FE tampilkan ke admin di drawer/modal.
> BE juga mengirim `temp_password` ke email guru secara async.
> Setelah guru login pertama dan aktivasi, `temp_password` tidak lagi relevan dan tidak ada di response lain.

**Error 409:** "Email atau NIP sudah terdaftar."

#### PATCH /admin/guru/:id
`mapel_kelas_map` bersifat **full replace** jika dikirim.
**Request:** `{ "nama"?, "email"?, "nip"?, "mapel_kelas_map"? }`
**Response 200:** `data` = `Guru` yang diperbarui.

#### DELETE /admin/guru/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — relasi wali kelas dilepas otomatis.

#### POST /admin/guru/bulk
**Content-Type:** `multipart/form-data`
**Form Fields:** `file` (CSV atau XLSX)

**Format CSV minimal:**
```
nama,nip,email
Ibu Sari,199001012020012001,sari@sekolah.id
```

**Response 200:**
```json
{
  "data": {
    "total": 10,
    "berhasil": 9,
    "gagal": 1,
    "errors": [{ "baris": 5, "pesan": "NIP sudah terdaftar." }]
  },
  "meta": null,
  "error": null
}
```

---

### 12.5 Siswa

#### GET /admin/siswa
**Query Params (opsional):** `kelas_id`, `status`

**Response 200:**
```json
{
  "data": [
    {
      "id": "s1",
      "nama": "Budi Santoso",
      "nis": "1234567890",
      "email": "budi@sekolah.id",
      "kelas_id": "x1",
      "status": "Aktif",
      "is_first_login": false,
      "bergabung": "2026-01-15T00:00:00.000Z",
      "last_login": "2026-05-01T08:30:00.000Z"
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 45, "total_pages": 3 },
  "error": null
}
```

> `status`: `"Aktif"` | `"Belum Aktif"` | `"Nonaktif"`
> Siswa menjadi `"Aktif"` setelah menyelesaikan alur aktivasi (`POST /auth/aktivasi`).

#### GET /admin/siswa/:id
**Response 200:** `data` = satu objek `Siswa`.

#### POST /admin/siswa
**Request:** `{ "nama": "Citra Dewi", "nis": "9876543210", "email": "citra@sekolah.id", "kelas_id": "x1" }`

> `kelas_id` tidak dikirim saat create — siswa ditambahkan ke kelas via
> `POST /admin/kelas/:id/siswa` dari halaman detail kelas.
> `bergabung` / tanggal bergabung — otomatis dari BE, tidak dikirim FE.
> `avatar`, `_tempPassword` tidak boleh dikirim FE di request ini.

**Response 201:**
```json
{
  "data": {
    "id": "s1",
    "nama": "Citra Dewi",
    "nis": "9876543210",
    "email": "citra@sekolah.id",
    "kelas_id": "x1",
    "status": "Belum Aktif",
    "is_first_login": true,
    "temp_password": "x7mK2pQn",
    "bergabung": "2026-05-27T00:00:00.000Z",
    "last_login": null,
    "avatar": null
  },
  "meta": null,
  "error": null
}
```

> `temp_password`: BE generate, dikembalikan **sekali** di response create — FE tampilkan ke admin di drawer/modal.
> BE juga mengirim `temp_password` ke email siswa secara async.
> Setelah siswa aktivasi akun, `temp_password` tidak lagi relevan dan tidak ada di response lain.

#### PATCH /admin/siswa/:id
Jika `kelas_id` berubah, relasi kelas lama dilepas otomatis.
**Request Body (partial):** `{ "nama"?, "kelas_id"? }`
**Response 200:** `data` = `Siswa` yang diperbarui.

#### DELETE /admin/siswa/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

#### POST /admin/siswa/bulk
**Content-Type:** `multipart/form-data`
**Form Fields:** `file` (CSV atau XLSX), `kelas_id` (opsional)

**Format CSV minimal:**
```
nama,nis,email
Budi Santoso,1234567890,budi@sekolah.id
```

**Response 200:** sama dengan `/admin/guru/bulk`.

---

## 13. GURU — Tim 6 BE

> Endpoint profil dan data guru yang diakses oleh guru itu sendiri.

### GET /guru/:id
Profil guru. Guru hanya bisa mengakses profil sendiri; admin bisa akses semua.

**Auth:** role `guru` (hanya `id` sendiri) atau `admin`

**Response 200:**
```json
{
  "data": {
    "id": "g1",
    "nama": "Ibu Sari",
    "nip": "199001012020012001",
    "email": "sari@sekolah.id",
    "avatar": null,
    "mapel_kelas_map": { "mat": ["x1", "x2"] },
    "kelas_aktif": [
      { "id": "x1", "nama": "X-1", "tingkat": "X" },
      { "id": "x2", "nama": "X-2", "tingkat": "X" }
    ],
    "mapel_aktif": [
      { "id": "mat", "label": "Matematika", "icon": "📐" }
    ]
  },
  "meta": null,
  "error": null
}
```

---

### GET /guru/:id/konten
Riwayat konten yang pernah dipublish guru.

**Auth:** role `guru` (hanya `id` sendiri)

**Query Params:** `mapel_id`?, `kelas_id`?, `page`?, `limit`?

**Response 200:**
```json
{
  "data": [
    {
      "publish_id": "pub_mat_bil_aljabar_x1_20260501",
      "mapel_id": "mat",
      "mapel_label": "Matematika",
      "mapel_icon": "📐",
      "elemen_id": "bil_aljabar",
      "elemen_label": "Bilangan dan Aljabar",
      "materi": "Persamaan Linear",
      "materi_id": "mat__persamaan_linear",
      "kelas_id": "x1",
      "kelas_nama": "X-1",
      "jenjang": "X",
      "atp": "Siswa mampu...",
      "published_at": "2026-05-01T09:00:00.000Z",
      "game_penyelesaian": [
        {
          "level": "low",
          "game_id": "game_1746342000_low",
          "siswa_selesai": [
            { "siswa_id": "s1", "nama": "Budi Santoso", "selesai_at": "2026-05-01T10:00:00.000Z" }
          ]
        }
      ]
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 5, "total_pages": 1 },
  "error": null
}
```

> `game_penyelesaian` hanya berisi siswa yang **benar-benar menyelesaikan** game (bukan yang hanya membuka).
> Di UI guru: level yang diselesaikan tampil normal; level yang tidak diselesaikan **berwarna slate**.

---

### GET /kelas/:id/progress — Tim 6 BE
Progress belajar semua siswa dalam satu kelas untuk satu mapel. Digunakan guru sebagai **initial load** sebelum WebSocket aktif, dan sebagai **fallback polling** jika WebSocket tidak tersedia.

**Auth:** role `guru` (hanya kelas yang diampu) atau `admin`

**Query Params:**
- `mapel_id` (wajib jika guru mengampu >1 mapel di kelas ini; opsional jika hanya 1 mapel)

**Response 200:**
```json
{
  "data": {
    "kelas_id": "x1",
    "mapel_id": "mat",
    "total_siswa": 30,
    "aktif_hari_ini": 12,
    "rata_rata_progress": 68,
    "siswa": [
      {
        "siswa_id": "s1",
        "nama": "Budi Santoso",
        "avatar": null,
        "elemen_id": "bil_aljabar",
        "elemen_label": "Bilangan dan Aljabar",
        "materi": "Persamaan Linear",
        "materi_id": "mat__persamaan_linear",
        "level": "mid",
        "nilai_terakhir": 84,
        "durasi_menit": 45,
        "last_active": "2026-05-01T09:15:00.000Z",
        "aktif": true
      }
    ]
  },
  "meta": null,
  "error": null
}
```

> - `aktif_hari_ini`: jumlah siswa yang membuka chatbot hari ini (berdasarkan `sesi.dimulai_at` hari berjalan WIB).
> - `siswa[].aktif: true` = siswa sedang aktif di chatbot saat ini. Siswa yang belum pernah aktif ditandai dengan `nilai_terakhir: null` dan `durasi_menit: 0`.
> - Response ini adalah **snapshot** — tidak real-time. Untuk data real-time gunakan WebSocket.

**Error 400:** `mapel_id` wajib diisi jika guru mengampu lebih dari 1 mapel di kelas ini.
**Error 403:** Guru tidak mengampu kelas ini.

---

## 14. SISWA — Tim 6 BE

### GET /siswa/:id
**Auth:** role `siswa` (hanya `id` sendiri), `guru`, atau `admin`

**Response 200:**
```json
{
  "data": {
    "id": "s1",
    "nama": "Budi Santoso",
    "nis": "1234567890",
    "email": "budi@sekolah.id",
    "kelas_id": "x1",
    "kelas_nama": "X-1",
    "status": "Aktif",
    "avatar": null,
    "bergabung": "2026-01-15T00:00:00.000Z",
    "last_login": "2026-05-01T08:30:00.000Z"
  },
  "meta": null,
  "error": null
}
```

---

### GET /siswa/:id/kpi
KPI dashboard siswa — streak, topik, poin quiz, durasi. Dipakai di Hero Banner dashboard.

**Auth:** role `siswa` (hanya `id` sendiri)

**Response 200:**
```json
{
  "data": {
    "siswa_id": "s1",
    "streak_hari": 5,
    "total_topik": 8,
    "total_poin_quiz": 420,
    "total_durasi_menit": 195
  },
  "meta": null,
  "error": null
}
```

> **Formula poin quiz:** `Σ (mc_score × 60% + essay_score × 40%)` di semua sesi.
> **Total topik:** jumlah elemen/materi unik yang pernah dipelajari.

---

### GET /siswa/:id/progress
Progress belajar siswa per mapel dan elemen. Dipakai di dashboard + ProgressSection.

**Auth:** role `siswa` (hanya `id` sendiri)

**Query Params:** `mapel_id`? (opsional, filter per mapel)

> **Formula `progress_pct`:** `round(selesai / (selesai + dalam_proses + belum_dimulai) × 100)` — integer, dihitung BE.

**Response 200:**
```json
{
  "data": {
    "siswa_id": "s1",
    "by_mapel": [
      {
        "mapel_id": "mat",
        "mapel_label": "Matematika",
        "mapel_icon": "📐",
        "selesai": 2,
        "dalam_proses": 1,
        "belum_dimulai": 3,
        "progress_pct": 33,
        "elemen": [
          {
            "elemen_id": "bil_aljabar",
            "elemen_label": "Bilangan dan Aljabar",
            "status": "selesai",
            "level_terakhir": "high",
            "nilai_agregasi_terakhir": 88,
            "materi": [
              {
                "materi_id": "mat__persamaan_linear",
                "materi_label": "Persamaan Linear",
                "status": "selesai",
                "level_terakhir": "high",
                "nilai_agregasi_terakhir": 88
              }
            ]
          }
        ]
      }
    ],
    "sudah_selesai_ids": ["pub_mat_bil_aljabar_x1_20260501", "pub_mat_data_statistika_x1_20260501"],
    "sedang_dipelajari_ids": ["pub_mat_geometri_x1_20260501"]
  },
  "meta": null,
  "error": null
}
```

> `sudah_selesai_ids` dan `sedang_dipelajari_ids` dipakai sebagai input ke `POST /rag/rekomendasi`.

---

### GET /siswa/:id/konten
Semua paket konten yang sudah dipublish guru untuk kelas siswa ini.

**Auth:** role `siswa` (hanya `id` sendiri)

**Query Params:** `mapel_id`?, `elemen_id`?, `materi_id`?

**Response 200:**
```json
{
  "data": [
    {
      "publish_id": "pub_mat_bil_aljabar_x1_20260501",
      "initial_level": "low",
      "current_level": "mid",
      "mapel_id": "mat",
      "mapel_label": "Matematika",
      "mapel_icon": "📐",
      "elemen_id": "bil_aljabar",
      "elemen_label": "Bilangan dan Aljabar",
      "materi": "Persamaan Linear",
      "materi_id": "mat__persamaan_linear",
      "kelas_id": "x1",
      "jenjang": "X",
      "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel.",
      "konten_list": [
        {
          "konten_id": "konten_mat_bacaan_low_1746342000",
          "tipe": "bacaan",
          "level": "low",
          "content": {
            "judul": "Memahami Persamaan Linear dalam Konteks Kehidupan Sehari-hari",
            "text": "### Persamaan Linear di Sekitar Kita\n\n...",
            "source": "Matematika SMA Kelas X",
            "image_path": "assets/extracted/mat/grafik_persamaan_linear.png"
          }
        },
        {
          "konten_id": "konten_mat_quiz_pg_low_1746342000",
          "tipe": "quiz_pg",
          "level": "low",
          "content": {
            "soal": [
              {
                "id": "q1",
                "stimulus": "Penyebaran informasi hoaks...",
                "image_path": "assets/extracted/mat/grafik_eksponen_01.png",
                "soal": "Berapakah nilai x...",
                "pilihan": ["1", "2", "3", "4"],
                "jawaban": 1,
                "penjelasan": "..."
              }
            ]
          }
        },
        {
          "konten_id": "konten_mat_quiz_essay_low_1746342000",
          "tipe": "quiz_essay",
          "level": "low",
          "content": {
            "pertanyaan": [
              {
                "id": "e1",
                "stimulus": "Sebuah perusahaan rintisan...",
                "image_path": "assets/extracted/mat/diagram_undangan.png",
                "soal": "Bagaimana kamu dapat...",
                "rubrik": "...",
                "penjelasan": "..."
              }
            ]
          }
        },
        {
          "konten_id": "konten_mat_flashcard_low_1746342000",
          "tipe": "flashcard",
          "level": "low",
          "content": {
            "cards": [
              { "depan": "Persamaan Linear", "belakang": "Persamaan berderajat satu dengan satu atau lebih variabel" }
            ],
            "source": "Matematika SMA Kelas X"
          }
        },
        {
          "konten_id": "konten_mat_mindmap_1746342000",
          "tipe": "mindmap",
          "level": null,
          "content": {
            "nodes": [
              { "id": "n1", "label": "Persamaan Linear", "parent_id": null, "penjelasan": "" },
              { "id": "n2", "label": "Pengertian", "parent_id": "n1", "penjelasan": "Persamaan berderajat satu dengan satu variabel." }
            ]
          }
        },
        {
          "game_id": "game_1746342000_low",
          "tipe": "game",
          "level": "low",
          "content": {
            "status": "ready",
            "html_string": "<!DOCTYPE html>...",
            "game_selesai": true,
            "selesai_at": "2026-05-01T10:00:00.000Z"
          }
        }
      ]
    }
  ],
  "meta": null,
  "error": null
}
```

> **Catatan konten_list:** 16 item total per paket: `bacaan×3 + quiz_pg×3 + quiz_essay×3 + flashcard×3 + mindmap×1 + game×3`
>
> FE memfilter `konten_list` berdasarkan level siswa saat ini (hasil pretest).
> Field `game_selesai: true` berarti siswa sudah menyelesaikan game di level tersebut; `false` atau `null` berarti belum.
> `GET /game/:id` hanya digunakan guru saat polling pra-publish — siswa tidak perlu memanggil endpoint ini.
> Field `image_path` pada `bacaan`, `quiz_pg`, dan `quiz_essay` bersifat opsional — `null` atau omit jika tidak ada gambar. FE render via `<img src={image_path}>` tanpa transformasi. BE wajib menyimpan dan mengembalikan field ini apa adanya.
> `initial_level`: level siswa saat pertama kali pretest untuk topik ini — digunakan FE untuk menampilkan konten bacaan. Tidak berubah walaupun siswa naik level.
> `current_level`: level aktif siswa saat ini — digunakan FE untuk memfilter quiz, essay, flashcard, dan game. Berubah saat siswa naik level.
> Jika siswa belum pernah pretest untuk topik ini, `initial_level` dan `current_level` bernilai `"low"`.

---

### GET /siswa/:id/pretest/status
Status pretest siswa untuk semua elemen/materi dalam satu mapel.

**Auth:** role `siswa` (hanya `id` sendiri)

**Query Params:** `mapel_id` (wajib)

**Response 200:**
```json
{
  "data": [
    { "elemen_id": "bil_aljabar", "materi_id": null, "status": "selesai", "level": "mid" },
    { "elemen_id": "geometri", "materi_id": null, "status": "belum", "level": null },
    { "elemen_id": "data_statistika", "materi_id": "mat__statistika_deskriptif", "status": "selesai", "level": "low" }
  ],
  "meta": null,
  "error": null
}
```

> - `materi_id: null` = status untuk elemen langsung (tanpa sub-materi)
> - `status`: `"belum"` | `"selesai"`
> - `level`: `"low"` | `"mid"` | `"high"` jika `status: "selesai"`, `null` jika `"belum"`

**Error 400:** "mapel_id wajib diisi."

---

### GET /siswa/:id/quiz
Riwayat quiz siswa per elemen/materi, dikelompokkan per level.

**Auth:** role `siswa` (hanya `id` sendiri)

**Query Params:** `elemen_id` (wajib), `materi_id` (opsional)

**Response 200:**
```json
{
  "data": {
    "current_level": "mid",
    "riwayat": [
      { "hasil_quiz_id": "hq_20260501_0001", "tipe": "mc", "level": "low", "nilai": 85, "terkunci": true, "dikerjakan_at": "2026-05-01T09:00:00.000Z" },
      { "hasil_quiz_id": "hq_20260501_0002", "tipe": "essay", "level": "low", "nilai": 78, "terkunci": true, "dikerjakan_at": "2026-05-01T09:10:00.000Z" },
      { "hasil_quiz_id": "hq_20260501_0003", "tipe": "mc", "level": "mid", "nilai": 60, "terkunci": false, "dikerjakan_at": "2026-05-01T10:00:00.000Z" }
    ]
  },
  "meta": null,
  "error": null
}
```

> - `riwayat[].terkunci: true` = level sudah dilewati → quiz level ini **read-only**.
> - `riwayat[].hasil_quiz_id`: dipakai FE untuk trigger CTA "Tanya Mentor AI".

**Response jika belum pernah quiz:**
```json
{ "data": { "current_level": "low", "riwayat": [] }, "meta": null, "error": null }
```

**Error 400:** "elemen_id wajib diisi."

---

### POST /siswa/:id/quiz/mc — Tim 6 BE
Submit jawaban Quiz Pilihan Ganda. BE menilai langsung karena kunci jawaban tersedia.

**Auth:** role `siswa` (hanya `id` sendiri)

**Request:**
```json
{
  "siswa_id": "s1",
  "publish_id": "pub_mat_bil_aljabar_x1_20260501",
  "level": "Low",
  "jawaban": {
    "q1": "1",
    "q2": "0",
    "q3": "2"
  }
}
```

> - `jawaban`: key = `id` soal. Value = string index pilihan
> - `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id` dihapus dari request — **BE lookup dari `publish_id`** (**V3.8**)
> - **BE menghitung nilai sendiri** — FE tidak mengirim `score`

**Response 200:**
```json
{
  "data": {
    "tipe": "mc",
    "nilai": 80,
    "benar": 8,
    "total": 10,
    "elemen_id": "bil_aljabar",
    "level": "low",
    "naik_level": false,
    "agregasi": null,
    "menunggu_essay": true,
    "kkm": 75,
    "hasil_quiz_id": "hq_20260501_0001",
    "dicatat_at": "2026-05-01T09:15:00.000Z"
  },
  "meta": null,
  "error": null
}
```

> - `naik_level`: selalu `false` dari endpoint ini — naik level baru ditentukan setelah agregasi MC+Essay selesai. BE push via WebSocket event `essay_dinilai`.

---

### POST /siswa/:id/quiz/essay — Tim 6 BE + Tim 3 RAG
Submit jawaban Quiz Essay. BE forward ke Tim 3 RAG untuk dinilai secara asinkronus.

**Auth:** role `siswa` (hanya `id` sendiri)

**Request:**
```json
{
  "siswa_id": "s1",
  "publish_id": "pub_mat_bil_aljabar_x1_20260501",
  "level": "Low",
  "jawaban": {
    "e1": "Langkah pertama adalah memindahkan konstanta ke ruas kanan...",
    "e2": "Variabel adalah simbol yang mewakili nilai yang tidak diketahui..."
  }
}
```

> - `jawaban`: key = `id` soal essay. Value = string teks jawaban siswa
> - `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id` dihapus dari request — **BE lookup dari `publish_id`** (**V3.8**)

**Response 200:**
```json
{
  "data": {
    "tipe": "essay",
    "nilai": null,
    "elemen_id": "bil_aljabar",
    "level": "low",
    "menunggu_penilaian": true,
    "naik_level": null,
    "agregasi": null,
    "hasil_quiz_id": "hq_20260501_0002",
    "dicatat_at": "2026-05-01T09:20:00.000Z"
  },
  "meta": null,
  "error": null
}
```

> **Logika agregasi & naik level (di BE, otomatis setelah essay dinilai):**
> ```
> agregasi = nilai_mc × 60% + nilai_essay × 40%
> naik_level = agregasi >= 75
> ```
> BE push hasil via WebSocket event `essay_dinilai` → FE update UI naik level tanpa polling.

---

### GET /siswa/:id/notifikasi
**Auth:** role `siswa` (hanya `id` sendiri)

**Query Params:** `dibaca`? (`true`|`false`), `page`?, `limit`?

**Response 200:**
```json
{
  "data": [
    {
      "id": "notif_123",
      "guru_nama": "Ibu Sari",
      "guru_mapel": "📐 Matematika",
      "pesan": "Coba ulangi materi persamaan linear.",
      "dibaca": false,
      "dibuat_at": "2026-05-01T09:00:00.000Z"
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 3, "total_pages": 1 },
  "error": null
}
```

---

## 15. KONTEN — Tim 3 RAG + Tim 6 BE

> **Ownership endpoint:**
> - `POST /konten/generate` → **Tim 3 RAG** — generate konten dari VectorDB
> - `POST /konten/publish` → **Tim 6 BE** — simpan konten ke database MVP

---

### POST /konten/generate — Tim 3 RAG
Guru generate satu tipe konten per request. FE memanggil endpoint ini **14× paralel** saat klik "Generate Konten":
- `bacaan` × 3 level (Low/Mid/High)
- `quiz_pg` × 3 level
- `quiz_essay` × 3 level
- `flashcard` × 3 level
- `mindmap` × 1 (tanpa level)
- `pretest` × 1 (tanpa level)

> Game **tidak** melalui endpoint ini — gunakan `POST /game/generate` (Tim 4).

**Auth:** role `guru`

**Request:**
```json
{
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "kelas_id": "x1",
  "jenjang": "X",
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata.",
  "tipe": "bacaan",
  "level": "Low",
  "konten_id": "konten_mat_bacaan_low_1746342000",
  "instruksi_revisi": "Tambahkan contoh soal cerita"
}
```

> - `materi` dan `materi_id`: opsional — hanya diisi jika guru menentukan sub-materi spesifik
> - `atp`: opsional tapi **sangat disarankan**
> - `level`: `"Low"` | `"Mid"` | `"High"` — **null atau omit** untuk `mindmap` dan `pretest`
> - `konten_id`: opsional — referensi konteks untuk Tim 3 saat regenerate
> - `instruksi_revisi`: opsional — jika ada = regenerate
> - `kelas_id`: **wajib** — digunakan Tim 3 RAG untuk konteks generate konten dan pretest per kelas

**Response 200 (contoh `bacaan`):**
```json
{
  "data": {
    "konten_id": "konten_mat_bacaan_low_1746342000",
    "tipe": "bacaan",
    "level": "low",
    "content": {
      "judul": "Memahami Persamaan Linear dalam Konteks Kehidupan Sehari-hari",
      "text": "### Persamaan Linear di Sekitar Kita\n\n...",
      "source": "Matematika SMA Kelas X",
      "image_path": "assets/extracted/mat/grafik_persamaan_linear.png"
    },
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Response 200 (contoh `quiz_pg`):**
```json
{
  "data": {
    "konten_id": "konten_mat_quiz_pg_low_1746342000",
    "tipe": "quiz_pg",
    "level": "low",
    "content": {
      "soal": [
        {
          "id": "q1",
          "stimulus": "Penyebaran informasi hoaks...",
          "image_path": "assets/extracted/mat/grafik_eksponen_01.png",
          "soal": "Berapakah nilai x dari persamaan 2x + 3 = 7?",
          "pilihan": ["1", "2", "3", "4"],
          "jawaban": 1,
          "penjelasan": "..."
        }
      ]
    },
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Response 200 (contoh `quiz_essay`):**
```json
{
  "data": {
    "konten_id": "konten_mat_quiz_essay_low_1746342000",
    "tipe": "quiz_essay",
    "level": "low",
    "content": {
      "pertanyaan": [
        {
          "id": "e1",
          "stimulus": "Di sebuah desa nelayan...",
          "image_path": "assets/extracted/mat/grafik_penyebaran_nelayan.png",
          "soal": "Bagaimana kamu menggunakan...",
          "rubrik": "...",
          "penjelasan": "..."
        }
      ]
    },
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Response 200 (contoh `mindmap`):**
```json
{
  "data": {
    "konten_id": "konten_mat_mindmap_1746342000",
    "tipe": "mindmap",
    "level": null,
    "content": {
      "nodes": [
        { "id": "n1", "label": "Persamaan Linear", "parent_id": null, "penjelasan": "" },
        { "id": "n2", "label": "Pengertian", "parent_id": "n1", "penjelasan": "Persamaan berderajat satu dengan satu variabel, berbentuk ax + b = c." },
        { "id": "n3", "label": "Langkah Penyelesaian", "parent_id": "n1", "penjelasan": "Isolasi variabel dengan operasi aljabar yang sama di kedua ruas." }
      ]
    },
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Response 200 (contoh `pretest`):**
```json
{
  "data": {
    "konten_id": "konten_mat_pretest_1746342000",
    "tipe": "pretest",
    "level": null,
    "content": {
      "soal": [
        {
          "id": "pretest_mat_bil_aljabar_1",
          "tingkat_kesulitan": "low",
          "stimulus": "Penyebaran informasi hoaks di media sosial seringkali sangat cepat. Jika satu orang menyebarkan hoaks kepada dua orang, dan setiap penerima menyebarkannya lagi kepada dua orang lain, pada fase pertama ada 2 penerima, fase kedua 4, fase ketiga 8.",
          "soal": "Bagaimana cara paling ringkas untuk menyatakan perkalian berulang bilangan 2 sebanyak 5 kali dalam notasi eksponen?",
          "pilihan": [
            "Menulisnya sebagai 2 x 5",
            "Menyatakannya dengan 2^5",
            "Menggunakan bentuk 5^2",
            "Menulisnya sebagai 2 + 2 + 2 + 2 + 2",
            "Menggunakan notasi 5 x 2"
          ],
          "jawaban": 1
        }
      ]
    },
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

> - `tipe: "pretest"` → Tim 3 generate 5 soal deterministik dengan distribusi low×2, mid×2, high×1
> - `level: null` — pretest tidak berlevel sebagai paket
> - Response ini **diabaikan FE** — Tim 3 menyimpan soal langsung ke database Tim 6 BE

---

**Struktur `content` per `tipe`:**

| tipe | Struktur `content` | Jumlah |
|------|-------------------|--------|
| `bacaan` | `{ "judul": string, "text": "markdown string", "source": string, "image_path": string\|null }` | — |
| `quiz_pg` | `{ "soal": [{ "id", "stimulus": string, "image_path": string\|null, "soal", "pilihan": string[], "jawaban": number, "penjelasan": string }] }` | 10 soal |
| `quiz_essay` | `{ "pertanyaan": [{ "id", "stimulus": string, "image_path": string\|null, "soal", "rubrik": string, "penjelasan": string }] }` | 5 pertanyaan |
| `flashcard` | `{ "cards": [{ "depan", "belakang" }], "source": string }` | 5–10 kartu |
| `mindmap` | `{ "nodes": [{ "id", "label", "parent_id", "penjelasan": string }] }` | — |
| `pretest` | `{ "soal": [{ "id", "tingkat_kesulitan": "low"\|"mid"\|"high", "stimulus": string, "soal", "pilihan": string[], "jawaban": number }] }` | 5 soal (low×2, mid×2, high×1) |

> **Catatan field konten:**
>
> `bacaan.judul` = judul utama konten (wajib, non-empty). Tim 3 RAG mengisi dari field `judul_utama` Tim 2. FE menggunakan ini sebagai elemen header komponen bacaan — tidak boleh fallback ke `elemen_label`.
>
> `quiz_pg.soal[].stimulus` = teks konteks/skenario (50–150 kata, wajib). Tim 3 RAG mengisi dari `stimulus` Tim 2. FE merender di atas soal, sebelum pertanyaan dan pilihan ditampilkan.
>
> `quiz_pg.soal[].pilihan` = **wajib array string** dengan 4–5 elemen. Tim 3 RAG wajib mengkonversi format `options: {A,B,C,D,E}` dari Tim 2 menjadi array. FE tidak melakukan konversi ini.
>
> `quiz_pg.soal[].jawaban` = **index integer** (0-based). Tim 3 RAG wajib mengkonversi `"A"→0, "B"→1, "C"→2, "D"→3, "E"→4` dari format Tim 2.
>
> `quiz_pg.soal[].penjelasan` = ditampilkan FE **setelah** siswa submit quiz MC, tidak saat mengerjakan.
>
> `quiz_essay.pertanyaan[].stimulus` = teks konteks (50–200 kata, wajib). FE merender di atas soal.
>
> `quiz_essay.pertanyaan[].rubrik` = 1 string deskriptif per soal yang mengandung kriteria + panduan poin. Tim 3 RAG normalize dari array `rubric_points` Tim 2 menjadi narasi 1 string. **Digunakan eksklusif Tim 3 RAG sebagai acuan penilaian essay siswa — tidak ditampilkan ke siswa maupun guru.**
>
> `quiz_essay.pertanyaan[].penjelasan` = pembahasan kunci jawaban. **Tidak ditampilkan ke siswa.** Otomatis ter-inject ke Tim 5 via mekanisme `hasil_quiz_id` lookup di `POST /mentor/evaluasi`.
>
> `quiz_essay.pertanyaan[].id` = **wajib di-generate BE** (format: `"e1"`, `"e2"`, dst.) — tidak boleh berasal langsung dari AI output Tim 2.
>
> `mindmap.nodes[]` = **wajib flat array**. Tim 3 RAG melakukan DFS flatten dari nested tree Tim 2 (`root→children→children`) sebelum menyimpan ke DB. FE dilarang melakukan rekursi flatten. Node root memiliki `parent_id: null`. Field `name` Tim 2 dipetakan ke `label`; field `description` dipetakan ke `penjelasan`. `id` node di-generate Tim 3 (format: `"n1"`, `"n2"`, dst.).
>
> `flashcard.cards[].depan` / `belakang` = Tim 3 RAG rename dari `front`/`back` format Tim 2. Konten LaTeX inline `$...$` dipertahankan.
>
> `pretest.soal[].tingkat_kesulitan` = level kesulitan per soal (`"low"/"mid"/"high"`). Tim 3 RAG mengkonversi `LOTS→"low"`, `MOTS→"mid"`, `HOTS→"high"` dari Tim 2. **FE tidak merender badge ini ke siswa** — hanya digunakan BE untuk algoritma Lowest Failed Level. BE lookup nilai ini dari database saat `POST /pretest/submit`.
>
> `bacaan.source` & `flashcard.source` = kosong `""` jika tidak ada sumber spesifik.
> `bacaan.image_path` = path relatif gambar pendukung konten bacaan (opsional — `null` atau omit jika tidak ada gambar). Contoh: `"assets/extracted/mat/grafik_persamaan_linear.png"`. Tim 3 RAG mengisi dari hasil ekstraksi dokumen sumber. File sudah tersedia di folder static FE di VPS sebelum sistem berjalan. FE render via `<img src={image_path}>` — browser resolve relatif terhadap domain FE. BE simpan dan kembalikan apa adanya (pass-through murni).
>
> `quiz_pg.soal[].image_path` = path relatif gambar ilustrasi per soal PG (opsional — `null` atau omit jika tidak ada). Posisi render: di antara `stimulus` dan teks `soal`. Tim 3 RAG mengisi jika dokumen sumber memiliki gambar yang relevan dengan soal tersebut.
>
> `quiz_essay.pertanyaan[].image_path` = path relatif gambar ilustrasi per soal essay (opsional — `null` atau omit jika tidak ada). Posisi render: di antara `stimulus` dan teks `soal`. Tim 3 RAG mengisi jika dokumen sumber memiliki gambar yang relevan.
>
> **Aturan render `image_path` (wajib dipatuhi Tim 6 FE):** Gunakan langsung sebagai nilai atribut `src` — `<img src={image_path} alt="..." />`. Browser otomatis resolve path relatif terhadap domain FE. Tidak perlu prefix, tidak perlu konversi URL, tidak perlu env var. Tambahkan `onError` handler untuk graceful fallback jika file tidak ditemukan. Field opsional — guard null sebelum render.
>
> **Peran BE MVP untuk `image_path`:** Pass-through murni — simpan dan kembalikan string apa adanya. BE tidak melakukan validasi path, serve file, konversi URL, atau transformasi apapun.
>
> **Peran Tim 3 RAG untuk `image_path`:** Ekstraksi gambar dilakukan saat ingest dokumen ke Qdrant. File gambar di-deploy ke folder static FE di VPS (`/assets/extracted/...`). Path di Qdrant disesuaikan dengan lokasi file di VPS sehingga langsung bisa dirender browser.

---

**Catatan Pretest (V3.7):**

FE mengirim **1 request eksplisit** dengan `tipe: "pretest"` sebagai bagian dari batch 14 panggilan paralel (13 konten + 1 pretest). Tim 3 RAG men-generate 5 soal pretest berdasarkan konteks `mapel_id + elemen_id + materi_id + kelas_id` dan menyimpannya langsung ke database Tim 6 BE. Response dari request `tipe: "pretest"` **tidak digunakan FE** (diabaikan / `.catch()` silent). Pretest **tidak ditampilkan** di panel review guru.

Request pretest eksplisit:
```json
{
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "kelas_id": "x1",
  "jenjang": "X",
  "atp": "Siswa mampu ...",
  "tipe": "pretest",
  "level": null
}
```

Tim 3 cukup return `{ "status": "ok" }` atau HTTP 200 kosong — FE tidak membaca response ini.

**Error 422:** "elemen_id tidak dikenal di VectorDB."
**Error 429:** "Terlalu banyak request. Coba beberapa saat lagi."

---

### POST /konten/publish — Tim 6 BE
Guru publish paket konten ke siswa setelah **semua item disetujui**. Konten disimpan permanen di database MVP. **Tidak bisa di-publish ulang** setelah publish pertama.

> **Atomik:** Operasi ini atomik — jika gagal di tengah jalan, tidak ada data yang tersimpan parsial. FE aman untuk retry.

**Auth:** role `guru`

**Request:**
```json
{
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "kelas_id": "x1",
  "jenjang": "X",
  "guru_id": "g1",
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata.",
  "konten_list": [
    {
      "konten_id": "konten_mat_bacaan_low_1746342000",
      "tipe": "bacaan",
      "level": "Low",
      "content": {
        "judul": "Memahami Persamaan Linear dalam Konteks Kehidupan Sehari-hari",
        "text": "### Persamaan Linear di Sekitar Kita\n...",
        "source": "Matematika SMA Kelas X",
        "image_path": "assets/extracted/mat/grafik_persamaan_linear.png"
      },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_quiz_pg_low_1746342000",
      "tipe": "quiz_pg",
      "level": "Low",
      "content": {
        "soal": [{
          "id": "q1",
          "stimulus": "...",
          "image_path": "assets/extracted/mat/grafik_eksponen_01.png",
          "soal": "...",
          "pilihan": ["..."],
          "jawaban": 1,
          "penjelasan": "..."
        }]
      },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_quiz_essay_low_1746342000",
      "tipe": "quiz_essay",
      "level": "Low",
      "content": {
        "pertanyaan": [{
          "id": "e1",
          "stimulus": "...",
          "image_path": "assets/extracted/mat/grafik_eksponen_01.png",
          "soal": "...",
          "rubrik": "...",
          "penjelasan": "..."
        }]
      },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_flashcard_low_1746342000",
      "tipe": "flashcard",
      "level": "Low",
      "content": { "cards": [{ "depan": "...", "belakang": "..." }], "source": "..." },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_mindmap_1746342000",
      "tipe": "mindmap",
      "level": null,
      "content": { "nodes": [{ "id": "n1", "label": "...", "parent_id": null, "penjelasan": "" }] },
      "disetujui": true
    },
    {
      "game_id": "game_1746342000_low",
      "tipe": "game",
      "level": "Low",
      "content": {
        "status": "ready",
        "html_string": "<!DOCTYPE html><html>...</html>"
      },
      "disetujui": true
    }
  ]
}
```

> `konten_list` berisi **16 item total**: `bacaan×3 + quiz_pg×3 + quiz_essay×3 + flashcard×3 + mindmap×1 + game×3`
>
> FE **hanya boleh publish jika semua 16 item `disetujui: true`**.
> Item `game` menggunakan `game_id` — tidak ada `konten_id` untuk item game.
> `html_string` untuk game dikirim penuh saat publish agar BE menyimpannya ke database tanpa perlu call ke Tim 4 lagi.

**Response 201:**
```json
{
  "data": {
    "publish_id": "pub_mat_bil_aljabar_x1_20260501",
    "kelas_ids": ["x1"],
    "dipublish_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Error 400:** "Semua konten harus disetujui sebelum publish."
**Error 409:** "Konten untuk elemen ini sudah pernah dipublish ke kelas ini."

---

## 16. SESI — Tim 6 BE

> Sesi belajar adalah unit terkecil dari aktivitas siswa. Dimulai saat siswa membuka chatbot, berakhir saat menutup atau timeout.

### POST /sesi
Mulai sesi belajar baru. Dipanggil saat siswa membuka chatbot (setelah izin kamera diberikan).

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "publish_id": "pub_mat_bil_aljabar_x1_20260501"
}
```

> **V3.8:** `mapel_id`, `elemen_id`, `materi_id` dihapus dari request — BE lookup dari `publish_id`.

**Response 201:**
```json
{
  "data": {
    "sesi_id": "sesi_s1_20260501_bil_aljabar",
    "dimulai_at": "2026-05-01T09:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

---

### PATCH /sesi/:id
Update sesi (durasi, violations, emosi akhir). Dipanggil saat siswa menutup chatbot.

**Auth:** role `siswa`

**Request:**
```json
{
  "durasi_menit": 45,
  "emosi_akhir": "antusias",
  "violations": [
    { "detail": "Berpindah Tab / Menyembunyikan Halaman", "terjadi_at": "2026-05-01T09:30:00.000Z" }
  ]
}
```

**Response 200:**
```json
{
  "data": {
    "sesi_id": "sesi_s1_20260501_bil_aljabar",
    "durasi_menit": 45,
    "selesai_at": "2026-05-01T09:45:00.000Z"
  },
  "meta": null,
  "error": null
}
```

---

### POST /sesi/:id/summary — Tim 3 RAG
Generate summary AI untuk satu sesi belajar siswa. Dipanggil guru dari panel detail drawer monitoring.

**Auth:** role `guru`

**Request:**
```json
{
  "siswa_id": "s1",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "materi_id": "mat__persamaan_linear",
  "durasi_menit": 45,
  "hasil_quiz": [
    { "level": "low", "tipe": "mc", "nilai": 80 },
    { "level": "low", "tipe": "essay", "nilai": 72 },
    { "level": "mid", "tipe": "mc", "nilai": 60 }
  ],
  "last_quiz": {
    "nilai_mc": 60,
    "nilai_essay": 72,
    "agregasi": 64.8
  },
  "emosi_sesi": ["antusias", "bingung", "antusias"],
  "violations": [
    { "detail": "Berpindah Tab / Menyembunyikan Halaman", "terjadi_at": "2026-05-01T09:30:00.000Z" }
  ]
}
```

**Response 200:**
```json
{
  "data": {
    "teks": "Summary sesi 2026-05-01 — Budi Santoso:\n\nMateri level Menengah sudah dicoba namun perlu pendalaman pada soal essay...",
    "dibuat_at": "2026-05-01T09:50:00.000Z",
    "berlaku_hingga": "2026-05-02T09:50:00.000Z"
  },
  "meta": null,
  "error": null
}
```

**Error 422:** "Data sesi tidak cukup untuk menghasilkan evaluasi."

---

### GET /sesi/:id/emosi
**Auth:** role `siswa` (sesi sendiri) atau `guru`

**Response 200:**
```json
{
  "data": [
    { "emosi": "antusias", "confidence": 0.91, "terdeteksi_at": "2026-05-01T09:05:00.000Z" },
    { "emosi": "bingung",  "confidence": 0.84, "terdeteksi_at": "2026-05-01T09:20:00.000Z" },
    { "emosi": "antusias", "confidence": 0.78, "terdeteksi_at": "2026-05-01T09:35:00.000Z" }
  ],
  "meta": null,
  "error": null
}
```

> Guru hanya melihat log saat terjadi **perubahan emosi** — data sudah disaring BE.

---

### GET /sesi/:id/chat
**Auth:** role `siswa` (sesi sendiri)

**Response 200:**
```json
{
  "data": [
    { "role": "user", "teks": "Aku bingung cara menyelesaikan 2x + 3 = 7", "dikirim_at": "2026-05-01T09:10:00.000Z" },
    { "role": "ai", "teks": "Tenang ya, kita mulai dari yang paling dasar...", "dikirim_at": "2026-05-01T09:10:05.000Z" }
  ],
  "meta": null,
  "error": null
}
```

---

## 17. PRETEST — Tim 3 RAG (generate) + Tim 6 BE (serve)

> Pretest **berbeda** dari quiz MC & essay di chatbot. Dipakai sekali untuk menentukan level awal konten siswa.
> **Tim 3 RAG** meng-generate soal pretest bersamaan dengan generate konten (via request eksplisit FE, response diabaikan).
> **Tim 6 BE** menyimpan dan melayani soal pretest ke siswa.
> `GET /siswa/:id/pretest/status` → **Tim 6 BE** (lihat Section 12)

### POST /pretest/soal — Tim 6 BE
Ambil 5 soal pretest untuk elemen/materi yang akan dipelajari.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "materi_id": "mat__persamaan_linear"
}
```

> `materi_id` opsional — jika tidak ada, pretest untuk level elemen.

**Response 200:**
```json
{
  "data": {
    "sesi_pretest_id": "pretest_1746342000_bil_aljabar",
    "soal": [
      {
        "id": "pretest_mat_bil_aljabar_1",
        "tingkat_kesulitan": "low",
        "stimulus": "Penyebaran informasi hoaks di media sosial seringkali sangat cepat. Jika satu orang menyebarkan kepada dua orang, dan setiap penerima menyebarkan lagi kepada dua orang lain, pada fase pertama ada 2 penerima, fase kedua 4, fase ketiga 8.",
        "soal": "Bagaimana cara paling ringkas untuk menyatakan perkalian berulang bilangan 2 sebanyak 5 kali?",
        "pilihan": ["Menulisnya sebagai 2 x 5", "Menyatakannya dengan 2^5", "Menggunakan bentuk 5^2", "Menulisnya sebagai 2 + 2 + 2 + 2 + 2", "Menggunakan notasi 5 x 2"],
        "jawaban": 1
      }
    ]
  },
  "meta": null,
  "error": null
}
```

> - Soal disajikan dengan urutan acak. `tingkat_kesulitan` ada di response agar FE bisa merender `stimulus` per soal, tetapi **FE tidak menampilkan badge level ke siswa**.
> - **Error 404:** `NOT_FOUND` — "Soal pretest untuk elemen ini belum tersedia. Pastikan guru sudah mempublish konten untuk elemen ini."

---

### POST /pretest/submit — Tim 6 BE
Submit jawaban pretest, dapatkan level awal siswa.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "materi_id": null,
  "sesi_pretest_id": "pretest_1746342000_bil_aljabar",
  "jawaban": {
    "pretest_mat_bil_aljabar_1": "1",
    "pretest_mat_bil_aljabar_2": "3",
    "pretest_mat_bil_aljabar_3": "1",
    "pretest_mat_bil_aljabar_4": "2",
    "pretest_mat_bil_aljabar_5": "3"
  }
}
```

> - `jawaban`: key = `id` soal, value = **string index pilihan** yang dipilih siswa (0-based).
> - BE lookup `tingkat_kesulitan` per soal dari database menggunakan `sesi_pretest_id` — FE tidak perlu mengirimnya.

**Response 200:**
```json
{
  "data": {
    "level": "mid",
    "benar_per_tingkat": {
      "low": 2,
      "mid": 1,
      "high": 0
    },
    "total_benar": 3,
    "total_soal": 5
  },
  "meta": null,
  "error": null
}
```

> **Algoritma Penetapan Level — Lowest Failed Level (dievaluasi BE, bukan FE):**
>
> BE menentukan level berdasarkan prinsip "tingkat kesalahan terendah", dievaluasi **berurutan** dari level paling dasar. `tingkat_kesulitan` per soal diambil BE dari database — FE tidak perlu mengirimnya.
>
> ```
> Distribusi soal: low×2, mid×2, high×1
>
> 1. Periksa soal Low (2 soal):
>    → Jika ada ≥1 salah → level = "low". STOP.
>
> 2. Periksa soal Mid (2 soal):
>    → Jika ada ≥1 salah → level = "mid". STOP.
>
> 3. Periksa soal High (1 soal):
>    → Jika salah → level = "high". STOP.
>    → Jika benar → level = "high". STOP.
> ```
>
> Semua jalur di level High menghasilkan `"high"` — siswa yang menjawab benar semua maupun yang hanya gagal di High sama-sama ditempatkan di level tertinggi, karena fondasi Low dan Mid-nya sudah solid.
>
> BE menyimpan hasil ini permanen. Status dapat diambil via `GET /siswa/:id/pretest/status`.

---

## 18. QUIZ — Tim 6 BE

> Endpoint quiz langsung berada di domain siswa (lihat Section 12):
> - `GET /siswa/:id/quiz` — riwayat quiz
> - `POST /siswa/:id/quiz/mc` — submit quiz MC
> - `POST /siswa/:id/quiz/essay` — submit quiz Essay

Tidak ada endpoint quiz yang berdiri sendiri di domain `/quiz`. Desain ini sengaja dipilih karena quiz adalah **milik siswa**, bukan resource independen.

---

## 19. RAG — Tim 3

> Semua endpoint yang memerlukan komputasi AI dari Tim 3 RAG dikumpulkan di domain `/rag`.

### POST /rag/rekomendasi
Rekomendasi elemen/materi berikutnya berdasarkan progress siswa.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "available_ids": [
    "pub_mat_bil_aljabar_x1_20260501",
    "pub_mat_geometri_x1_20260501",
    "pub_fis_besaran_x1_20260501"
  ],
  "sudah_selesai_ids": [
    "pub_mat_bil_aljabar_x1_20260501",
    "pub_mat_data_statistika_x1_20260501"
  ],
  "sedang_dipelajari_ids": [
    "pub_mat_geometri_x1_20260501"
  ]
}
```

> - `available_ids`: semua `publish_id` konten yang dipublish guru ke kelas siswa ini — Tim 3 gunakan sebagai universe kandidat rekomendasi. BE lookup mapel, elemen, materi, ATP dari `publish_id`.
> - `sudah_selesai_ids`: `publish_id` materi yang sudah selesai (siswa sudah di level `high` + agregasi >= KKM).
> - `sedang_dipelajari_ids`: `publish_id` materi yang sedang dipelajari (sudah dibuka tapi belum selesai).
> - `sudah_selesai_ids` dan `sedang_dipelajari_ids` adalah subset dari `available_ids`.
> - Materi yang ada di `available_ids` tapi tidak di keduanya = belum pernah dibuka → kandidat rekomendasi baru.
> - `levels` dihapus — Tim 3 cukup inferensikan dari posisi di tiga bucket di atas.

**Response 200:**
```json
{
  "data": [
    {
      "mapel_id": "mat",
      "elemen_id": "geometri",
      "elemen_label": "Geometri dan Pengukuran",
      "materi": "Teorema Pythagoras",
      "materi_id": "mat__teorema_pythagoras",
      "alasan": "Kamu sudah menguasai aljabar dasar, saatnya melanjutkan ke geometri."
    }
  ],
  "meta": null,
  "error": null
}
```

> Maksimal 3 item rekomendasi.

---

### POST /rag/insight
Generate teks insight personal untuk Hero Banner dashboard siswa.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "nama": "Budi Santoso",
  "streak": 5,
  "total_topik": 8,
  "total_poin_kuiz": 420,
  "total_durasi": 195
}
```

**Response 200:**
```json
{
  "data": {
    "teks": "🚀 Keren, Budi! Streak 5 hari berturut-turut — kamu konsisten sekali. Yuk lanjutkan momentum ini!"
  },
  "meta": null,
  "error": null
}
```

> Teks: 1–2 kalimat motivasi, dimulai satu emoji.

---

## 20. GAME — Tim 4 + Tim 6 BE

> Tim 4 deliver game dalam format **HTML string** (bukan URL). FE me-render via:
> ```html
> <iframe srcDoc={gameData.html_string} sandbox="allow-scripts allow-same-origin allow-forms" />
> ```
> Game menghasilkan **3 level** (Low/Mid/High). Tracking hanya boolean selesai/tidak selesai.
>
> **Trigger selesai:** Game HTML mengirim event ke parent FE via `window.parent.postMessage`:
> ```javascript
> window.parent.postMessage({ type: 'game:selesai' }, '*');
> ```
> FE listen via `window.addEventListener('message', ...)` dan memanggil `PATCH /game/:id/penyelesaian`.
> FE juga menerima format lama `'game:selesai'` (string) dan `{ event: 'game:selesai' }` untuk kompatibilitas.

### POST /game/generate — Tim 4
Guru generate game baru. Dipanggil **3×** (Low/Mid/High) paralel saat guru klik "Generate Konten".

**Auth:** role `guru`

**Request:**
```json
{
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "kelas_id": "x1",
  "jenjang": "X",
  "atp": "Siswa mampu...",
  "level": "Low",
  "bacaan": {
    "judul": "Memahami Persamaan Linear dalam Konteks Kehidupan Sehari-hari",
    "text": "### Persamaan Linear di Sekitar Kita\n\n..."
  }
}
```

**Response 200:**
```json
{
  "data": {
    "game_id": "game_1746342000_low",
    "nama": "Quest: Persamaan Linear",
    "deskripsi": "Game edukasi interaktif tentang Persamaan Linear — level low",
    "mapel_id": "mat",
    "elemen_id": "bil_aljabar",
    "materi_id": "mat__persamaan_linear",
    "level": "low",
    "status": "ready",
    "html_string": "<!DOCTYPE html><html>...</html>"
  },
  "meta": null,
  "error": null
}
```

> Jika `status: "generating"` → `html_string: null` → FE **poll** `GET /game/:id` setiap 3 detik hingga `status: "ready"`.

**Error 422:** "elemen_id tidak dikenal."

---

### POST /game/regenerate — Tim 4
Guru minta generate ulang game spesifik. Iterative refinement menggunakan konteks game sebelumnya.

**Auth:** role `guru`

**Request:**
```json
{
  "game_id": "game_1746342000_low",
  "instruksi_revisi": "Tambahkan level kesulitan di pertanyaan terakhir dan buat feedback lebih informatif"
}
```

**Response 200:** identik dengan `POST /game/generate`.

**Error 404:** "game_id tidak ditemukan."
**Error 422:** "instruksi_revisi wajib diisi untuk regenerate."

---

### GET /game/:id — Tim 4
Detail satu game. Digunakan untuk polling saat status masih `"generating"` dan preview guru pra-publish.

**Auth:** role `guru`

**Response 200:** identik dengan `POST /game/generate`.

> `GET /game/:id` hanya untuk: (1) polling status generate sebelum publish, (2) preview game sebelum publish. Setelah dipublish, `html_string` sudah tersedia di `GET /siswa/:id/konten`.

---

### PATCH /game/:id/penyelesaian — Tim 6 BE
Catat bahwa siswa **menyelesaikan** game.

**Auth:** role `siswa`

**Request:** `{ "siswa_id": "s1", "level": "Low" }`

**Response 200:**
```json
{
  "data": {
    "tercatat": true,
    "game_id": "game_1746342000_low",
    "siswa_id": "s1",
    "level": "low",
    "selesai_at": "2026-05-01T10:00:00.000Z"
  },
  "meta": null,
  "error": null
}
```

---

## 21. EMOSI — Tim 1

> Dipanggil dari `useWebcamEmotion` hook setiap **5 detik** selama siswa aktif di chatbot.

### POST /emosi/deteksi — Tim 1

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "sesi_id": "sesi_s1_20260501_bil_aljabar",
  "frame_base64": "base64_jpeg_string_224x224"
}
```

> `frame_base64`: JPEG base64 **tanpa** prefix `"data:image/jpeg;base64,"`.

**Response 200:**
```json
{
  "data": {
    "emosi": "antusias",
    "confidence": 0.89,
    "terdeteksi_at": "2026-05-01T09:10:00.000Z"
  },
  "meta": null,
  "error": null
}
```

> `emosi`: `"antusias"` | `"bosan"` | `"bingung"` | `"frustrasi"` | `"tidak_terdeteksi"`

**Error 400:** `{ "data": null, "meta": null, "error": { "code": "VALIDATION_ERROR", "message": "Frame tidak valid.", "details": { "emosi": "tidak_terdeteksi" } } }`

---

## 22. MENTOR — Tim 5

> **Tanggung jawab Tim 5:**
> - Interaksi chatbot selama sesi belajar
> - Feedback evaluasi quiz via CTA badge "📊 Evaluasi Kuis" (endpoint terpisah)
> - CTA di panel quiz riwayat berlabel 'Tanya Mentor AI'. Response AI masuk ke chat dengan badge '📊 Evaluasi Kuis'.

### POST /mentor/pesan
Kirim pesan ke mentor, tunggu full response. Fallback jika SSE tidak tersedia.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "sesi_id": "sesi_s1_20260501_bil_aljabar",
  "level": "Mid",
  "pesan": "Aku bingung cara menyelesaikan 2x + 3 = 7",
  "konteks": {
    "emosi": "bingung",
    "publish_id": "pub_mat_bil_aljabar_x1_20260501",
    "bacaan": "# Persamaan Linear\n\n## A. Pengertian... (maks 3000 karakter)"
  }
}
```

> - **V3.8:** `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id`, `atp` dihapus dari request — BE lookup dari `sesi_id` → `publish_id`.
> - `level` **tetap dikirim eksplisit** — level siswa bisa berubah mid-session (naik level setelah agregasi quiz ≥ KKM). Tim 5 butuh nilai level aktif saat ini.
> - `konteks.emosi`: real-time dari deteksi kamera — tetap dikirim FE.
> - `konteks.bacaan`: konten yang sedang dirender siswa saat ini — opsional, `null` jika belum tersedia. Maks 3000 karakter. Tim 5 gunakan sebagai referensi utama. Tetap dikirim FE karena BE tidak menyimpan state render.

**Response 200:**
```json
{
  "data": {
    "balasan": "Tenang ya, kita mulai dari yang paling dasar. Persamaan 2x + 3 = 7 artinya...",
    "sesi_id": "sesi_s1_20260501_bil_aljabar"
  },
  "meta": null,
  "error": null
}
```

---

### POST /mentor/pesan/stream
Identik dengan `/mentor/pesan` tapi response via **SSE**.

**Request Body:** sama persis dengan `/mentor/pesan` (**V3.8**: tanpa `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id`, `atp`).

**Response:** `Content-Type: text/event-stream`

```
data: Tenang \n\n
data: ya, \n\n
data: [DONE]\n\n
```

> FE menggunakan `EventSource` atau `fetch` dengan `ReadableStream`. Ketika `data: [DONE]` diterima, FE tutup koneksi.

---

### POST /mentor/evaluasi — Tim 5
Evaluasi hasil quiz siswa via CTA "Tanya Mentor AI". System prompt Tim 5 terpisah dari chat normal — fokus pada analisis jawaban.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "hasil_quiz_id": "hq_20260501_0001"
}
```

> - `hasil_quiz_id`: **wajib** — BE Tim 6 lookup dan inject seluruh data quiz ke Tim 5: soal, jawaban siswa, kunci jawaban (MC) / rubrik (essay), `penjelasan` per soal, nilai per soal, nilai total.
> - **V3.8:** `sesi_id`, `mapel_id`, `elemen_id`, `elemen_label`, `materi`, `materi_id`, `level`, `atp` dihapus dari request — BE lookup semua dari `hasil_quiz_id`.
> - Tidak ada field `pesan` — tidak ada input teks dari siswa di flow ini.

**Response 200:**
```json
{
  "data": {
    "balasan": "Kamu sudah mengerjakan quiz dengan baik! Skor kamu 80/100. Ada 2 soal yang perlu diperhatikan...",
    "sesi_id": "sesi_s1_20260501_bil_aljabar"
  },
  "meta": null,
  "error": null
}
```

---

### POST /mentor/evaluasi/stream — Tim 5
Identik dengan `/mentor/evaluasi` tapi response via **SSE**.

**Request Body:** sama persis dengan `/mentor/evaluasi` (**V3.8**: hanya `siswa_id` dan `hasil_quiz_id`).

**Response:** `Content-Type: text/event-stream` — identik dengan `/mentor/pesan/stream`.

---

## 23. LEADERBOARD — Tim 6 BE

### GET /leaderboard
Ranking siswa per kelas berdasarkan akumulasi nilai quiz.

**Auth:** role `siswa`

**Query Params:**
- `kelas_id` (wajib)
- `mode`: `"daily"` | `"monthly"` (default: `"monthly"`)
  - `"daily"`: poin hari ini, reset tengah malam WIB
  - `"monthly"`: poin bulan berjalan, reset tanggal 1

**Formula poin:** `Σ (nilai_mc × 60% + nilai_essay × 40%)` semua sesi dalam periode.

**Response 200:**
```json
{
  "data": [
    {
      "peringkat": 1,
      "siswa_id": "s1",
      "nama": "Budi Santoso",
      "avatar": "https://cdn.sekolahrakyat.id/avatars/s1.jpg",
      "kelas_id": "x1",
      "total_poin": 420,
      "streak_hari": 5
    }
  ],
  "meta": {
    "mode": "monthly",
    "periode": "2026-05",
    "kelas_id": "x1",
    "diperbarui_at": "2026-05-01T12:00:00.000Z"
  },
  "error": null
}
```

---

## 24. NOTIFIKASI — Tim 6 BE

> Notifikasi satu arah dari guru ke siswa.

### POST /notifikasi
Guru kirim pesan/rekomendasi ke siswa.

**Auth:** role `guru`

**Request:**
```json
{
  "guru_id": "g1",
  "siswa_id": "s1",
  "mapel_id": "mat",
  "pesan": "Coba ulangi materi persamaan linear, fokus pada soal dua variabel."
}
```

**Response 201:**
```json
{
  "data": { "id": "notif_123", "dibuat_at": "2026-05-01T09:00:00.000Z" },
  "meta": null,
  "error": null
}
```

---

### PATCH /notifikasi/:id/baca
Tandai notifikasi sudah dibaca.

**Auth:** role `siswa`

**Response 200:**
```json
{ "data": { "dibaca": true }, "meta": null, "error": null }
```

---

## 25. WEBSOCKET SPEC — Tim 6 BE

### 25.1 Koneksi

**23.1.1 WebSocket Guru — Monitoring Real-Time**

**URL:**
```
wss://api.sekolahrakyat.id/v1/ws/monitoring
```

**Query Params:**
```
?kelas_id={kelas_id}&mapel_id={mapel_id}&token={access_token}
```

> `token` dikirim sebagai query param (bukan header) karena keterbatasan browser WebSocket API.
> `mapel_id` wajib jika guru mengampu lebih dari 1 mapel di kelas tersebut.

**23.1.2 WebSocket Siswa — Notifikasi Async (essay dinilai & naik level)**

**URL:**
```
wss://api.sekolahrakyat.id/v1/ws/siswa
```

**Query Params:**
```
?siswa_id={siswa_id}&sesi_id={sesi_id}&token={access_token}
```

> Digunakan siswa untuk menerima notifikasi async dari BE — khususnya event `essay_dinilai`.
> FE siswa connect ke endpoint ini **setelah** `POST /sesi` berhasil dan chatbot terbuka.
> `sesi_id` wajib — server hanya push event yang relevan dengan sesi aktif siswa tersebut.

Setelah koneksi berhasil, server mengirim event `connected`:
```json
{
  "type": "connected",
  "payload": { "siswa_id": "s1", "sesi_id": "sesi_s1_20260501_bil_aljabar" },
  "timestamp": "2026-05-01T09:00:00.000Z"
}
```

Reconnect & fallback siswa: exponential backoff, refresh token jika expired. Jika WS tidak tersedia, FE siswa **poll** `GET /siswa/:id/quiz?elemen_id=` setiap 10 detik.

**Env:**
```
VITE_WS_URL=wss://api.sekolahrakyat.id/v1/ws
```

---

### 25.2 Handshake (Guru)

Setelah koneksi berhasil, server mengirim event `connected`:
```json
{
  "type": "connected",
  "payload": {
    "kelas_id": "x1",
    "mapel_id": "mat",
    "siswa_online": ["s1", "s3", "s5"]
  },
  "timestamp": "2026-05-01T09:00:00.000Z"
}
```

---

### 25.3 Event Types (Server → Client)

Semua event menggunakan envelope:
```json
{
  "type": "<event_type>",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {},
  "timestamp": "2026-05-01T09:15:30.000Z"
}
```

**`siswa_aktif`** — Siswa mulai belajar:
```json
{
  "type": "siswa_aktif",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "mapel_id": "mat",
    "elemen_id": "bil_aljabar",
    "materi_id": "mat__persamaan_linear",
    "sesi_id": "sesi_s1_20260501_bil_aljabar"
  },
  "timestamp": "2026-05-01T09:00:00.000Z"
}
```

**`siswa_nonaktif`** — Siswa menutup chatbot:
```json
{
  "type": "siswa_nonaktif",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": { "sesi_id": "sesi_s1_20260501_bil_aljabar", "durasi_menit": 45 },
  "timestamp": "2026-05-01T09:45:00.000Z"
}
```

**`progress_siswa`** — Update progress belajar:
```json
{
  "type": "progress_siswa",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "mapel_id": "mat",
    "elemen_id": "bil_aljabar",
    "materi_id": "mat__persamaan_linear",
    "level": "mid",
    "progress_pct": 65
  },
  "timestamp": "2026-05-01T09:15:30.000Z"
}
```

> `progress_pct` = progress per mapel siswa: `round(selesai / total_elemen × 100)`.

**`quiz_siswa`** — Siswa submit quiz:
```json
{
  "type": "quiz_siswa",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "mapel_id": "mat",
    "elemen_id": "bil_aljabar",
    "materi_id": "mat__persamaan_linear",
    "tipe": "mc",
    "nilai": 80,
    "level": "low",
    "naik_level": false
  },
  "timestamp": "2026-05-01T09:15:30.000Z"
}
```

**`emosi_siswa`** — Deteksi emosi berubah:
```json
{
  "type": "emosi_siswa",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "emosi": "bingung",
    "confidence": 0.84,
    "durasi_emosi_negatif_menit": 5
  },
  "timestamp": "2026-05-01T09:15:30.000Z"
}
```

**`pelanggaran_siswa`** — Siswa terdeteksi pelanggaran:
```json
{
  "type": "pelanggaran_siswa",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": { "detail": "Berpindah Tab / Menyembunyikan Halaman" },
  "timestamp": "2026-05-01T09:15:30.000Z"
}
```

**`essay_dinilai`** — Tim 3 selesai menilai essay, agregasi sudah dihitung:
```json
{
  "type": "essay_dinilai",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "elemen_id": "bil_aljabar",
    "materi_id": "mat__persamaan_linear",
    "level": "low",
    "nilai_essay": 78,
    "nilai_mc": 80,
    "agregasi": 79.2,
    "naik_level": true,
    "kkm": 75
  },
  "timestamp": "2026-05-01T09:25:00.000Z"
}
```

> Event ini dikirim ke **dua channel sekaligus:**
> - `wss://.../ws/siswa` → FE siswa update UI naik level di chatbot tanpa polling.
> - `wss://.../ws/monitoring` → FE guru update tabel aktivitas siswa di halaman monitoring.

**`smart_alert`** — Alert otomatis untuk guru:
```json
{
  "type": "smart_alert",
  "siswa": { "id": "s1", "nama": "Budi", "avatar": null },
  "payload": {
    "jenis": "emosi_negatif_berkepanjangan",
    "detail": "Budi terdeteksi bingung/frustrasi selama >15 menit berturut-turut.",
    "durasi_menit": 17
  },
  "timestamp": "2026-05-01T09:30:00.000Z"
}
```

> **Kondisi smart_alert:** emosi negatif (bosan/bingung/frustrasi) > 15 menit berturut-turut; atau siswa melakukan pelanggaran.

---

### 25.4 Event Types (Client → Server)

**`ping`** — Keepalive dari FE:
```json
{ "type": "ping" }
```

Server merespons dengan `pong`:
```json
{ "type": "pong", "timestamp": "2026-05-01T09:15:30.000Z" }
```

---

### 25.5 Reconnect & Fallback

| Kondisi | Behavior |
|---------|----------|
| Koneksi terputus | FE retry dengan **exponential backoff**: 1s → 2s → 4s → 8s (maks 30s) |
| Token expired | FE refresh token via `POST /auth/refresh`, lalu reconnect |
| Server unreachable setelah 3 retry | FE fallback ke polling `GET /kelas/:id/progress` setiap 30 detik |
| `kelas_id` tidak valid | Server kirim event `error` dan tutup koneksi |

**Error event dari server:**
```json
{
  "type": "error",
  "payload": {
    "code": "INVALID_KELAS",
    "message": "kelas_id tidak valid atau guru tidak mengampu kelas ini."
  }
}
```

---

## 26. HIRARKI KURIKULUM (ATURAN GLOBAL)

```
Kurikulum Merdeka
  └── Mapel       (Matematika)              ← fase & deskripsi_cp ada di sini
        └── Elemen   (Bilangan dan Aljabar)  ← elemen_id SELALU WAJIB
              └── Materi  (Persamaan Linear) ← opsional, diisi guru/siswa
```

### 26.1 Field Wajib di Semua Payload Content / Game / Mentor

| Field | Keterangan |
|-------|-----------|
| `mapel_id` | Selalu wajib |
| `elemen_id` | **Selalu wajib**, tidak boleh null atau omit |
| `elemen_label` | Wajib di semua payload mutasi (POST/PUT) — untuk konteks LLM & display |

### 26.2 Field Opsional

| Field | Keterangan |
|-------|-----------|
| `materi` | Nama materi (string label), hanya diisi jika guru/siswa turun ke level materi |
| `materi_id` | Format: `"{mapel_id}__{snake_case}"` — contoh: `"mat__persamaan_linear"` |
| `atp` | Alur Tujuan Pembelajaran — opsional tapi direkomendasikan untuk generate konten |

### 26.3 Aturan Validasi (Wajib Semua Tim)

- BE / Tim 3 / Tim 4 / Tim 5 **wajib menolak** payload yang punya `mapel_id` tapi **tidak punya `elemen_id`**
- FE membangun `materi_id`: `` materi_id = `${mapel_id}__${materi.toLowerCase().replace(/\s+/g, '_')}` ``
- `materi_id` hanya dikirim jika `materi` juga dikirim (keduanya sinkron)

---

## 27. STANDARD RESPONSE & ERROR

### 27.1 Response Envelope (Wajib Semua Endpoint)

```json
{
  "data": "<object|array|null>",
  "meta": "<object|null>",
  "error": "<object|null>"
}
```

### 27.2 HTTP Status Code Reference

| Code | Penggunaan |
|------|-----------|
| 200 | OK — request berhasil |
| 201 | Created — resource baru berhasil dibuat |
| 400 | Bad Request — validasi gagal, field tidak valid |
| 401 | Unauthorized — token tidak valid atau expired |
| 403 | Forbidden — role tidak diizinkan mengakses resource ini |
| 404 | Not Found — resource tidak ditemukan |
| 409 | Conflict — duplikat data (email, NIP, NIS, dll) |
| 422 | Unprocessable Entity — data valid secara format tapi tidak valid secara logika bisnis |
| 429 | Too Many Requests — rate limit (khususnya endpoint LLM/RAG) |
| 500 | Internal Server Error — kesalahan server tidak terduga |

### 27.3 Caching Strategy (Rekomendasi)

| Endpoint | Cache Strategy | TTL |
|----------|---------------|-----|
| `GET /admin/mapel` | Server-side cache | 1 jam |
| `GET /admin/mapel/:mapel_id/elemen` | Server-side cache | 1 jam |
| `GET /leaderboard?mode=daily` | Server-side cache | 5 menit |
| `GET /leaderboard?mode=monthly` | Server-side cache | 15 menit |
| `GET /siswa/:id/kpi` | No cache (real-time) | — |
| `POST /rag/*` | No cache (LLM call) | — |
| `GET /siswa/:id/konten` | Client-side cache | Sampai invalidasi |

---

## 28. BOOKS — Tim 6 BE + Tim 3 (Ingestion)

> Fitur upload buku teks PDF oleh guru sebagai knowledge source untuk pipeline RAG Tim 3.
> Tim 3 hanya memproses halaman dalam `included_ranges` — halaman di luar range dianggap noise.
> **Auth:** semua endpoint section ini hanya untuk role `guru`.

### Mekanisme Internal BE ↔ Tim 3

**Alur POST /books:**
1. BE Tim 6 terima file multipart, validasi auth + ownership kelas
2. BE simpan metadata buku ke database dengan `status: "processing"`
3. BE kirim file + metadata ke Tim 3 (mekanisme: TBD — push callback / internal queue)
4. Tim 3 proses hanya halaman dalam range `page_start`–`page_end`
5. Tim 3 update status ke BE via callback: `status: "indexed"` atau `"failed"`
6. BE update field `status` dan `indexed_at` di database

**Alur DELETE /books/:id:**
1. BE Tim 6 hapus metadata dari database
2. BE notify Tim 3 untuk hapus semua vector yang terkait `book_id` dari VectorDB

### POST /books — Tim 6 BE
Upload PDF buku baru. Multipart form data. Proses ingestion bersifat async.

**Auth:** role `guru`
**Content-Type:** `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Keterangan |
|-------|------|----------|------------|
| `file` | File (PDF) | yes | File PDF buku, maks 50 MB |
| `tingkat` | string | yes | `"X"` \| `"XI"` \| `"XII"` |
| `kelas_id` | string | yes | ID kelas yang diampu guru |
| `mapel_id` | string | yes | ID mata pelajaran |
| `page_start` | number | yes | Halaman awal materi (min: 1) |
| `page_end`   | number | yes | Halaman akhir materi (harus > page_start) |

> `guru_id` tidak dikirim FE — BE resolve dari JWT token.
> `kelas_id` menerima nilai `"semua"` jika guru ingin buku digunakan di semua kelas
> yang diampu untuk tingkat tersebut. BE resolve daftar kelas dari JWT + `tingkat`.
> `kelas_nama` di response berisi `"Semua Kelas {tingkat}"` untuk kasus ini.
> BE wajib mengkonversi `page_start` + `page_end` menjadi
> `included_ranges: [{ start, end }]` sebelum menyimpan ke database dan mengembalikan di response.

**Response 201:**
```json
{
  "data": {
    "book_id": "book_001",
    "nama_file": "Matematika_X_Kelas_X.pdf",
    "tingkat": "X",
    "kelas_id": "x1",
    "kelas_nama": "X-1",
    "mapel_id": "mat",
    "mapel_label": "Matematika",
    "mapel_icon": "📐",
    "included_ranges": [{ "start": 5, "end": 198 }],
    "status": "processing",
    "uploaded_at": "2026-05-27T09:00:00.000Z",
    "guru_id": "g1"
  },
  "meta": null,
  "error": null
}
```

**Error 400:** "Format file tidak didukung. Hanya PDF yang diizinkan."
**Error 400:** "Ukuran file melebihi batas maksimal."
**Error 400:** "Halaman akhir harus lebih besar dari halaman awal."
**Error 403:** "Guru tidak mengampu kelas ini."

---

### GET /books — Tim 6 BE
Daftar semua buku yang pernah diupload guru.

**Auth:** role `guru`

**Query Params (opsional):** `guru_id`, `mapel_id`, `kelas_id`, `tingkat`, `status`

**Response 200:**
```json
{
  "data": [
    {
      "book_id": "book_001",
      "nama_file": "Matematika_X_Kelas_X.pdf",
      "tingkat": "X",
      "kelas_id": "x1",
      "kelas_nama": "X-1",
      "mapel_id": "mat",
      "mapel_label": "Matematika",
      "mapel_icon": "📐",
      "included_ranges": [{ "start": 5, "end": 198 }],
      "status": "processing",
      "uploaded_at": "2026-05-27T09:00:00.000Z",
      "indexed_at": null,
      "guru_id": "g1"
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 3, "total_pages": 1 },
  "error": null
}
```

> `status`: `"processing"` | `"indexed"` | `"failed"`

---

### GET /books/:id/status — Tim 6 BE
Polling status ingestion satu buku. Dipakai FE setelah upload sukses.

**Auth:** role `guru`

**Response 200:**
```json
{
  "data": {
    "book_id": "book_001",
    "status": "indexed",
    "indexed_at": "2026-05-27T09:05:00.000Z",
    "message": null
  },
  "meta": null,
  "error": null
}
```

> `message`: diisi jika `status: "failed"` — berisi deskripsi error dari Tim 3.

**Error 404:** "Buku tidak ditemukan."
**Error 403:** "Guru tidak memiliki akses ke buku ini."

---

### DELETE /books/:id — Tim 6 BE
Hapus buku dari sistem. Tim 3 wajib membersihkan vector DB untuk book_id ini.

**Auth:** role `guru`

**Response 200:**
```json
{ "data": { "deleted": true }, "meta": null, "error": null }
```

**Error 404:** "Buku tidak ditemukan."
**Error 403:** "Guru tidak memiliki akses ke buku ini."

---

## LAMPIRAN: QUICK REFERENCE ENDPOINT

### Auth
| Method | Path | Role | Keterangan |
|--------|------|------|-----------|
| POST | `/auth/login` | PUBLIC | Login |
| POST | `/auth/refresh` | PUBLIC | Refresh token |
| POST | `/auth/logout` | semua | Logout |
| POST | `/auth/aktivasi` | siswa | Aktivasi akun + pilih mapel |
| PATCH | `/auth/password` | semua | Ganti password |
| POST | `/auth/lupa-password` | PUBLIC | Kirim link reset |
| GET | `/auth/me` | semua | Profil sesi aktif |
| PUT | `/auth/avatar` | semua | Upload avatar |

### Admin
| Method | Path | Keterangan |
|--------|------|-----------|
| GET/POST | `/admin/mapel` | List & buat mapel |
| GET/PATCH/DELETE | `/admin/mapel/:id` | Detail, update, hapus |
| GET/POST | `/admin/mapel/:mapel_id/elemen` | List & buat elemen |
| GET/PATCH/DELETE | `/admin/mapel/:mapel_id/elemen/:id` | Detail, update, hapus |
| GET/POST | `/admin/kelas` | List & buat kelas |
| GET/PATCH/DELETE | `/admin/kelas/:id` | Detail, update, hapus |
| GET | `/admin/kelas/:id/siswa` | Siswa dalam kelas |
| POST | `/admin/kelas/:id/mapel` | Tambah mapel ke kelas |
| PATCH/DELETE | `/admin/kelas/:id/mapel/:mapel_id` | Update/hapus mapel dari kelas |
| POST | `/admin/kelas/:id/siswa` | Tambah siswa ke kelas |
| DELETE | `/admin/kelas/:id/siswa/:siswa_id` | Lepas siswa dari kelas |
| GET/POST | `/admin/guru` | List & buat guru |
| GET/PATCH/DELETE | `/admin/guru/:id` | Detail, update, hapus |
| POST | `/admin/guru/bulk` | Upload massal guru |
| GET/POST | `/admin/siswa` | List & buat siswa |
| GET/PATCH/DELETE | `/admin/siswa/:id` | Detail, update, hapus |
| POST | `/admin/siswa/bulk` | Upload massal siswa |

### Guru
| Method | Path | Keterangan |
|--------|------|-----------|
| GET | `/guru/:id` | Profil guru |
| GET | `/guru/:id/konten` | Riwayat konten guru |
| GET | `/kelas/:id/progress` | Progress semua siswa di kelas (initial load monitoring) |

### Siswa
| Method | Path | Keterangan |
|--------|------|-----------|
| GET | `/siswa/:id` | Profil siswa |
| GET | `/siswa/:id/kpi` | KPI dashboard |
| GET | `/siswa/:id/progress` | Progress belajar |
| GET | `/siswa/:id/konten` | Konten tersedia |
| GET | `/siswa/:id/pretest/status` | Status pretest |
| GET | `/siswa/:id/quiz` | Riwayat quiz |
| POST | `/siswa/:id/quiz/mc` | Submit quiz MC |
| POST | `/siswa/:id/quiz/essay` | Submit quiz Essay (async, Tim 3 nilai) |
| GET | `/siswa/:id/notifikasi` | Notifikasi dari guru |

### Konten (Guru)
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/konten/generate` | Tim 3 RAG | Generate satu tipe konten |
| POST | `/konten/publish` | Tim 6 BE | Publish paket konten |

### Sesi
| Method | Path | Keterangan |
|--------|------|-----------|
| POST | `/sesi` | Mulai sesi belajar |
| PATCH | `/sesi/:id` | Update/tutup sesi |
| POST | `/sesi/:id/summary` | Generate evaluasi AI (Tim 3) |
| GET | `/sesi/:id/emosi` | Log emosi sesi |
| GET | `/sesi/:id/chat` | Riwayat chat sesi |

### Pretest
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/pretest/soal` | Tim 6 BE | Ambil soal pretest |
| POST | `/pretest/submit` | Tim 6 BE | Submit jawaban, terima level via algoritma Lowest Failed Level |

### RAG
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/rag/rekomendasi` | Tim 3 | Rekomendasi topik |
| POST | `/rag/insight` | Tim 3 | Insight personal dashboard |

### Game
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/game/generate` | Tim 4 | Generate game baru |
| POST | `/game/regenerate` | Tim 4 | Regenerate game via `game_id` |
| GET | `/game/:id` | Tim 4 | Detail game + polling |
| PATCH | `/game/:id/penyelesaian` | Tim 6 BE | Catat penyelesaian game |

### Emosi
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/emosi/deteksi` | Tim 1 | Deteksi emosi dari frame |

### Mentor
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/mentor/pesan` | Tim 5 | Chat mentor normal (non-streaming) |
| POST | `/mentor/pesan/stream` | Tim 5 | Chat mentor normal (SSE streaming) |
| POST | `/mentor/evaluasi` | Tim 5 | Evaluasi quiz CTA (non-streaming) |
| POST | `/mentor/evaluasi/stream` | Tim 5 | Evaluasi quiz CTA (SSE streaming) |

### Leaderboard & Notifikasi
| Method | Path | Keterangan |
|--------|------|-----------|
| GET | `/leaderboard` | Ranking kelas |
| POST | `/notifikasi` | Guru kirim notifikasi |
| PATCH | `/notifikasi/:id/baca` | Tandai notifikasi dibaca |

### WebSocket
| URL | Role | Keterangan |
|-----|------|-----------|
| `wss://api.sekolahrakyat.id/v1/ws/monitoring?kelas_id=&mapel_id=&token=` | guru | Real-time monitoring guru |
| `wss://api.sekolahrakyat.id/v1/ws/siswa?siswa_id=&sesi_id=&token=` | siswa | Notifikasi async siswa (essay dinilai, naik level) |

### Books
| Method | Path | Keterangan |
|--------|------|-----------|
| POST | `/books` | Upload PDF buku baru (multipart) |
| GET | `/books` | Daftar buku guru |
| GET | `/books/:id/status` | Polling status ingestion |
| DELETE | `/books/:id` | Hapus buku |

---

*— End of API Contract SR MVP V3.8 —*
