# API CONTRACT — SEKOLAH RAKYAT MVP
## Versi 3.6 — Production-Ready | Single Source of Truth

> **Status:** FINAL — Acuan wajib untuk Tim 6 BE, Tim 3 RAG, Tim 4 Game, Tim 5 Mentor, Tim 1 Emosi, Tim 6 FE  
> **Tanggal:** 2026-05-12  
> **Basis:** V3.5 + V3.6 (Fix audit: GET /kelas/:id/progress sebagai section resmi, hasil_quiz_id di riwayat quiz, atomik note publish, WebSocket siswa endpoint, timing trigger pretest)

---

## DAFTAR ISI

1. [Konvensi Global](#1-konvensi-global)
2. [Changelog V3.1 — Addendum Sesi & Konteks Mentor](#2-changelog-v31-addendum-sesi--konteks-mentor)
3. [Changelog V3.3 — Pemisahan Endpoint Quiz, Konten, Game, Mentor](#3-changelog-v33-pemisahan-endpoint-quiz-konten-game-mentor)
4. [Changelog V3.4 — Penyesuaian Struktur Konten](#4-changelog-v34-penyesuaian-struktur-konten)
5. [Changelog V3.5 — Fix Audit Contract](#5-changelog-v35-fix-audit-contract)
6. [Changelog V3.6 — Fix Audit Contract (Lanjutan)](#6-changelog-v36-fix-audit-contract-lanjutan)
7. [Peta Domain Endpoint](#7-peta-domain-endpoint)
8. [AUTH — Tim 6 BE](#8-auth--tim-6-be)
9. [ADMIN — Tim 6 BE](#9-admin--tim-6-be)
10. [GURU — Tim 6 BE](#10-guru--tim-6-be)
11. [SISWA — Tim 6 BE](#11-siswa--tim-6-be)
12. [KONTEN — Tim 3 RAG + Tim 6 BE](#12-konten--tim-3-rag--tim-6-be)
13. [SESI — Tim 6 BE](#13-sesi--tim-6-be)
14. [PRETEST — Tim 3 RAG (generate) + Tim 6 BE (serve)](#14-pretest--tim-3-rag-generate--tim-6-be-serve)
15. [QUIZ — Tim 6 BE](#15-quiz--tim-6-be)
16. [RAG — Tim 3](#16-rag--tim-3)
17. [GAME — Tim 4](#17-game--tim-4)
18. [EMOSI — Tim 1](#18-emosi--tim-1)
19. [MENTOR — Tim 5](#19-mentor--tim-5)
20. [LEADERBOARD — Tim 6 BE](#20-leaderboard--tim-6-be)
21. [NOTIFIKASI — Tim 6 BE](#21-notifikasi--tim-6-be)
22. [WebSocket Spec — Tim 6 BE](#22-websocket-spec--tim-6-be)
23. [Hirarki Kurikulum (Aturan Global)](#23-hirarki-kurikulum-aturan-global)
24. [Standard Response & Error](#24-standard-response--error)

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
| `POST /konten/regenerate` | **Baru** — iterative refinement per konten per level menggunakan `konten_id` |
| `POST /game/regenerate` | **Baru** — iterative refinement per game menggunakan `game_id` |
| `POST /mentor/evaluasi` | **Baru** — evaluasi quiz CTA, system prompt Tim 5 terpisah dari chat normal |
| `POST /mentor/evaluasi/stream` | **Baru** — versi SSE dari evaluasi |
| WebSocket | **Tambah** event `essay_dinilai` — push agregasi setelah Tim 3 selesai nilai essay |

### 3.1 Masalah di V2 yang Diperbaiki

**A. Endpoint Tidak Konsisten / Action-Based (diperbaiki ke resource-based):**
- `POST /summary/siswa/:id` → `POST /sesi/:id/summary`
- `GET /content/siswa` → `GET /siswa/:id/konten`
- `GET /content/progress/siswa` → `GET /siswa/:id/progress`
- `GET /content/progress/guru` → `GET /kelas/:id/progress`
- `POST /game/selesai` → `PATCH /game/:id/penyelesaian`

**B. Endpoint Overloaded (dipecah):**
- `GET /content/progress/siswa` → `GET /siswa/:id/kpi` + `GET /siswa/:id/progress`

**C. Redundansi & Overlap:**
- `GET /guru/rekomendasi` dan `POST /guru/rekomendasi` → domain `/notifikasi`
- `GET /emotion/history` → `GET /sesi/:id/emosi`
- `GET /mentor/chat/history` → `GET /sesi/:id/chat`

**D. FE Mengirim Data yang Seharusnya Dihitung BE:**
- `POST /content/quiz/submit` meminta FE mengirim `score` → BE hitung sendiri
- `POST /content/insight` meminta FE mengirim KPI → BE ambil dari database

### 3.2 Ringkasan Perubahan URL

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
| 5 | `POST /konten/generate`, `POST /konten/publish`, semua endpoint terkait pretest | **Pretest di-generate bersamaan konten** oleh Tim 3 RAG secara internal; tidak dikembalikan ke FE/guru |
| 6 | `PATCH /game/:id/penyelesaian` | **Trigger selesai via `postMessage`** — game HTML kirim `{ type: 'game:selesai' }` ke parent FE |

> **Catatan kompatibilitas:** FE yang masih membaca `html_url` dari response game akan gagal. Adapter `studentContent.js` dan `game.js` **wajib diperbarui** untuk menggunakan `html_string`.

---

## 5. CHANGELOG V3.5 (Fix Audit Contract)

| # | Titik | Perubahan |
|---|-------|-----------|
| 1 | Konvensi 1.6 | **Perjelas aturan level casing** — semua response BE wajib lowercase; ditegaskan dengan catatan wajib |
| 2 | Semua response endpoint | **Fix level casing** — ganti semua `"Low"/"Mid"/"High"` di response menjadi `"low"/"mid"/"high"` |
| 3 | `GET /siswa/:id/progress` | **Fix trailing comma JSON** di `nilai_agregasi_terakhir` dalam object materi |
| 4 | `GET /admin/mapel/:id`, `GET /admin/mapel/:mapel_id/elemen/:id` | **Fix envelope** — tambahkan standard response envelope yang sebelumnya hilang |
| 5 | Struktur dokumen | **Fix penomoran section** — section ADMIN dari 5.x → 6.x, WebSocket dari 18.x → 21.x, Kurikulum dari 19.x → 22.x, Standard Response dari 20.x → 23.x |
| 6 | `POST /rag/insight` | **Fix kontradiksi request body** — hapus catatan menyesatkan; FE tetap kirim KPI sesuai body yang sudah ada |
| 7 | `POST /konten/publish` | **Fix error 409** — hapus kalimat "Gunakan endpoint update" karena endpoint tersebut tidak ada |
| 8 | `POST /konten/publish` — item game | **Fix konten_id game** — hapus klaim "keduanya wajib"; item game cukup gunakan `game_id` tanpa `konten_id` |
| 9 | Quick reference | **Fix path elemen** — `/admin/elemen` → `/admin/mapel/:mapel_id/elemen` |
| 10 | Quick reference | **Fix notifikasi** — pisahkan GET dan PATCH ke path yang benar |
| 11 | Quick reference | **Fix POST kelas mapel** — `POST /admin/kelas/:id/mapel/:mapel_id` → `POST /admin/kelas/:id/mapel` |
| 12 | Section 15 QUIZ | **Update referensi** — ganti `POST /siswa/:id/quiz` (deprecated) ke endpoint baru `/mc` dan `/essay` |
| 13 | Caching strategy | **Fix path elemen** — `/admin/elemen` → `/admin/mapel/:mapel_id/elemen` |
| 14 | WebSocket semua event | **Fix timestamp** — seragamkan semua WS timestamp ke ISO 8601 penuh |
| 15 | Section pretest | **Fix ownership** — perjelas Tim 3 RAG generate, Tim 6 BE serve |
| 16 | Footer | **Fix versi** — "End of API Contract SR MVP V3.1" → V3.5 |
| 17 | `POST /sesi` — catatan | **Hapus referensi V3.2** yang tidak ada; ganti dengan keterangan yang benar |
| 18 | Section Context Injection Mentor | **Hapus referensi V3.2**; perbarui narasi sesuai kondisi aktual (hasil_quiz_id hanya untuk /mentor/evaluasi) |

---

## 6. CHANGELOG V3.6 (Fix Audit Contract — Lanjutan)

Perubahan V3.6 bersifat **additive dan backward-compatible**.

| # | Titik | Perubahan |
|---|-------|-----------|
| 1 | `GET /siswa/:id/quiz` — response | **Tambah field `hasil_quiz_id`** di setiap item `riwayat[]` — dipakai FE trigger CTA "Tanya Kak Nusa" dari riwayat lama |
| 2 | `POST /konten/publish` — deskripsi | **Tambah catatan atomik** — operasi ini atomik, FE aman retry jika gagal |
| 3 | `POST /konten/generate` — catatan Pretest | **Perjelas timing trigger pretest** — generate pretest di-trigger pada panggilan `POST /konten/generate` pertama dari 13 panggilan paralel |
| 4 | Section 10 GURU | **Tambah `GET /kelas/:id/progress` sebagai section resmi** — endpoint kritis untuk initial load monitoring guru; sebelumnya hanya ada di Quick Reference |
| 5 | Section 22 WebSocket | **Tambah `22.1.2 WebSocket Siswa`** — endpoint `wss://.../ws/siswa` tersendiri untuk siswa; clarify bahwa event `essay_dinilai` dikirim ke dua channel (siswa + guru) |
| 6 | Daftar Isi | **Renumbering** — tambah entry Changelog V3.6 (nomor 6), section domain mulai dari 7 |
| 7 | Quick Reference WebSocket | **Tambah baris WS siswa** |
| 8 | Footer | **Fix versi** — V3.5 → V3.6 |

---

## 7. PETA DOMAIN ENDPOINT

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

## 8. AUTH — Tim 6 BE

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

## 9. ADMIN — Tim 6 BE

> Semua endpoint section ini hanya untuk role **`admin`**. Response `403` jika role lain mengakses.

---

### 8.1 Mapel (Mata Pelajaran)

#### GET /admin/mapel
Daftar semua mapel.

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
Detail satu mapel.

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
  "id": "mat",
  "label": "Matematika",
  "icon": "📐",
  "tingkat": "X",
  "fase": "Fase E (Kelas X)",
  "deskripsi_cp": "Pada akhir fase ini..."
}
```
**Response 201:** `data` = `Mapel` yang dibuat.  
**Error 409:** "ID mapel sudah digunakan."

#### PATCH /admin/mapel/:id
`id` tidak bisa diubah.

**Request Body (partial):**
```json
{
  "label": "Matematika",
  "icon": "📐",
  "fase": "Fase E (Kelas X)",
  "deskripsi_cp": "..."
}
```

**Response 200:** `data` = `Mapel` yang diperbarui.

#### DELETE /admin/mapel/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — elemen di bawah mapel ini dihapus cascade.

---

### 8.2 Elemen (Kurikulum)

> Elemen adalah level kurikulum langsung di bawah mapel. Hanya menyimpan `{ id, mapel_id, label }`.

#### GET /admin/mapel/:mapel_id/elemen
Ambil semua elemen untuk satu mapel. `mapel_id` wajib.

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
Detail satu elemen.

**Response 200:**
```json
{
  "data": { "id": "bil_aljabar", "mapel_id": "mat", "label": "Bilangan dan Aljabar" },
  "meta": null,
  "error": null
}
```

#### POST /admin/mapel/:mapel_id/elemen
**Request Body:**
```json
{ "label": "Bilangan dan Aljabar" }
```

**Response 201:**
```json
{
  "data": { "id": "bil_aljabar", "mapel_id": "mat", "label": "Bilangan dan Aljabar" },
  "meta": null,
  "error": null
}
```

**Error 409:** "Label elemen sudah ada di mapel ini."

#### PATCH /admin/mapel/:mapel_id/elemen/:id
`id` dan `mapel_id` tidak bisa diubah.

**Request Body:**
```json
{ "label": "Nama Elemen Baru" }
```

**Response 200:**
```json
{
  "data": { "id": "bil_aljabar", "mapel_id": "mat", "label": "Nama Elemen Baru" },
  "meta": null,
  "error": null
}
```

**Error 409:** "Label elemen sudah ada di mapel ini."

#### DELETE /admin/mapel/:mapel_id/elemen/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — konten guru yang terkait dilepas otomatis.

---

### 8.3 Kelas

#### GET /admin/kelas
Daftar semua kelas.

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
      "mapel_guru_map": {
        "mat": "g1",
        "bio": "g2"
      }
    }
  ],
  "meta": null,
  "error": null
}
```

#### GET /admin/kelas/:id
Detail satu kelas. **Response 200:** `data` = satu objek `Kelas`.

#### GET /admin/kelas/:id/siswa
Daftar siswa dalam kelas.

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
**Request:**
```json
{
  "nama": "X-1",
  "tingkat": "X",
  "tahun_ajaran": "2025/2026",
  "wali_kelas_id": null
}
```
**Response 201:** `data` = `Kelas` yang dibuat.

#### PATCH /admin/kelas/:id
Body: field yang diubah saja (`nama`, `wali_kelas_id`, `tahun_ajaran`).  
**Response 200:** `data` = `Kelas` yang diperbarui.

#### DELETE /admin/kelas/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — siswa dilepas dari kelas (`kelas_id` → null), tidak dihapus.

---

#### POST /admin/kelas/:id/mapel
Tambah mapel ke kelas + assign guru pengampu.

**Request Body:**
```json
{ "mapel_id": "mat", "guru_id": "g1" }
```

**Response 201:** `data` = `Kelas` yang diperbarui.  
**Error 409:** "Mapel sudah ada di kelas ini."

#### PATCH /admin/kelas/:id/mapel/:mapel_id
Ganti guru pengampu untuk mapel di kelas ini.  
**Request:** `{ "guru_id": "g2" }`  
**Response 200:** `{ "data": { "mapel_id": "mat", "guru_id": "g2" }, "meta": null, "error": null }`

#### DELETE /admin/kelas/:id/mapel/:mapel_id
Lepas mapel dari kelas.  
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

#### POST /admin/kelas/:id/siswa
Tambah siswa ke kelas. **Request:** `{ "siswa_id": "s5" }`  
**Response 201:** `data` = siswa yang diperbarui.  
**Error 409:** "Siswa sudah ada di kelas ini."

#### DELETE /admin/kelas/:id/siswa/:siswa_id
Lepas siswa dari kelas.  
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

---

### 8.4 Guru

#### GET /admin/guru
Daftar semua guru.

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
      "mapel_kelas_map": {
        "mat": ["x1", "x2"]
      }
    }
  ],
  "meta": { "page": 1, "limit": 20, "total": 5, "total_pages": 1 },
  "error": null
}
```

#### GET /admin/guru/:id
Detail satu guru. **Response 200:** `data` = satu objek `Guru`.

#### POST /admin/guru
**Request:**
```json
{
  "nama": "Pak Budi",
  "nip": "199501152019011001",
  "email": "budi.guru@sekolah.id",
  "mapel_kelas_map": { "bio": ["x1"] }
}
```
**Response 201:** `data` = `Guru` yang dibuat.  
**Error 409:** "Email atau NIP sudah terdaftar."

> BE generate password awal dan kirim via email setelah guru berhasil dibuat.

#### PATCH /admin/guru/:id
`mapel_kelas_map` bersifat **full replace** jika dikirim.  
**Request:** `{ "nama"?, "email"?, "nip"?, "mapel_kelas_map"? }`  
**Response 200:** `data` = `Guru` yang diperbarui.

#### DELETE /admin/guru/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }` — relasi wali kelas dilepas otomatis.

#### POST /admin/guru/bulk
Upload data guru massal.

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

### 8.5 Siswa

#### GET /admin/siswa
Daftar semua siswa.

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
Detail satu siswa. **Response 200:** `data` = satu objek `Siswa`.

#### POST /admin/siswa
**Request:**
```json
{
  "nama": "Citra Dewi",
  "nis": "9876543210",
  "email": "citra@sekolah.id",
  "kelas_id": "x1"
}
```
**Response 201:** `data` = `Siswa` dengan `status: "Belum Aktif"`, `is_first_login: true`.

> BE generate password sementara dan kirim via email.

#### PATCH /admin/siswa/:id
Jika `kelas_id` berubah, relasi kelas lama dilepas otomatis.

**Request Body (partial):**
```json
{
  "nama": "Citra Dewi Updated",
  "kelas_id": "x2"
}
```
**Response 200:** `data` = `Siswa` yang diperbarui.

#### DELETE /admin/siswa/:id
**Response 200:** `{ "data": { "deleted": true }, "meta": null, "error": null }`

#### POST /admin/siswa/bulk
Upload data siswa massal.

**Content-Type:** `multipart/form-data`  
**Form Fields:** `file` (CSV atau XLSX), `kelas_id` (opsional — semua siswa di file masuk kelas ini)

**Format CSV minimal:**
```
nama,nis,email
Budi Santoso,1234567890,budi@sekolah.id
```

**Response 200:** sama dengan `/admin/guru/bulk`.

---

## 10. GURU — Tim 6 BE

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
    "mapel_kelas_map": {
      "mat": ["x1", "x2"]
    },
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
Progress belajar semua siswa dalam satu kelas untuk satu mapel. Digunakan guru sebagai **initial load** sebelum WebSocket aktif di halaman monitoring, dan sebagai **fallback polling** jika WebSocket tidak tersedia.

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
> - `rata_rata_progress`: rata-rata `progress_pct` seluruh siswa di kelas ini untuk mapel yang dipilih.
> - `siswa[].aktif: true` = siswa sedang aktif di chatbot saat ini; `false` = tidak sedang aktif. Siswa yang belum pernah aktif ditandai dengan `nilai_terakhir: null` dan `durasi_menit: 0`.
> - `siswa[].level`: level konten terakhir siswa. `null` jika belum pernah belajar.
> - Response ini adalah **snapshot** — tidak real-time. Untuk data real-time gunakan WebSocket `wss://.../ws/monitoring`.

**Error 400:** `mapel_id` wajib diisi jika guru mengampu lebih dari 1 mapel di kelas ini.  
**Error 403:** Guru tidak mengampu kelas ini.

---

## 11. SISWA — Tim 6 BE

### GET /siswa/:id
Profil siswa. Siswa hanya bisa akses profil sendiri; guru dan admin bisa akses semua.

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
> **Total durasi menit:** total waktu yang dihabiskan siswa untuk belajar.

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
    "sudah_selesai_ids": ["bil_aljabar", "data_statistika"],
    "sedang_dipelajari_ids": ["geometri"]
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
            "text": "# Persamaan Linear\n\n## A. Pengertian\n...",
            "source": [
              { "judul": "Matematika SMA Kelas X", "penulis": "Kemendikbud", "tahun": "2022" }
            ]
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
                "soal": "Berapakah nilai x dari persamaan 2x + 3 = 7?",
                "pilihan": ["1", "2", "3", "4"],
                "jawaban": 1,
                "penjelasan": "2x + 3 = 7 → 2x = 4 → x = 2, maka jawaban yang benar adalah indeks 1 (nilai '2')."
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
                "soal": "Jelaskan langkah-langkah menyelesaikan persamaan linear satu variabel.",
                "rubrik": "Menyebutkan 3+ langkah dengan benar",
                "placeholder": "Tuliskan jawabanmu di sini..."
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
            "source": [
              { "judul": "Matematika SMA Kelas X", "penulis": "Kemendikbud", "tahun": "2022" }
            ]
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

> **Catatan konten_list:** 16 item total per paket:
> `bacaan×3 + quiz_pg×3 + quiz_essay×3 + flashcard×3 + mindmap×1 + game×3`
>
> FE memfilter `konten_list` berdasarkan level siswa saat ini (hasil pretest).  
> Konten `bacaan` dirender dari `content.text` (markdown) di chatbot.  
> Konten `game` di sini **tidak** mengandung `html_string` — FE fetch `html_string` via `GET /game/:id` saat siswa klik "Main Game".  
> Field `game_selesai: true` berarti siswa sudah menyelesaikan game di level tersebut; `false` atau `null` berarti belum.

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

**Query Params:**
- `elemen_id` (wajib)
- `materi_id` (opsional)

**Response 200:**
```json
{
  "data": {
    "level_aktif": "mid",
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

> - `level_aktif`: level aktif siswa sekarang. Dipakai FE untuk mengisi `levelMap[activeKey]`.
> - `riwayat[].terkunci: true` = level sudah dilewati (siswa naik level) → quiz level ini **read-only**.
> - Riwayat per level: **1 record per tipe** (yang terbaru). BE simpan hanya hasil terakhir per `(siswa_id, elemen_id, materi_id, level, tipe)`.
> - `riwayat[].hasil_quiz_id`: dipakai FE untuk trigger CTA "Tanya Kak Nusa" dari panel riwayat quiz — tersedia meski siswa membuka kembali sesi lama.

**Response jika belum pernah quiz:**
```json
{ "data": { "level_aktif": "low", "riwayat": [] }, "meta": null, "error": null }
```

**Error 400:** "elemen_id wajib diisi."

---

### POST /siswa/:id/quiz/mc — Tim 6 BE
Submit jawaban Quiz Pilihan Ganda. BE menilai langsung karena kunci jawaban tersedia.

**Auth:** role `siswa` (hanya `id` sendiri)

**Request:**
```json
{
  "publish_id": "pub_mat_bil_aljabar_x1_20260501",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "level": "Low",
  "jawaban": {
    "q1": "1",
    "q2": "0",
    "q3": "2"
  }
}
```

> - `jawaban`: key = `id` soal. Value = string index pilihan
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

> - `menunggu_essay: true` → agregasi belum dihitung, essay belum dikerjakan
> - `menunggu_essay: false` → essay sudah dikerjakan, agregasi tersedia di field `agregasi`
> - `hasil_quiz_id`: FE simpan dan teruskan ke `POST /mentor/evaluasi` saat siswa klik CTA "Tanya Kak Nusa"
> - `naik_level`: selalu `false` dari endpoint ini — naik level baru ditentukan setelah agregasi MC+Essay selesai. BE push via WebSocket event `essay_dinilai`

---

### POST /siswa/:id/quiz/essay — Tim 6 BE + Tim 3 RAG
Submit jawaban Quiz Essay. BE forward ke Tim 3 RAG untuk dinilai secara asinkronus.

**Auth:** role `siswa` (hanya `id` sendiri)

**Request:**
```json
{
  "publish_id": "pub_mat_bil_aljabar_x1_20260501",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "level": "Low",
  "jawaban": {
    "e1": "Langkah pertama adalah memindahkan konstanta ke ruas kanan...",
    "e2": "Variabel adalah simbol yang mewakili nilai yang tidak diketahui..."
  }
}
```

> - `jawaban`: key = `id` soal essay. Value = string teks jawaban siswa

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

> - `menunggu_penilaian: true` → Tim 3 sedang menilai, FE tampilkan indikator loading di panel quiz
> - `nilai: null` → akan diisi setelah Tim 3 selesai, BE push via WebSocket
> - Jika Tim 3 gagal: BE retry otomatis dengan exponential backoff — FE tidak perlu spam retry
> - `hasil_quiz_id`: FE simpan untuk referensi CTA evaluasi

> **Logika agregasi & naik level (di BE, otomatis setelah essay dinilai):**
> ```
> agregasi = nilai_mc × 60% + nilai_essay × 40%
> naik_level = agregasi >= 75
> ```
> BE push hasil via WebSocket event `essay_dinilai` → FE update UI naik level tanpa polling.

---

### GET /siswa/:id/notifikasi
Siswa ambil semua notifikasi dari guru.

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

## 12. KONTEN — Tim 3 RAG + Tim 6 BE

> **Ownership endpoint:**
> - `POST /konten/generate` → **Tim 3 RAG** — generate konten dari VectorDB
> - `POST /konten/publish` → **Tim 6 BE** — simpan konten ke database MVP

---

### POST /konten/generate — Tim 3 RAG
Guru generate satu tipe konten per request. FE memanggil endpoint ini **13× paralel** saat klik "Generate Konten":
- `bacaan` × 3 level (Low/Mid/High)
- `quiz_pg` × 3 level
- `quiz_essay` × 3 level
- `flashcard` × 3 level
- `mindmap` × 1 (tanpa level)

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
  "jenjang": "X",
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata.",
  "tipe": "bacaan",
  "level": "Low"
}
```

> - `materi` dan `materi_id`: opsional — hanya diisi jika guru menentukan sub-materi spesifik
> - `atp`: opsional tapi **sangat disarankan**
> - `level`: `"Low"` | `"Mid"` | `"High"` — **null atau omit** untuk `mindmap`

**Response 200 (contoh `bacaan`):**
```json
{
  "data": {
    "konten_id": "konten_mat_bacaan_low_1746342000",
    "tipe": "bacaan",
    "level": "low",
    "content": {
      "text": "# Persamaan Linear\n\n## A. Pengertian\n...",
      "source": [
        { "judul": "Matematika SMA Kelas X", "penulis": "Kemendikbud", "tahun": "2022" }
      ]
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
          "soal": "Berapakah nilai x dari persamaan 2x + 3 = 7?",
          "pilihan": ["1", "2", "3", "4"],
          "jawaban": 1,
          "penjelasan": "2x + 3 = 7 → 2x = 7 - 3 = 4 → x = 4 ÷ 2 = 2, maka jawaban yang benar adalah indeks 1 (nilai '2')."
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

> `konten_id`: identifier per konten per level, di-generate Tim 3. FE simpan di state review guru untuk dipakai di `POST /konten/regenerate` jika guru klik "Ulangi".

**Struktur `content` per `tipe`:**

| tipe | Struktur `content` | Jumlah |
|------|-------------------|--------|
| `bacaan` | `{ "text": "markdown string", "source": [{ "judul", "penulis", "tahun" }] }` | — |
| `quiz_pg` | `{ "soal": [{ "id", "soal", "pilihan": string[], "jawaban": number, "penjelasan": string }] }` | 10 soal |
| `quiz_essay` | `{ "pertanyaan": [{ "id", "soal", "rubrik", "placeholder" }] }` | 5 pertanyaan |
| `flashcard` | `{ "cards": [{ "depan", "belakang" }], "source": [{ "judul", "penulis", "tahun" }] }` | 5–10 kartu |
| `mindmap` | `{ "nodes": [{ "id", "label", "parent_id", "penjelasan": string }] }` | — |

> `quiz_pg.soal[].jawaban` = **index integer** dari array `pilihan` (bukan string jawaban).  
> `quiz_pg.soal[].penjelasan` = ditampilkan FE **setelah** siswa submit quiz MC, tidak saat mengerjakan.  
> `mindmap.nodes[].penjelasan` = tooltip saat hover. Boleh string kosong `""` untuk node root.  
> `bacaan.source` & `flashcard.source` = array kosong `[]` jika tidak ada sumber spesifik.

> **Pretest (V3.4):** Tim 3 generate soal pretest bersamaan dengan ke-13 konten — **secara internal, tanpa request tambahan dari FE**. Trigger generate pretest terjadi pada **panggilan `POST /konten/generate` pertama** dari 13 panggilan paralel (Tim 3 RAG mendeteksi ini adalah batch baru berdasarkan kombinasi `mapel_id + elemen_id + materi_id + kelas_id`). Soal pretest disimpan langsung ke database Tim 6 BE. **Tidak dikembalikan** di response ini dan **tidak ditampilkan** di panel review guru.

**Error 422:** "elemen_id tidak dikenal di VectorDB."  
**Error 429:** "Terlalu banyak request. Coba beberapa saat lagi."

---

### POST /konten/regenerate — Tim 3 RAG
Guru minta generate ulang satu konten spesifik. Iterative refinement — Tim 3 menggunakan konten sebelumnya sebagai referensi.

Dipanggil saat guru klik **"Ulangi"** di panel review per konten per level.

**Auth:** role `guru`

**Request:**
```json
{
  "konten_id": "konten_mat_bacaan_low_1746342000",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "jenjang": "X",
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata.",
  "tipe": "bacaan",
  "level": "Low",
  "instruksi_revisi": "Tambahkan contoh soal cerita konteks kehidupan sehari-hari di setiap sub-bab"
}
```

> - `konten_id`: **wajib** — identifier dari response `POST /konten/generate`
> - `instruksi_revisi`: **wajib** — deskripsi perubahan yang diinginkan guru

**Response 200:**
```json
{
  "data": {
    "konten_id": "konten_mat_bacaan_low_1746342000",
    "tipe": "bacaan",
    "level": "low",
    "content": {
      "text": "# Persamaan Linear\n\n## A. Pengertian\n... (versi baru)",
      "source": [
        { "judul": "Matematika SMA Kelas X", "penulis": "Kemendikbud", "tahun": "2022" }
      ]
    },
    "dibuat_at": "2026-05-01T09:05:00.000Z"
  },
  "meta": null,
  "error": null
}
```

> `konten_id` tetap sama — hanya `content` yang berubah. FE replace konten di state review guru.

**Error 404:** "konten_id tidak ditemukan."  
**Error 422:** "instruksi_revisi wajib diisi untuk regenerate."  
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
      "content": { "text": "# Persamaan Linear\n...", "source": [{ "judul": "...", "penulis": "...", "tahun": "..." }] },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_bacaan_mid_1746342000",
      "tipe": "bacaan",
      "level": "Mid",
      "content": { "text": "# Persamaan Linear — Menengah\n...", "source": [] },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_bacaan_high_1746342000",
      "tipe": "bacaan",
      "level": "High",
      "content": { "text": "# Persamaan Linear — Lanjutan\n...", "source": [] },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_quiz_pg_low_1746342000",
      "tipe": "quiz_pg",
      "level": "Low",
      "content": { "soal": [{ "id": "q1", "soal": "...", "pilihan": ["..."], "jawaban": 1, "penjelasan": "..." }] },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_quiz_essay_low_1746342000",
      "tipe": "quiz_essay",
      "level": "Low",
      "content": { "pertanyaan": [{ "id": "e1", "soal": "...", "rubrik": "...", "placeholder": "..." }] },
      "disetujui": true
    },
    {
      "konten_id": "konten_mat_flashcard_low_1746342000",
      "tipe": "flashcard",
      "level": "Low",
      "content": { "cards": [{ "depan": "...", "belakang": "..." }], "source": [{ "judul": "...", "penulis": "...", "tahun": "..." }] },
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

> `konten_list` berisi **16 item total**:
> `bacaan×3 + quiz_pg×3 + quiz_essay×3 + flashcard×3 + mindmap×1 + game×3`
>
> FE **hanya boleh publish jika semua 16 item `disetujui: true`**.  
> Item `game` menggunakan `game_id` dari response `POST /game/generate` — tidak ada `konten_id` untuk item game.  
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

## 13. SESI — Tim 6 BE

> Sesi belajar adalah unit terkecil dari aktivitas siswa. Dimulai saat siswa membuka chatbot, berakhir saat menutup atau timeout.

### POST /sesi
Mulai sesi belajar baru. Dipanggil saat siswa membuka chatbot (setelah izin kamera diberikan).

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "materi_id": "mat__persamaan_linear",
  "publish_id": "pub_mat_bil_aljabar_x1_20260501"
}
```

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

> - `hasil_quiz`: **seluruh** quiz dalam sesi, bukan hanya yang terakhir
> - `last_quiz`: shortcut agregasi akhir untuk Tim 3
> - `violations`: kosong `[]` jika tidak ada pelanggaran

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
Riwayat emosi sepanjang satu sesi.

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
Riwayat percakapan satu sesi chatbot.

**Auth:** role `siswa` (sesi sendiri)

**Response 200:**
```json
{
  "data": [
    {
      "role": "user",
      "teks": "Aku bingung cara menyelesaikan 2x + 3 = 7",
      "dikirim_at": "2026-05-01T09:10:00.000Z"
    },
    {
      "role": "ai",
      "teks": "Tenang ya, kita mulai dari yang paling dasar...",
      "dikirim_at": "2026-05-01T09:10:05.000Z"
    }
  ],
  "meta": null,
  "error": null
}
```

---

## 14. PRETEST — Tim 3 RAG (generate) + Tim 6 BE (serve)

> Pretest **berbeda** dari quiz MC & essay di chatbot. Dipakai sekali untuk menentukan level awal konten siswa.  
> **Tim 3 RAG** meng-generate soal pretest bersamaan dengan generate konten (internal, tanpa request FE).  
> **Tim 6 BE** menyimpan dan melayani soal pretest ke siswa.  
> `GET /siswa/:id/pretest/status` → **Tim 6 BE** (lihat Seksi 11)

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
        "soal": "Seberapa familiar kamu dengan topik 'Bilangan dan Aljabar'?",
        "pilihan": [
          "Belum pernah mempelajari topik ini sama sekali",
          "Pernah mendengar tapi belum memahami konsepnya",
          "Sudah memahami sebagian konsep dasar",
          "Sudah memahami dan bisa menerapkan konsepnya"
        ]
      }
    ]
  },
  "meta": null,
  "error": null
}
```

> **Catatan keamanan:** Field `jawaban` (kunci jawaban) **tidak dikembalikan** ke client. Penilaian dilakukan server-side saat `POST /pretest/submit`.

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
    "pretest_mat_bil_aljabar_1": "2",
    "pretest_mat_bil_aljabar_2": "3",
    "pretest_mat_bil_aljabar_3": "1",
    "pretest_mat_bil_aljabar_4": "2",
    "pretest_mat_bil_aljabar_5": "3"
  }
}
```

**Response 200:**
```json
{
  "data": {
    "level": "mid",
    "nilai": 60,
    "benar": 3,
    "total": 5
  },
  "meta": null,
  "error": null
}
```

> **Logika level:**
> - `nilai >= 80` → `"high"`
> - `nilai >= 60` → `"mid"`
> - `nilai < 60` → `"low"`
>
> BE menyimpan hasil ini permanen. Status dapat diambil via `GET /siswa/:id/pretest/status`.

---

## 15. QUIZ — Tim 6 BE

> Endpoint quiz langsung berada di domain siswa (lihat Seksi 11):
> - `GET /siswa/:id/quiz` — riwayat quiz
> - `POST /siswa/:id/quiz/mc` — submit quiz MC
> - `POST /siswa/:id/quiz/essay` — submit quiz Essay

Tidak ada endpoint quiz yang berdiri sendiri di domain `/quiz`. Desain ini sengaja dipilih karena quiz adalah **milik siswa**, bukan resource independen.

---

## 16. RAG — Tim 3

> Semua endpoint yang memerlukan komputasi AI dari Tim 3 RAG dikumpulkan di domain `/rag`.

### POST /rag/rekomendasi
Rekomendasi elemen/materi berikutnya berdasarkan progress siswa.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "levels": {
    "bil_aljabar": "mid",
    "data_statistika": "low"
  },
  "sudah_selesai_ids": ["bil_aljabar", "data_statistika"],
  "sedang_dipelajari_ids": ["geometri"]
}
```

> Data ini diambil dari response `GET /siswa/:id/progress`.

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

> FE mengirim data KPI ini dari hasil `GET /siswa/:id/kpi`. BE meneruskan ke Tim 3 RAG untuk generate insight.

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

## 17. GAME — Tim 4

> Tim 4 deliver game dalam format **HTML string** (bukan URL). FE me-render via:
> ```html
> <iframe srcDoc={gameData.html_string} sandbox="allow-scripts allow-same-origin allow-forms" />
> ```
> Game menghasilkan **3 level** (Low/Mid/High). Tracking hanya boolean selesai/tidak selesai.
>
> **Trigger selesai (V3.4):** Game HTML mengirim event ke parent FE via `window.parent.postMessage`:
> ```javascript
> window.parent.postMessage({ type: 'game:selesai' }, '*');
> ```
> FE listen via `window.addEventListener('message', ...)` dan memanggil `PATCH /game/:id/penyelesaian`.  
> FE juga menerima format lama `'game:selesai'` (string) dan `{ event: 'game:selesai' }` untuk kompatibilitas.  
> Tim 4 **tidak** memanggil endpoint ini langsung — tanggung jawab FE.

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
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata.",
  "level": "Low"
}
```

> - `atp`: **wajib** — untuk menyesuaikan skenario game dengan tujuan pembelajaran

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

> - `game_id`: **wajib**
> - `instruksi_revisi`: **wajib**

**Response 200:**
```json
{
  "data": {
    "game_id": "game_1746342000_low",
    "nama": "Quest: Persamaan Linear",
    "deskripsi": "Game edukasi interaktif tentang Persamaan Linear — level low (revised)",
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

**Error 404:** "game_id tidak ditemukan."  
**Error 422:** "instruksi_revisi wajib diisi untuk regenerate."

---

### GET /game/:id — Tim 4
Detail satu game. Digunakan untuk polling saat status masih `"generating"`, dan untuk mengambil `html_string` saat siswa klik "Main Game".

**Auth:** role `siswa` (saat buka game) atau `guru` (saat polling setelah generate)

**Response 200:**
```json
{
  "data": {
    "game_id": "game_1746342000_low",
    "nama": "Quest: Persamaan Linear",
    "deskripsi": "...",
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

> `GET /siswa/:id/konten` **tidak** menyertakan `html_string` di item game — FE fetch via `GET /game/:id` saat siswa klik "Main Game".  
> Jika `status: "generating"` → `html_string: null` → FE poll setiap 3 detik hingga `status: "ready"`.

---

### PATCH /game/:id/penyelesaian — Tim 4
Catat bahwa siswa **menyelesaikan** game.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "level": "Low"
}
```

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

## 18. EMOSI — Tim 1

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

**Error 400:**
```json
{ "data": null, "meta": null, "error": { "code": "VALIDATION_ERROR", "message": "Frame tidak valid.", "details": { "emosi": "tidak_terdeteksi" } } }
```

---

## 19. MENTOR — Tim 5

> **Tanggung jawab Tim 5:**
> - Interaksi chatbot selama sesi belajar
> - Feedback evaluasi quiz via CTA badge "📊 Evaluasi Kuis" (endpoint terpisah)
> - CTA di panel quiz riwayat berlabel 'Tanya Kak Nusa' (trigger oleh siswa). Response AI masuk ke chat dengan badge '📊 Evaluasi Kuis' (tampilan di FE).

### POST /mentor/pesan
Kirim pesan ke mentor, tunggu full response. Fallback jika SSE tidak tersedia.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "sesi_id": "sesi_s1_20260501_bil_aljabar",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "atp": "Siswa mampu...",
  "level": "Mid",
  "pesan": "Aku bingung cara menyelesaikan 2x + 3 = 7",
  "konteks": {
    "emosi": "bingung",
    "publish_id": "pub_mat_bil_aljabar_x1_20260501",
    "bacaan": "# Persamaan Linear\n\n## A. Pengertian... (maks 3000 karakter)"
  }
}
```

> - `atp`: **wajib** — mentor menyesuaikan penjelasan dengan tujuan pembelajaran
> - `level`: level konten siswa saat ini
> - `konteks.emosi`: dari Tim 1, `null` jika tidak ada
> - `konteks.publish_id`: opsional — `null` jika konten belum dipublish
> - `konteks.bacaan`: opsional — `null` jika konten belum tersedia. Maks 3000 karakter. Tim 5 gunakan sebagai referensi utama saat menjawab pertanyaan siswa

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
Identik dengan `/mentor/pesan` tapi response via **SSE** untuk efek ketik streaming.

**Request Body:** sama persis dengan `/mentor/pesan`.

**Response:** `Content-Type: text/event-stream`

**Format SSE:**
```
data: Tenang \n\n
data: ya, \n\n
data: kita \n\n
data: mulai \n\n
data: [DONE]\n\n
```

> FE menggunakan `EventSource` atau `fetch` dengan `ReadableStream`.  
> Ketika `data: [DONE]` diterima, FE tutup koneksi.

---

### POST /mentor/evaluasi — Tim 5
Evaluasi hasil quiz siswa via CTA "Tanya Kak Nusa". Endpoint ini **terpisah** dari chat normal — Tim 5 menggunakan system prompt yang fokus pada analisis jawaban, bukan percakapan mentoring.

**Auth:** role `siswa`

**Request:**
```json
{
  "siswa_id": "s1",
  "sesi_id": "sesi_s1_20260501_bil_aljabar",
  "hasil_quiz_id": "hq_20260501_0001",
  "mapel_id": "mat",
  "elemen_id": "bil_aljabar",
  "elemen_label": "Bilangan dan Aljabar",
  "materi": "Persamaan Linear",
  "materi_id": "mat__persamaan_linear",
  "level": "Low",
  "atp": "Siswa mampu menjelaskan dan menyelesaikan persamaan linear satu variabel dalam konteks nyata."
}
```

> - `hasil_quiz_id`: **wajib** — dari response `POST /siswa/:id/quiz/mc` atau `/essay`. BE Tim 6 lookup dan inject seluruh data quiz ke Tim 5: soal, jawaban siswa, kunci jawaban (MC) / rubrik (essay), nilai per soal, nilai total
> - `atp`: **wajib** — Tim 5 framing evaluasi berdasarkan tujuan pembelajaran
> - Tidak ada field `pesan` — tidak ada input teks dari siswa di flow ini

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
Identik dengan `/mentor/evaluasi` tapi response via **SSE**. Sesuai flow aplikasi: feedback masuk ke chat sebagai streaming response dengan badge **"📊 Evaluasi Kuis"**.

**Request Body:** sama persis dengan `/mentor/evaluasi`.

**Response:** `Content-Type: text/event-stream`

**Format SSE:** identik dengan `/mentor/pesan/stream`.

```
data: Kamu \n\n
data: sudah \n\n
data: mengerjakan \n\n
data: [DONE]\n\n
```

---

## 20. LEADERBOARD — Tim 6 BE

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

## 21. NOTIFIKASI — Tim 6 BE

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
  "data": {
    "id": "notif_123",
    "dibuat_at": "2026-05-01T09:00:00.000Z"
  },
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

## 22. WEBSOCKET SPEC — Tim 6 BE

### 21.1 Koneksi

**22.1.1 WebSocket Guru — Monitoring Real-Time**

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

**22.1.2 WebSocket Siswa — Notifikasi Async (essay dinilai & naik level)**

**URL:**
```
wss://api.sekolahrakyat.id/v1/ws/siswa
```

**Query Params:**
```
?siswa_id={siswa_id}&sesi_id={sesi_id}&token={access_token}
```

> Digunakan siswa untuk menerima notifikasi async dari BE — khususnya event `essay_dinilai` setelah Tim 3 selesai menilai essay.  
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

Reconnect & fallback siswa: sama dengan aturan 21.5 — exponential backoff, refresh token jika expired. Jika WS tidak tersedia, FE siswa **poll** `GET /siswa/:id/quiz?elemen_id=` setiap 10 detik untuk update status nilai essay.

**Env:**
```
VITE_WS_URL=wss://api.sekolahrakyat.id/v1/ws
```

---

### 21.2 Handshake

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

### 21.3 Event Types (Server → Client)

Semua event menggunakan envelope dengan timestamp **ISO 8601 penuh**:
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
  "payload": {
    "sesi_id": "sesi_s1_20260501_bil_aljabar",
    "durasi_menit": 45
  },
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

> **`progress_pct`** di sini adalah progress per mapel siswa tersebut: `round(selesai / total_elemen × 100)`.

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
  "payload": {
    "detail": "Berpindah Tab / Menyembunyikan Halaman"
  },
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

> **Kondisi smart_alert server-side:**
> - `emosi_negatif_berkepanjangan`: emosi negatif (bosan/bingung/frustrasi) > 15 menit berturut-turut
> - `pelanggaran_aktif`: siswa melakukan pelanggaran

---

### 21.4 Event Types (Client → Server)

**`ping`** — Keepalive dari FE:
```json
{ "type": "ping" }
```

Server merespons dengan `pong`:
```json
{ "type": "pong", "timestamp": "2026-05-01T09:15:30.000Z" }
```

---

### 21.5 Reconnect & Fallback

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

## 23. HIRARKI KURIKULUM (ATURAN GLOBAL)

```
Kurikulum Merdeka
  └── Mapel       (Matematika)              ← fase & deskripsi_cp ada di sini
        └── Elemen   (Bilangan dan Aljabar)  ← elemen_id SELALU WAJIB
              └── Materi  (Persamaan Linear) ← opsional, diisi guru/siswa
```

### 22.1 Field Wajib di Semua Payload Content / Game / Mentor

| Field | Keterangan |
|-------|-----------|
| `mapel_id` | Selalu wajib |
| `elemen_id` | **Selalu wajib**, tidak boleh null atau omit |
| `elemen_label` | Wajib di semua payload mutasi (POST/PUT) — untuk konteks LLM & display |

### 22.2 Field Opsional

| Field | Keterangan |
|-------|-----------|
| `materi` | Nama materi (string label), hanya diisi jika guru/siswa turun ke level materi |
| `materi_id` | Format: `"{mapel_id}__{snake_case}"` — contoh: `"mat__persamaan_linear"` |
| `atp` | Alur Tujuan Pembelajaran — opsional tapi direkomendasikan untuk generate konten |

### 22.3 Aturan Validasi (Wajib Semua Tim)

- BE / Tim 3 / Tim 4 / Tim 5 **wajib menolak** payload yang punya `mapel_id` tapi **tidak punya `elemen_id`**
- FE membangun `materi_id`: `` materi_id = `${mapel_id}__${materi.toLowerCase().replace(/\s+/g, '_')}` ``
- `materi_id` hanya dikirim jika `materi` juga dikirim (keduanya sinkron)

---

## 24. STANDARD RESPONSE & ERROR

### 23.1 Response Envelope (Wajib Semua Endpoint)

```json
{
  "data": "<object|array|null>",
  "meta": "<object|null>",
  "error": "<object|null>"
}
```

### 23.2 Success Response

**Single resource:**
```json
{
  "data": { "id": "s1", "nama": "Budi" },
  "meta": null,
  "error": null
}
```

**Collection (tanpa pagination):**
```json
{
  "data": [{ "id": "s1" }, { "id": "s2" }],
  "meta": null,
  "error": null
}
```

**Collection (dengan pagination):**
```json
{
  "data": [{ "id": "s1" }, { "id": "s2" }],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "total_pages": 3
  },
  "error": null
}
```

**Action berhasil:**
```json
{
  "data": { "deleted": true },
  "meta": null,
  "error": null
}
```

### 23.3 Error Response

```json
{
  "data": null,
  "meta": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "elemen_id wajib diisi.",
    "details": {
      "field": "elemen_id",
      "constraint": "required"
    }
  }
}
```

**Multiple validation errors:**
```json
{
  "data": null,
  "meta": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Terdapat kesalahan validasi pada beberapa field.",
    "details": {
      "fields": [
        { "field": "elemen_id", "message": "Wajib diisi." },
        { "field": "mapel_id",  "message": "Nilai tidak dikenal." }
      ]
    }
  }
}
```

### 23.4 HTTP Status Code Reference

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

### 23.5 Caching Strategy (Rekomendasi)

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
| POST   | `/admin/kelas/:id/siswa`             | Tambah siswa ke kelas |
| DELETE | `/admin/kelas/:id/siswa/:siswa_id`   | Lepas siswa dari kelas |
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

### Kelas (Guru Monitoring)
| Method | Path | Keterangan |
|--------|------|-----------|
| GET | `/kelas/:id/progress` | Progress semua siswa di kelas |

> **Endpoint `/kelas/:id/progress`** — Tim 6 BE, dipakai guru di halaman monitoring (initial load sebelum WS aktif):

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

**Query Params:** `mapel_id` (wajib jika guru mengampu >1 mapel di kelas)

### Konten (Guru)
| Method | Path | Tim | Keterangan |
|--------|------|-----|-----------|
| POST | `/konten/generate` | Tim 3 RAG | Generate satu tipe konten (return `konten_id`) |
| POST | `/konten/regenerate` | Tim 3 RAG | Regenerate per konten per level via `konten_id` |
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
| POST | `/pretest/soal` | Tim 6 BE | Ambil soal pretest (soal di-generate Tim 3 RAG saat publish konten) |
| POST | `/pretest/submit` | Tim 6 BE | Submit jawaban pretest |

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
| PATCH | `/game/:id/penyelesaian` | Tim 4 | Catat penyelesaian game |

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

---

*— End of API Contract SR MVP V3.6 —*
