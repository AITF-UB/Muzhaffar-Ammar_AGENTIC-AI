Orkstrasi Generate data sft

1. I Have Metadata on  C:\Users\zara\Downloads\instruction\data

Kurikulum Merdeka/
└── SMA/
    ├── Kelas 10/
    │   ├── Matematika.json
    │   ├── Biologi.json
    │   └── ...
    │
    ├── Kelas 11/
    │   ├── Matematika.json
    │   └── ...
    │
    └── Kelas 12/
        ├── Bahasa Indonesia.json
        └── ...


the structure is 
[
  {
    "jenjang": "SMA",
    "kelas": "Kelas 12",
    "mata_pelajaran": "Matematika Tingkat Lanjut",
    "bab_judul": "Geometri Analitik",
    "sub_bab": "Lingkaran dan Garis Singgung",
    "keywords": [
      "Persamaan Lingkaran",
      "Titik Pusat",
      "Jari-jari",
      "Kedudukan Garis",
      "Garis Singgung Lingkaran"
    ]
  },
  {
    "jenjang": "SMA",
    "kelas": "Kelas 12",
    "mata_pelajaran": "Matematika Tingkat Lanjut",
    "bab_judul": "Geometri Analitik",
    "sub_bab": "Irisan Kerucut",
    "keywords": [
      "Parabola",
      "Elips",
      "Hiperbola",
      "Direktris",
      "Fokus",
      "Eksentrisitas"
    ]
  }
]


3. Grab one json then add on 
[KONTEKS_PEMBELAJARAN]
{context}

on each user prompt


4. First cycle create MATERI first until end (based on wat subject we already oin)
so we Have materi_[jenjang]_[subjects].json

5. Second cycle create flashcard until end  (based on wat subject we already oin)
so we ave flashcards_[jenjang]_[subjects].json

6. Second cycle create mindmap until end  (based on wat subject we already oin)
so we ave mindmap_[jenjang]_[subjects].json

7. Second cycle create bank soal pilgan until end  (based on wat subject we already oin)
so we ave pilgan_[jenjang]_[subjects].json

8. Second cycle create bank soal essay until end  (based on wat subject we already oin)
so we ave essay_[jenjang]_[subjects].json


the folder structure is mirrorin


Kurikulum Merdeka/
└── SMA/
    ├── Kelas 10/
    │   ├── Matematika
    |             |
			materi....json
			flashcard...json
			mindmap...json
			pilgan...json
			essay..json

9. Then we Have leveling LOTS/MOTS/HOTS
it will affect all task except mindmap

i want it dynamically for it as you can see the leveling criteria 

[KRITERIA_PEMBELAJARAN_MENDALAM]

1. LEVEL: LOTS (Memahami)
   - Taxonomy Bloom & SOLO: C1 (Mengingat), C2 (Memahami). Unistructural & Multistructural.
   - Fokus: Menghubungkan pengetahuan baru dengan pengalaman sebelumnya dan konteks sehari-hari.
   - Kedalaman: Mengidentifikasi, mendeskripsikan, menyebutkan, dan mengikuti prosedur sederhana. Menanamkan nilai moral dasar.

2 ...

3 ...

you could save it into list or any data structure tHen it support dynamically picking

like when we have materi task it should be

materi dari sumber json a level lots
materi dari sumber json a level mots
materi dari sumber json a level Hots
materi dari sumber json b level lots
dsb

it also for flascard, essay, pilgan but we HAve to finised the materi first before we move to flashcard


add --flag to control which task i want to produce data also for test and production mode

make one config to choose what the piloting subject what the model on open router we used now

for piloting we use matematika, sosiologi, and Bahasa Indonesia on kelas 12. also add some configs file to determine which metadata on C:\Local D\Galeri Belajar\Project\SR_02\data_prep_for_sft_v2\data\structured_output we used


please make it scalable and easy debug
create scalabel project strcuture to easy debug 




 

			
        


