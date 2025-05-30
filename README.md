# Book Recommendation System - Rosalia Indah Dwi Putriningsih

## Project Overview

Di era digital, sistem rekomendasi telah menjadi komponen penting dalam berbagai layanan untuk meningkatkan pengalaman pengguna dan konversi bisnis. Pada konteks literasi, sistem rekomendasi buku dapat membantu pembaca menemukan buku yang relevan dengan minat mereka, sekaligus mendorong peningkatan minat baca masyarakat. Fakta menunjukkan bahwa tingkat literasi Indonesia masih tergolong rendah [[1](https://www.tribunnews.com/nasional/2021/03/22/tingkat-literasi-indonesia-di-dunia-rendah-ranking-62-dari-70-negara)].

Proyek ini mengangkat topik **Pembuatan Sistem Rekomendasi Buku menggunakan Metode Collaborative Filtering**, dengan tujuan memberikan rekomendasi personal bagi pengguna berdasarkan histori rating mereka.

## Business Understanding

### Problem Statements
- Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana merekomendasikan buku yang belum dibaca oleh pengguna berdasarkan pola penilaian mereka?

### Goals
Tujuan dari proyek ini adalah:
- Menghasilkan 10 rekomendasi buku terbaik untuk pengguna tertentu berdasarkan data rating sebelumnya.
- Mengevaluasi performa model menggunakan metrik RMSE untuk melihat seberapa akurat prediksi terhadap rating aktual.

### Solution Statement
Solusi yang akan diterapkan pada proyek ini adalah menggunakan metode [Collaborative Filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics).

Metode ini akan menghasilkan rekomendasi sejumlah buku yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna dapat digunakan untuk mengidentifikasi buku-buku yang mirip dan belum pernah diberi rating oleh pengguna untuk direkomendasikan.

**Kelebihan** :
- Tidak diperlukan pengetahuan domain karena embeddings mempelajarinya secara otomatis.
- Dapat membantu pengguna menemukan minat baru
- Sistem hanya membutuhkan matriks umpan balik (feedback, misalnya berup rating) untuk melatih model faktorisasi matriks. Secara khusus, sistem tidak memerlukan fitur kontekstual.

**Kekurangan** :
- Model tidak dapat memberikan rekomendasi kepada pengguna baru karena sistem ini bergantung pada preferensi atau umpan balik yang ada (cold-start problem).
- Sulit untuk mengikutsertakan fitur lain (side-features) pada kueri atau item


## Data Understanding

Informasi Dataset:

Jenis | Keterangan
--- | ---
Title | Book Recommendation Dataset
Source | [Kaggle](https://www.kaggle.com/arashnic/book-recommendation-dataset)
Maintainer | [Möbius](https://www.kaggle.com/arashnic)
License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Usability | 10.0

Pada Dataset ini terdapat 3 berkas csv diantaranya yaitu `Books.csv` , `Ratings.csv` , dan `Users.csv`

Pada berkas `Books.csv` memuat data-data buku yang terdiri dari 271.360 baris dan memiliki 8 kolom, diantaranya adalah :

- `ISBN` : berisi kode ISBN dari buku
- `Book-Title` : berisi judul buku
- `Book-Author` : berisi penulis buku
- `Year-Of-Publication` : tahun terbit buku
- `Publisher` : penerbit buku
- `Image-URL-S` : URL menuju gambar buku berukuran kecil
- `Image-URL-M` : URL menuju gambar buku berukuran sedang
- `Image-URL-L` : URL menuju gambar buku berukuran besar

Pada berkas `Ratings.csv` memuat data rating buku yang diberikan oleh pengguna. Data ini memiliki 1.149.780 baris dan memiliki 3 kolom, yaitu :

 - `User-ID` : berisi ID unik pengguna
 - `ISBN` : berisi kode ISBN buku yang diberi rating oleh pengguna
 - `Book-Rating` : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

   | Column       Non-Null Count    Dtype 
---  ------       --------------    ----- 
 0   User-ID      1149780 non-null  int64 
 1   ISBN         1149780 non-null  object
 2   Book-Rating  1149780 non-null  int64 
dtypes: int64(2), object(1)

Pada data `Rating` ini juga ditemukan bahwa `User-ID` berupa ID angka yang berukuran cukup besar. Lalu `ISBN` merupakan string unik identitas buku gabungan angka dan huruf. Kedua nilai ini nantinya perlu dilakukan encoding agar dapat menghasilkan rekomendasi. Data rating ini juga merupakan data utama dalam membuat sistem rekomendasi dengan Collaborative Filtering pada proyek ini.

Pada berkas `Users.csv` memuat data pengguna. Data ini terdiri dari 278.858 baris dan memiliki 3 kolom, yaitu :

- `User-ID` : berisi ID unik pengguna
- `Location` : berisi data lokasi pengguna
- `Age` : berisi data usia pengguna

Berikut ini adalah hasil dari visualiasi jumlah rating buku yang diberikan oleh user.

|Book-Rating|User-ID|ISBN|
|---|---|---|
|0|716109|716109|
|1|1770|1770|
|2|2759|2759|
|3|5996|5996|
|4|8904|8904|
|5|50974|50974|
|6|36924|36924|
|7|76457|76457|
|8|103736|103736|
|9|67541|67541|
|10|78610|78610|

Pada data di atas dapat diketahui bahwa mayoritas user - ada lebih dari 700 ribu yang memberikan rating 0 pada buku sehingga data ini dikatakan tidak seimbang *(imbalance)*. Untuk itu pada data ini nantinya akan dilakukan penanganan agar dapat lebih seimbang.

## Data Preparation
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Handling Imbalanced Data** : Seperti yang telah diketahui sebelumnya bahwa jumlah rating tidak seimbang (imbalance) yang mana sebagian besar user memberikan rating 0 pada buku. Hal ini dapat mengakibatkan model memiliki kinerja yang buruk. Untuk mengatasi hal tersebut, pada proyek ini data dengan rating 0 akan dihapus *(di-drop)*. Walaupun jumlah data saat ini berkurang drastis namun distribusi data menjadi lebih seimbang dan diharapkan memiliki kinerja yang lebih baik.
- **Encoding** : dilakukan untuk menyandikan `User-ID` dan `ISBN` ke dalam indeks integer. Tahapan ini diperlukan karena kedua data tersebut berisi integer yang tidak berurutan (acak) dan gabungan string. Untuk itu perlu diubah ke dalam bentuk indeks.
- **Randomize Dataset** : pengacakan data agar distribusi datanya menjadi random. Pengacakan data bertujuan untuk mengurangi varians dan memastikan bahwa model tetap umum dan *overfit less*. Pengacakan data juga memastikan bahwa data yang digunakan saat validasi merepresentasikan seluruh distribusi data yang ada.
- **Data Standardization** : Pada data rating yang digunakan pada proyek ini berada pada rentang 0 hingga 10. Penerapan standarisasi menjadi rentang 0 hingga 1 dapat mempermudah saat proses training. Hal ini dikarenakan variabel yang diukur pada skala yang berbeda tidak memberikan kontribusi yang sama pada model fitting & fungsi model yang dipelajari dan mungkin berakhir dengan menciptakan bias jika data tidak distandarisasi terlebih dulu.
- **Data Splitting** : dataset dibagi menjadi 2 bagian, yaitu data yang akan digunakan untuk melatih model (sebesar 80%) dan data untuk memvalidasi model (sebesar 20%). Tujuan dari pembagian data uji dan validasi tidak lain adalah untuk proses melatih model serta mengukur kinerja model yang telah didapatkan.

## Modeling
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding.

Beberapa properti yang digunakan dalam kelas RecommenderNet dan menjadi parameter pada layer embedding untuk menghasilkan model diantaranya:
- `num_users` : jumlah data pengguna
- `num_isbn` : jumlah data buku, dihitung berdasarkan ISBN
- `embedding_size` : ukuran atau dimensi yang digunakan dalam embedding pada data user dan buku

Pertama, kita melakukan proses embedding terhadap data user dan buku. Jumlah user dan buku yang didefinisikan pada `num_users` dan `num_isbn` bertujuan sebagai input untuk membuat vektor embedding keduanya. Sedangkan `embedding_size` menentukan ukuran atau dimensi embedding yang dibuat. Semakin besar nilai dari `embedding_size` akan membuat model semakin akurat, namun jika berlebihan akan mengakibatkan model menjadi overfit. Untuk itu pada proyek ini juga menggunakan `optuna` untuk mencari nilai yang optimal. Selanjutnya, dilakukan operasi perkalian *dot product* antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Model ini juga di-compile dengan fungsi loss binarycrossentropy dan menggunakan Adam sebagai optimizer dengan learning rate sebesar 0.001

Model yang telah dibuat dapat menghasilkan top-10 rekomendasi buku seperti berikut ini.

4676/4676 ━━━━━━━━━━━━━━━━━━━━ 3s 689us/step
Rekomendasi untuk User ID: 129008
==============================

Buku dengan rating tertinggi dari user:
----------------------------------------
The Slippery Slope (A Series of Unfortunate Events, Book 10) - Lemony Snicket
The Honk and Holler Opening Soon - Billie Letts
The Iron Tonic: Or, A Winter Afternoon in Lonely Valley - Edward Gorey
The Day Jimmy's Boa Ate the Wash - Trinka Hakes Noble
Cypress Grove - James Sallis

Rekomendasi Buku Terbaik untuk User Ini:
----------------------------------------
Lonesome Dove - Larry McMurtry
The Return of the King (The Lord of the Rings, Part 3) - J.R.R. TOLKIEN
Harry Potter and the Sorcerer's Stone (Book 1) - J. K. Rowling
The Two Towers (The Lord of the Rings, Part 2) - J. R. R. Tolkien
The Giving Tree - Shel Silverstein
Calvin and Hobbes - Bill Watterson
52 Deck Series: 52 Ways to Celebrate Friendship - Lynn Gordon
My Sister's Keeper : A Novel (Picoult, Jodi) - Jodi Picoult
Dilbert: A Book of Postcards - Scott Adams
Harry Potter and the Chamber of Secrets Postcard Book - J. K. Rowling

## Evaluation
Pada proyek ini menggunakan metrik RMSE (Root Mean Square Error) untuk mengevaluasi kinerja model yang dihasilkan. RMSE adalah cara standar untuk mengukur kesalahan model dalam memprediksi data kuantitatif [[2](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)]. Root Mean Squared Error (RMSE) mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Perhitungan RMSE ditunjukkan pada rumus berikut ini.

![RMSE](https://i.postimg.cc/tgjfntZk/RMSE.png) 

`RMSE` = nilai root mean square error

`y`  = nilai hasil observasi

`ŷ`  = nilai hasil prediksi

`i`  = urutan data

`n`  = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Berikut ini adalah plot metrik RMSE setelah proses pelatihan model.

![Model Metrics](https://i.postimg.cc/m2n2YY5W/Model-Metrics.png)

Pada plot di atas dapat diketahui bahwa model memiliki skor nilai RMSE sebesar 0.185 yang mana bisa dikatakan sudah cukup bagus. Namun, meskipun begitu masih dapat dikembangkan lebih lanjut untuk meminimalkan error.

## Referensi

[[1](https://www.tribunnews.com/nasional/2021/03/22/tingkat-literasi-indonesia-di-dunia-rendah-ranking-62-dari-70-negara)] Utami, L. D. (2021). *Tingkat Literasi Indonesia di Dunia Rendah, Ranking 62 Dari 70 Negara*. Tribunnews. https://www.tribunnews.com/nasional/2021/03/22/tingkat-literasi-indonesia-di-dunia-rendah-ranking-62-dari-70-negara

[[2](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)] Moody, J. (2019). *What does RMSE really mean?*. Towards Data Science. https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e

