# Laporan Proyek Machine Learning Sistem Rekomendasi Buku - Ripan Renaldi

## Project Overview

  Buku merupakan sumber informasi dalam berbagai bidang, khususnya bidang pendidikan. Dalam suatu penelitian dijelaskan bahwa manfaat buku sangatlah banyak, diantaranya dapat memperluas wawasan, mencerdaskan akal dan pikiran, hingga dapat mengembangkan pribadi yang lebih baik [1]. Namun, UNESCO menyebutkan bahwa minat baca di Indonesia masihlah tergolong sangat rendah yakni sekitar 0.001%, yang artinya akan hanya ada satu orang dari tiap seribu orang yang ada. Dengan meningkatkan minat baca, SDM pun akan meningkat [2]. Tentunya buku ini sangatlah penting untuk kita di segala jenjang, baik itu dari SD hingga bangku perkuliahan.  
  
  Seperti yang kita ketahui, kita dapat meminjam buku di perpustakaan yang tersedia. Di sekolahan, kita dapat dengan mudah meminjam buku yang telah di sediakan. Seiring berjalannya waktu, koleksi buku yang ada di perpustakaan bertambah banyak. Hal tersebut berdampak terhadap banyaknya varian pilihan buku untuk dibaca oleh pengguna, hal tersebut juga berdampak terhadap kesulitan mencari buku yang diinginkan. berdasarkan hal tersebut. Diperlukan sebuah sistem yang dapat merekomendasikan buku yang sesuai dengen preferensi pengguna berdasarkan rating yang telah diberikan pengguna sebelumnya. Dengan demikian, hal itu perlu diselesaikan guna memberikan efisiensi kepada pengguna (pembaca buku) untuk memberikan rekomendasi buku, sehingga pemilihan buku dapat tidak memakan banyak waktu. Selain itu, dengan adanya sistem rekomendasi buku ini, pengguna dapat disuguhkan beberapa buku yang mungkin mereka sukai dan belum pernah dibaca sebelumnya.
  
## Business Understanding

### Problem Statements
Berdasarkan latar belakang masalah di atas, terdapat beberapa rumusan masalah yang akan diselesaikan:  
- Bagaimana cara mengefisiensikan waktu dalam mencari buku yang diinginkan pembaca?
- Berdasarkan rating dari pengguna, bagaimana sistem merekomendasikan buku yang belum pernah dibaca atau
mungkin disukai oleh pengguna?

### Goals
Berdasarkan rumusan masalah tersebut, tujuan yang akan dicapai yaitu : 
- Membuat sistem untuk merekomendasikan buku kepada pengguna.
- Membuat sistem rekomendasi yang sesuai dengan preferensi pengguna dengan teknik collaborative filtering.

### Solution statements
Adapun cara untuk meraih tujuan di atas yaitu : 
1. Membuat sistem rekomendasi menggunakan pendekatan atau teknik *Collaborative Filtering*


## Data Understanding
*Dataset* yang akan digunakan yakni *Book Recommendation Dataset* yang dapat diunduh pada tautan berikut [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
*Dataset* di atas berisi tiga buah file.
- Users merupakan data-data yang mengandung tentang user. File users ini memiliki 278858 baris data, dan 3 buah kolom seperti User-ID, Location, dan Age.
- Books merupakan file yang berisi mengenai informasi buku, seperti ISBN (kode unik buku), judul buku, pengarang buku, tahun publikasi, penerbit, hingga gambar buku berupa tautan link yang menuju amazon web site. File ini memiliki 271360 baris data dan 8 buah kolom. Tiap satu buah ISBN, mencakup satu buah buku. Selain itu, pada file ini terdapat beberapa missing value, tepatnya pada kolom book_author, publisher, dan image_url_l.
- Ratings merupakan file yang berisi rating yang telah diberikan pengguna kepada buku tertentu. File ini berjumlah 1149780 baris data dan 3 buah kolom yang mencakup user_id, isbn, dan book_rating.

Lebih lengkapnya, dataset di atas berisikan fitur-fitur berikut :  
- ISBN merupakan identitas unik suatu buku, satu ISBN untuk satu buku.
- book_title merupakan judul buku
- book_author merupakan pengarang
- year_of_publication merupakan tahun terbit buku
- publisher merupakan penerbit buku
- image_url_s merupakan gambar dari buku yang berukuran kecil
- image_url_m merupakan gambar dari buku yang berukuran sedang
- image_url_l merupakan gambar dari buku yang berukuran besar
- user_id merupakan id unik pengguna
- location menunjukkan lokasi pengguna
- age merupakan umur pengguna
- rating merupakan *rating* dari pengguna

Untuk mendapatkan pemahaman terhadap data yang ada, akan dilakukan exploratory data analysis yang mencakup : 
1. Melakukan pengecekan tipe data pada masing-masing *dataframe*.
2. memastikan satu isbn hanya mencakup satu buah buku.
3. Mengetahui jumlah data unik pada buku.
4. Melakukan pengecekan bahwa judul buku sama yang memiliki ISBN berbeda bukanlah entitas yang sama.
5. Mengetahui jumlah data unik pada *dataframe user*
6. Mengetahui jumlah pengguna yang memberikan *rating*, jumlah buku yang diberi *rating*, dan jumlah data pada *dataframe ratings*
7. Mengetahui rentang rating yang diberikan user terhadap buku.

Berdasarkan point di atas, didapatkan bahwa :
1. Tipe data yang ada pada data adalah int, object, dan float.
2. Satu isbn pada dataframe merujuk pada satu buah buku.
3. Terdapat 271360 buku yang unik, dengan 242135 judul yang unik.
4. Judul buku yang sama dengan ISBN yang berbeda merupakan dua buah buku yang berbeda. Hal ini dikarenakan satu buah judul buku bisa memiliki beberapa sequel lanjutannya.
5. Terdapat 278858 total pengguna.
6. Jumlah data rating yaitu 1149780 data, jumlah buku yang telah diberi rating yaitu 340556 data, dan jumlah user yang memberikan rating yaitu 105283 data.
7. Rentang rating berada antara 0 sampai 10.

## Data Preparation
Sebelum data benar-benar siap diolah oleh algoritma machine learning, perlu dilakukan beberapa tahapan terlebih dahulu. Tahapan tersebut meliputi : 
1. Menggabungkan data ratings dengan judul buku  
  Hal ini perlu dilakukan untuk melakukan pengecekan terhadap *missing value* nantinya. Dengan menggabungkan data ratings dengan judul buku, akan terlihat bahwa user memberikan rating terhadap isbn atau buku yang valid. 
2. Menangani missing value  
  Setelah dilakukan penggabungan, akan terlihat bahwa banyak sekali *missing value* pada judul buku yang berjumlah 118644 baris data. Hal ini terjadi karena ISBN pada baris data tertentu tidak valid, sehingga berdampak terhadap tidak ditemukannya judul buku pada *dataframe books*. Hal tersebut dapat dilihat pada data yang bernilai NaN. Oleh sebab itu, judul yang hilang tersebut perlu kita hilangkan karena ini merupakan data yang tidak valid. Mengingat data kita berjumlah satu juta lebih baris data setelah dilakukan penggabungan, sehingga tidak apa jika menghapus *missing value* tersebut. Setelah dilakukan pembersihan *missing value*, tersisa 1031136 baris dat dari 1149780 baris data.
3. Membuat *dataframe* baru yang berisi isbn dan judul buku  
  *Dataframe* yang baru ini akan digunakan untuk mendapatkan list buku yang belum dibaca saat mendapatkan rekomendasi nantinya.
4. Melakukan *encoding* terhadap fitur user_id dan isbn  
5. Menambah kolom baru (*mapping*) berdasarkan user_id dan isbn yang telah di *encode* tadi.
6. Mengubah tipe data pada kolom book_rating menjadi float  
  Hal ini dilakukan karena kita akan mengubah rentang rating menjadi 0-1.
7. Membagi data latih dan data validasi  
  Porsi pembagian data latih sebesar 85%, dan 15% data uji. Angka tersebut saya ambil karena data yang kita miliki berjumlah banyak yakni lebih dari satu juta data. Data latih ini digunakan untuk melatih model, sedangkan data validasi digunakan agar model yang kita latih tercegah dari *overfitting*.
  
## Modeling
Setelah data selesai melalui tahap preparation dan preprocessing, selanjutnya akan dibuat model sistem rekomendasi yang menghasilkan top-n rekomendasi buku untuk pengguna. Dalam penyelesaian rumusan masalah di atas, saya akan menggunakan pendekatan *Collaborative Filtering*. Jika dibandingkan dengan teknik *Content Based Filtering*, Content Based Filtering ini bergantung kepada konten tiap item. Sehingga pendekatan *Collaborative Filtering* ini lebih cocok untuk digunakan pada studi kasus ini, khususnya karena skala user pada data banyak. Selain itu, pendekatan ini dapat melihat ketertarikan pengguna yang lebih spesifik.  
Cara kerja *Collaborative Filtering* ini yaitu dengan memprediksi preferensi pengguna untuk suatu item atau layanan dengan mempelajari item pengguna dari sekelompok *user* yang mempunyai kesamaan preferensi dan minat di masa lalu [3].

## Evaluation
Pada studi kasus ini, metrik evaluasi yang akan digunakan yaitu *Root Mean Squared Error*  
![RMSE](https://github.com/RipanRenaldi/project-02-sistem-rekomendasi-buku/blob/main/assets/rmse_formula.png?raw=True)  
  Gambar 1. Formula MSE  
  
![MSE](https://github.com/RipanRenaldi/project-02-sistem-rekomendasi-buku/blob/main/assets/mse_formula.png?raw=True)  
  Gambar 2. Formula RMSE  
  
Cara kerja RMSE ini sebenernya kurang lebih sama seperti MSE. MSE akan menghitung selisih hasil prediksi dengan hasil sebenarnya, hasil ini cenderung akan menghasilkan nilai yang besar karena MSE akan mengkuadratkan hasil selisih tersebut sebelum dibagi total data untuk mendapatkan rata-ratanya. Oleh sebab itu, RMSE berperan untuk memperkecil skalanya dengan mengakarkan hasil dari MSE yang telah dihitung.  
Jika dikaitkan pada studi kasus di atas, error yang dihasilkan pada proses *training* data yaitu sebesar 0.323. Sedangkan erro pada data validasi yaitu sebesar 0.338. Berikut merupakan visualisasi evaluasi metric RMSE terhadap 25 epochs pada saat training.  
![evaluasi](https://github.com/RipanRenaldi/project-02-sistem-rekomendasi-buku/blob/main/assets/evaluation.png?raw=True)  
  Gambar 3. Evaluasi *Metric* RMSE  

### Kesimpulan
Berdasarkan gambar 3, model mendapatkan error yang cukup kecil yaitu memiliki nilai error akhir di 0.338 untuk data yang baru. Berdasarkan hal tersebut, pembuatan model dengan pendekatan *Collaborative Filtering* ini dapat digunakan untuk merekomendasikan buku yang belum pernah dibaca atau mungkin disukai pengguna. Selain itu, kini pengguna dapat mempersingkat waktu pencarian buku dengan memanfaatkan hasil rekomendasi yang telah diberikan oleh model.  

## Daftar Pustaka
[1]	Moh. Irfan, D. A. C, and H. F. R, “SISTEM REKOMENDASI: BUKU ONLINE DENGAN  METODE COLLABORATIVE FILTERING,” JURNAL TEKNOLOGI TECHNOSCIENTIA, vol. 7, no. 1, pp. 76–84, Aug. 2014.  
[2]	S. Kasiyun, “UPAYA MENINGKATKAN MINAT BACA SEBAGAI SARANA UNTUK  MENCERDASKAN BANGSA,” JURNAL PENA INDONESIA (JPI), vol. 1, no. 1, pp. 79–95, Mar. 2015.  
[3]	G. Indah Marthasari, Y. Azhar, and D. Kurnia Puspitaningrum, “SISTEM REKOMENDASI PENYEWAAN  PERLENGKAPAN PESTA MENGGUNAKAN  COLLABORATIVE FILTERING DAN PENGGALIAN  ATURAN ASOSIASI,” Jurnal SimanteC, vol. 5, no. 1, pp. 1–8, Dec. 2015.  
