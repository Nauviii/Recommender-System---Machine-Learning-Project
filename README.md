# Laporan Proyek Machine Learning - Fikri Kurnia

## Project Overview

Perkembangan pesat platform digital seperti layanan *streaming* film dan video on-demand telah menyebabkan peningkatan jumlah konten yang tersedia bagi pengguna. Kondisi ini menimbulkan permasalahan *information overload*, di mana pengguna kesulitan menemukan konten yang sesuai dengan preferensi mereka secara efisien. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan item yang relevan dengan memanfaatkan pola interaksi pengguna terhadap konten yang tersedia.

Pendekatan *Collaborative Filtering* (CF) merupakan salah satu metode yang paling banyak digunakan dalam sistem rekomendasi karena kemampuannya memanfaatkan pola kesamaan preferensi antar pengguna maupun antar item. Namun, pendekatan CF konvensional menghadapi tantangan utama berupa tingginya tingkat *sparsity* pada matriks interaksi user–item, terutama pada skala data yang besar. Dalam konteks dataset film yang digunakan pada proyek ini, analisis awal menunjukkan bahwa lebih dari 98% pasangan user–item tidak memiliki interaksi, serta sekitar dua pertiga item tergolong sebagai *cold-start item*. Kondisi ini dapat menurunkan kualitas rekomendasi jika tidak ditangani dengan pendekatan yang tepat.

Untuk mengatasi permasalahan tersebut, proyek ini mengimplementasikan pendekatan **Neural Collaborative Filtering (NFC)**, yang menggabungkan konsep *collaborative filtering* dengan kemampuan *neural network* dalam mempelajari representasi laten. Dengan memanfaatkan embedding untuk merepresentasikan pengguna dan film dalam ruang berdimensi rendah, NFC diharapkan mampu melakukan generalisasi yang lebih baik pada data yang bersifat sparse dibandingkan metode CF tradisional. Model yang dibangun bertujuan untuk mempelajari pola preferensi pengguna berdasarkan riwayat interaksi dan menghasilkan rekomendasi film dalam bentuk *Top-K recommendation*.

Proyek ini tidak hanya berfokus pada pembangunan model, tetapi juga melakukan analisis data secara menyeluruh, mencakup analisis *sparsity*, *cold-start*, serta evaluasi berbasis metrik *ranking* seperti Precision@K, Recall@K, HitRate@K, dan NDCG@K. Dengan demikian, proyek ini diharapkan dapat memberikan gambaran komprehensif mengenai tantangan dan performa sistem rekomendasi berbasis Neural Collaborative Filtering pada data dunia nyata yang bersifat sangat jarang (*sparse*).

**Referensi**  
[1] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative Filtering,” *Proceedings of the 26th International World Wide Web Conference (WWW)*, pp. 173–182, 2017.  
[2] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” *IEEE Computer*, vol. 42, no. 8, pp. 30–37, 2009.  
[3] J. Bobadilla, F. Ortega, A. Hernando, and A. Gutiérrez, “Recommender systems survey,” *Knowledge-Based Systems*, vol. 46, pp. 109–132, 2013.


## Business Understanding

Bagian ini menjelaskan proses klarifikasi masalah yang menjadi dasar pengembangan sistem rekomendasi film pada proyek ini. Fokus utama proyek adalah membangun sistem rekomendasi yang mampu memberikan rekomendasi film yang relevan kepada pengguna berdasarkan riwayat interaksi, meskipun data yang tersedia bersifat sangat jarang (*sparse*) dan memiliki karakteristik *cold-start* pada item.

### Problem Statements

Berdasarkan analisis awal terhadap data dan konteks permasalahan, diperoleh beberapa pernyataan masalah sebagai berikut:

1. **Bagaimana cara memberikan rekomendasi film yang relevan kepada pengguna di tengah tingginya jumlah konten dan keterbatasan waktu pengguna untuk mengeksplorasi film secara manual?**  
   Pengguna dihadapkan pada banyak pilihan film, sehingga diperlukan sistem yang mampu menyaring dan merekomendasikan film sesuai preferensi mereka.

2. **Bagaimana membangun sistem rekomendasi yang tetap mampu bekerja secara efektif pada data dengan tingkat sparsity yang sangat tinggi?**  
   Matriks interaksi user–item pada dataset yang digunakan memiliki tingkat sparsity lebih dari 98%, yang dapat menurunkan performa metode collaborative filtering konvensional.

3. **Bagaimana mengevaluasi kualitas rekomendasi film secara tepat dari sisi relevansi dan peringkat (ranking) rekomendasi?**  
   Evaluasi sistem rekomendasi tidak cukup hanya menggunakan metrik kesalahan prediksi (misalnya RMSE), tetapi memerlukan metrik berbasis peringkat seperti Precision@K, Recall@K, HitRate@K, dan NDCG@K untuk mencerminkan pengalaman pengguna secara nyata.

### Goals

Tujuan proyek ini dirancang untuk menjawab pernyataan masalah di atas, yaitu:

1. **Mengembangkan sistem rekomendasi film yang mampu menghasilkan daftar rekomendasi Top-K yang relevan untuk setiap pengguna.**  
   Sistem diharapkan dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka secara efisien.

2. **Menerapkan pendekatan model yang mampu menangani permasalahan sparsity pada data interaksi user–item.**  
   Dengan memanfaatkan representasi laten melalui embedding, model diharapkan dapat melakukan generalisasi yang lebih baik pada data yang jarang.

3. **Melakukan evaluasi sistem rekomendasi menggunakan metrik berbasis ranking untuk mengukur kualitas rekomendasi secara komprehensif.**  
   Evaluasi dilakukan menggunakan Precision@K, Recall@K, HitRate@K, dan NDCG@K untuk memastikan rekomendasi yang dihasilkan tidak hanya akurat, tetapi juga relevan dari sudut pandang pengguna.


## Data Understanding

Dataset yang digunakan pada proyek ini adalah **dataset rating film MovieLens**, yang berisi interaksi antara pengguna dan film dalam bentuk rating eksplisit. Dataset ini banyak digunakan dalam penelitian dan pengembangan sistem rekomendasi karena merepresentasikan skenario dunia nyata dengan karakteristik data yang *sparse* dan beragam preferensi pengguna.

### Sumber Dataset:
Dataset yang digunakan berasal dari platform kaggle.
- https://www.kaggle.com/datasets/grouplens/movielens-latest-small

### Jumlah baris, kolom, dan kondisi data
Pada proyek ini, data difokuskan pada file rating yang berisi informasi utama untuk membangun sistem rekomendasi berbasis *collaborative filtering*. Dataset yang digunakan yaitu, **ratings.csv** dan **movies.csv**.
Masing-masing dataset memiliki:
1. ratings.csv
   - Jumlah baris dan kolom, yaitu (100836 baris, 4 kolom)
   - Tidak terdapat missing value
   - Tidak terdapat data duplikat
  
2. movies.csv
   - Jumlah baris dan kolom, yaitu (100836 baris, 3 kolom)
   - Tidak terdapat missing value
   - Tidak terdapat data duplikat

### Deskripsi Variabel

Berikut adalah penjelasan setiap variabel yang digunakan dalam dataset yang digunakan:

### 1. ratings.csv

| Nama Kolom | Deskripsi | Tipe Data |
|------------|-----------|-----------|
| **userId** | Merupakan identitas unik pengguna yang memberikan rating terhadap film. Variabel ini digunakan untuk merepresentasikan entitas pengguna dalam sistem rekomendasi. | int |
| **movieId** | Merupakan identitas unik film yang diberi rating oleh pengguna. Variabel ini merepresentasikan item yang akan direkomendasikan. | int |
| **rating** | Nilai numerik yang diberikan pengguna terhadap film tertentu. Rating bersifat eksplisit dan mencerminkan tingkat preferensi pengguna. Dalam dataset MovieLens, rating umumnya berada pada rentang 0.5 hingga 5.0. | float |
| **timestamp** | Menunjukkan waktu ketika rating diberikan. Pada proyek ini, variabel timestamp tidak digunakan secara langsung dalam pemodelan, namun dapat dimanfaatkan untuk analisis temporal pada pengembangan lanjutan. | int |

---

### 2. movies.csv

| Nama Kolom | Deskripsi | Tipe Data |
|------------|-----------|-----------|
| **movieId** | Merupakan identitas unik film yang diberi rating oleh pengguna. Variabel ini merepresentasikan item yang akan direkomendasikan. | int |
| **title** | Judul dari setiap film yang terdapat pada dataset. | object |
| **genres** | Berisi gabungan genre dari setiap film pada dataset. | object |


### Tahapan Awal Pemahaman Data (Exploratory Data Analysis)

Untuk memahami karakteristik data sebelum pemodelan, beberapa tahapan eksplorasi dilakukan, antara lain:

1. **Analisis Distribusi Rating**  
   Dilakukan untuk mengetahui kecenderungan pengguna dalam memberikan rating (misalnya apakah pengguna cenderung memberi rating tinggi atau rendah).

2. **Analisis Sparsity**  
   Matriks user–item dianalisis untuk mengetahui tingkat kelangkaan interaksi. Hasil analisis menunjukkan bahwa dataset memiliki tingkat sparsity sekitar **98.30%**, yang mengindikasikan bahwa hanya sebagian kecil film yang pernah diberi rating oleh setiap pengguna.

3. **Cold-Start Analysis**  
   Dilakukan untuk mengidentifikasi jumlah pengguna dan film yang memiliki sangat sedikit atau bahkan tidak memiliki interaksi:
   - Cold-start items (film): ±66% dari total film
   - Cold-start users: 0%  
   Hasil ini menunjukkan bahwa permasalahan *cold-start* lebih dominan terjadi pada sisi item dibandingkan pengguna.

Hasil dari tahap Data Understanding ini menjadi dasar pemilihan pendekatan **Collaborative Filtering berbasis Neural Network (Neural Collaborative Filtering)**, yang diharapkan mampu menangkap pola laten preferensi pengguna meskipun data bersifat sparse.


## Data Preparation

Tahap **Data Preparation** dilakukan untuk memastikan data yang digunakan bersih, konsisten, dan siap digunakan dalam pengembangan sistem rekomendasi berbasis *Collaborative Filtering*. Seluruh tahapan dilakukan secara berurutan sesuai dengan implementasi pada notebook.

### 1. Fixing the Mismatch between `movieId` and `title`

Pada tahap awal, ditemukan bahwa terdapat beberapa film dengan **judul yang sama tetapi memiliki `movieId` yang berbeda**. Kondisi ini berpotensi menimbulkan ambiguitas dalam analisis, khususnya pada tahap eksplorasi data dan interpretasi hasil rekomendasi.

Untuk mengatasi hal ini, dilakukan pemeriksaan kesesuaian antara `movieId` dan `title` guna memastikan bahwa setiap entitas film direpresentasikan secara konsisten.

**Alasan:**  
Ketidaksesuaian antara `movieId` dan `title` dapat menyebabkan duplikasi informasi film dan memengaruhi kualitas rekomendasi yang dihasilkan.

---

### 2. Menggabungkan Dataset Movies dan Ratings

Dataset **Movies** dan **Ratings** digabungkan menjadi satu dataset menggunakan kolom `movieId`. Dataset hasil penggabungan disimpan dalam variabel `movies_ratings`.

**Alasan:**  
Penggabungan dataset diperlukan agar informasi film (judul dan genre) dan interaksi pengguna (rating) berada dalam satu struktur data, sehingga mempermudah proses preprocessing, eksplorasi data, dan analisis lanjutan.

---

### 3. Normalisasi Teks pada Fitur `title` dan `genres`

Dilakukan normalisasi teks pada kolom `title` dan `genres`, antara lain:
- Menghapus karakter khusus yang tidak diperlukan
- Menyeragamkan format penulisan

**Alasan:**  
Normalisasi teks bertujuan untuk meningkatkan konsistensi data, sehingga proses lanjutan seperti *content-based filtering*, *collaborative filtering*, NLP, atau perhitungan similarity dapat dilakukan dengan lebih akurat.

---

### 4. Memproses Tahun pada Fitur `title`

Beberapa film teridentifikasi memiliki **tahun rilis yang tidak valid atau tidak tersedia** pada judul film. Untuk mengatasi hal tersebut, dilakukan perbaikan dengan melakukan pencarian referensi eksternal (misalnya melalui sumber daring yang kredibel).

**Alasan:**  
Informasi tahun rilis penting untuk analisis temporal dan eksplorasi data. Kesalahan pada tahun rilis dapat menghasilkan insight yang keliru dan memengaruhi analisis lanjutan.

---

### 5. Memproses Fitur `genres`

Tahap ini mencakup:
- Melengkapi genre pada film yang tidak memiliki informasi genre
- Menyeragamkan penulisan genre yang tidak konsisten
- Menghapus atau memperbaiki genre yang tidak valid

**Alasan:**  
Genre merupakan salah satu atribut penting dalam analisis film. Konsistensi dan kelengkapan genre membantu dalam eksplorasi data dan memungkinkan pengembangan metode rekomendasi berbasis konten di masa depan.

---

### 6. Encoding User dan Item

Fitur `userId` dan `movieId` yang bersifat kategorikal diubah menjadi indeks numerik berurutan melalui proses encoding:
- `userId` → `user`
- `movieId` → `movie`

Mapping ini juga disimpan untuk keperluan inferensi dan interpretasi hasil rekomendasi.

**Alasan:**  
Model *Neural Collaborative Filtering* menggunakan embedding yang hanya dapat menerima input berupa indeks numerik. Encoding juga meningkatkan efisiensi komputasi dan penggunaan memori.

---

### 7. Pengacakan Data (Shuffling)

Dataset diacak sebelum proses pembagian data latih dan validasi dengan menggunakan *random seed* tetap.

**Alasan:**  
Pengacakan data bertujuan untuk menghindari bias akibat urutan data tertentu serta memastikan distribusi data yang lebih merata pada data latih dan validasi.

---

### 8. Pembagian Data Train dan Validasi

Dataset dibagi menjadi:
- **80% data training**
- **20% data validasi**

Pembagian dilakukan setelah proses pengacakan data.

**Alasan:**  
Data training digunakan untuk melatih model, sedangkan data validasi digunakan untuk mengevaluasi performa model dan memantau potensi overfitting.

## Modeling

Pada tahap **Modeling**, dikembangkan sebuah sistem rekomendasi film berbasis **Collaborative Filtering menggunakan pendekatan Neural Collaborative Filtering (NCF)**. Pendekatan ini dipilih karena mampu mempelajari representasi laten pengguna dan item secara non-linear, sehingga lebih adaptif terhadap data dengan tingkat sparsity yang tinggi dibandingkan metode collaborative filtering tradisional.

### Arsitektur Model

Model yang dibangun mengadopsi konsep *matrix factorization* yang diperluas dengan *neural network*, dengan komponen utama sebagai berikut:

- **User Embedding Layer**  
  Merepresentasikan setiap pengguna ke dalam vektor laten berdimensi tetap. Vektor ini menangkap preferensi tersembunyi pengguna terhadap film.

- **Movie Embedding Layer**  
  Merepresentasikan setiap film ke dalam vektor laten dengan dimensi yang sama seperti user embedding, sehingga interaksi keduanya dapat dihitung secara langsung.

- **Bias Term (User & Movie Bias)**  
  Digunakan untuk menangkap kecenderungan global pengguna (misalnya pengguna yang cenderung memberi rating tinggi) dan popularitas film.

- **Dot Product Interaction**  
  Interaksi antara user embedding dan movie embedding dihitung menggunakan operasi dot product untuk merepresentasikan tingkat kesesuaian pengguna terhadap film.

- **Activation Function (Sigmoid)**  
  Digunakan pada output layer untuk menghasilkan skor prediksi dalam rentang [0, 1], yang kemudian dapat dipetakan kembali ke skala rating.

Secara matematis, prediksi rating dirumuskan sebagai:

Prediksi rating pengguna *u* terhadap item *i* dirumuskan sebagai:

$\hat{r}_{u,i} = \mathbf{p}_u \cdot \mathbf{q}_i + b_u + b_i$

di mana:
- $\mathbf{p}_u$ adalah vektor embedding pengguna  
- $\mathbf{q}_i$ adalah vektor embedding film  
- $b_u$ dan $b_i$ masing-masing adalah bias pengguna dan film

---

### Proses Training Model

Model dilatih menggunakan:
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (dengan ReduceLROnPlateau)
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metric**: Root Mean Squared Error (RMSE)
- **Regularization**: L2 regularization pada embedding


Training dilakukan menggunakan *mini-batch gradient descent* dengan pipeline `tf.data.Dataset` untuk efisiensi komputasi. Selain itu, digunakan *Early Stopping* untuk menghentikan training ketika performa validasi tidak lagi meningkat.

---

### Top-N Recommendation

Setelah model terlatih, sistem digunakan untuk menghasilkan **Top-N Recommendation** untuk setiap pengguna dengan langkah berikut:

1. Mengambil seluruh film yang **belum pernah ditonton** oleh pengguna pada data training.
2. Melakukan prediksi skor preferensi pengguna terhadap film-film tersebut menggunakan model NCF.
3. Mengurutkan film berdasarkan skor prediksi secara menurun.
4. Mengambil **Top-N film** dengan skor tertinggi sebagai rekomendasi.

Output Top-N recommendation ini merepresentasikan daftar film yang paling relevan untuk pengguna tertentu berdasarkan pola preferensi yang dipelajari oleh model.

*Sampel hasil output rekomendasi berdasarkan user_id = 203*
| movieId | title | pred_score |
|---------|-------|------------|
| 246 | Hoop Dreams (1994) | 4.574862 |
| 1198 | Raiders of the Lost Ark (Indiana Jones and the...) | 4.556180 |
| 4973 | Amelie (Fabuleux destin d'Amélie Poulain, Le) ... | 4.502178 |
| 50 | Usual Suspects, The (1995) | 4.498779 |
| 356 | Forrest Gump (1994) | 4.494165 |
| 4011 | Snatch (2000) | 4.438409 |
| 475 | In the Name of the Father (1993) | 4.437773 |
| 177593 | Three Billboards Outside Ebbing, Missouri (2017) | 4.435908 |
| 58 | Postman, The (Postino, Il) (1994) | 4.433905 |
| 58559 | Dark Knight, The (2008) | 4.418793 |


---

### Kelebihan Pendekatan Neural Collaborative Filtering

- **Mampu menangkap pola non-linear**  
  NCF dapat mempelajari hubungan kompleks antara pengguna dan film yang tidak dapat ditangkap oleh metode matrix factorization konvensional.

- **Efektif pada data sparse**  
  Representasi embedding memungkinkan model melakukan generalisasi meskipun sebagian besar interaksi user–item tidak tersedia.

- **Fleksibel dan mudah dikembangkan**  
  Arsitektur model dapat diperluas dengan menambahkan layer neural network lain atau mengombinasikannya dengan fitur konten.

---

### Keterbatasan Pendekatan

- **Masalah cold-start pada item**  
  Model masih mengalami kesulitan dalam merekomendasikan film baru yang memiliki sangat sedikit atau tidak memiliki interaksi.

- **Ketergantungan pada data historis**  
  Model hanya dapat merekomendasikan berdasarkan pola yang pernah terjadi, sehingga kurang adaptif terhadap perubahan preferensi yang sangat cepat.

- **Waktu evaluasi yang relatif tinggi**  
  Proses inferensi untuk evaluasi Top-N recommendation membutuhkan prediksi terhadap banyak item, sehingga memerlukan optimasi lebih lanjut untuk skala besar.

Secara keseluruhan, pendekatan Neural Collaborative Filtering yang digunakan pada proyek ini mampu menghasilkan rekomendasi Top-N yang relevan dan menjadi solusi yang efektif untuk permasalahan sistem rekomendasi film berbasis data interaksi pengguna.

---

## Evaluation

Tahap **Evaluation** bertujuan untuk mengukur sejauh mana sistem rekomendasi yang dibangun mampu menghasilkan rekomendasi film yang relevan bagi pengguna. Mengingat tujuan utama sistem adalah menghasilkan **Top-N recommendation**, maka metrik evaluasi yang digunakan berfokus pada kualitas peringkat (*ranking-based metrics*), bukan hanya akurasi prediksi rating.

### Metrik Evaluasi yang Digunakan

Evaluasi dilakukan menggunakan beberapa metrik berikut:

---

### 1. Precision@K

**Precision@K** mengukur proporsi item relevan dari total item yang direkomendasikan pada posisi Top-K.

Precision@K didefinisikan sebagai:

$\text{Precision@K} = \frac{|\text{Relevant Items} \cap \text{Recommended Items@K}|}{K}$

**Interpretasi:**  
Nilai Precision@K menunjukkan seberapa tepat rekomendasi yang diberikan. Semakin tinggi nilainya, semakin besar proporsi film yang benar-benar relevan di antara rekomendasi yang ditampilkan.

---

### 2. Recall@K

**Recall@K** mengukur seberapa banyak item relevan yang berhasil direkomendasikan oleh sistem dari seluruh item relevan yang tersedia.

Recall@K didefinisikan sebagai:

$\text{Recall@K} = \frac{|\text{Relevant Items} \cap \text{Recommended Items@K}|}{|\text{Relevant Items}|}$

**Interpretasi:**  
Recall@K menggambarkan kemampuan sistem dalam menemukan film-film yang memang relevan bagi pengguna.

---

### 3. Hit Rate@K

**Hit Rate@K** mengukur apakah sistem berhasil merekomendasikan setidaknya satu item relevan pada Top-K recommendation.

Hit Rate@K didefinisikan sebagai:

HitRate@K = I(|Relevant Items ∩ Recommended Items@K| > 0)

I(x) = 1 jika x benar, dan 0 jika x salah.

**Interpretasi:**  
Metrik ini sangat relevan untuk pengalaman pengguna, karena satu rekomendasi yang tepat sering kali sudah cukup untuk menarik perhatian pengguna.

---

### 4. Normalized Discounted Cumulative Gain (NDCG@K)

**NDCG@K** mempertimbangkan posisi item relevan dalam daftar rekomendasi, dengan memberikan bobot lebih besar pada item relevan yang muncul di peringkat atas.

DCG@K didefinisikan sebagai:

$\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}$

NDCG@K didefinisikan sebagai:

$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$

di mana:
- $rel_i$ adalah nilai relevansi item pada posisi ke-$i$
- $\text{IDCG@K}$ adalah nilai DCG ideal

**Interpretasi:**  
NDCG@K menilai tidak hanya apakah item relevan direkomendasikan, tetapi juga apakah item tersebut ditempatkan pada posisi yang tepat.


---

### Hasil Evaluasi Model

Berdasarkan proses evaluasi terhadap data validasi, diperoleh hasil sebagai berikut:

- **Precision@K**: 0.0482  
- **Recall@K**: 0.0408  
- **HitRate@K**: 0.3465  
- **NDCG@K**: 0.0563  

---

### Analisis Hasil Evaluasi

- Nilai **Precision@K** yang relatif rendah menunjukkan bahwa hanya sebagian kecil dari rekomendasi Top-K yang benar-benar relevan. Hal ini umum terjadi pada dataset dengan tingkat sparsity yang sangat tinggi.
  
- **Recall@K** yang rendah mengindikasikan bahwa tidak semua film relevan berhasil direkomendasikan, terutama karena jumlah item relevan per pengguna relatif sedikit.

- **HitRate@K** yang cukup tinggi (±34%) menunjukkan bahwa dalam banyak kasus, sistem berhasil merekomendasikan setidaknya satu film relevan kepada pengguna. Dari sudut pandang pengalaman pengguna, hal ini merupakan indikasi yang positif.

- **NDCG@K** yang rendah namun konsisten dengan Precision dan Recall menunjukkan bahwa meskipun item relevan berhasil direkomendasikan, posisinya belum selalu berada di peringkat teratas.

---

### Kesimpulan Evaluasi

Secara keseluruhan, hasil evaluasi menunjukkan bahwa model **Neural Collaborative Filtering** mampu memberikan rekomendasi yang relevan meskipun dihadapkan pada tantangan utama berupa **tingkat sparsity data yang sangat tinggi (±98%)** dan **dominasi cold-start pada item**.  

Metrik berbasis ranking yang digunakan telah sesuai dengan tujuan sistem rekomendasi Top-N dan memberikan gambaran yang komprehensif mengenai performa model dari berbagai aspek relevansi dan peringkat.

Hasil ini juga membuka peluang pengembangan lanjutan, seperti:
- Integrasi pendekatan *hybrid recommendation*
- Penerapan negative sampling yang lebih optimal
- Optimasi inferensi untuk evaluasi berskala besar

