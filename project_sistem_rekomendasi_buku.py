# -*- coding: utf-8 -*-
"""project_sistem_rekomendasi_buku jadi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kmNqJUK-haFg_l8pHRjjX4Svt1kutCZN
"""

from google.colab import drive
drive.mount("drive")

"""# Import library yang dibutuhkan"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""# Load Data"""

base_dir = "/content/drive/MyDrive/datasets/book/"
books = pd.read_csv(base_dir+"Books.csv")
ratings = pd.read_csv(base_dir+"Ratings.csv")
users = pd.read_csv(base_dir+"Users.csv")

"""# Exploratory Data Analysis

## Deskripsi Variabel
Dataset terbagi menjadi 3 kategori, *ratings*, *user profile*, dan *books*.

- ISBN merupakan identitas unik suatu buku, satu ISBN untuk satu buku.
- book_title merupakan judul buku
- book_author merupakan pengarang
- year_of_publication merupakan tahun terbit buku
- publisher merupakan penerbit buku
- image_url_s merupakan gambar dari buku yang berukuran kecil
- image_url_m merupakan gambar dari buku yang berukuran medium
- image_url_l merupakan gambar dari buku yang berukuran besar
- user_id merupakan id unik pengguna
- location merupakan lokasi pengguna
- age merupakan umur pengguna
- rating merupakan *rating* dari pengguna
"""

books

"""## Mengubah teks pada kolom menjadi huruf kecil sekaligus mengubah "-" menjadi "_" agar mempermudah proses pemanggilan."""

books.columns = books.columns.str.lower()
books.columns = books.columns.str.replace("-","_")

ratings.columns = ratings.columns.str.lower()
ratings.columns = ratings.columns.str.replace("-","_")

users.columns = users.columns.str.lower()
users.columns = users.columns.str.replace("-","_")

# books
# ratings
users

"""## Melihat jenis tipe data *dataframe books*"""

books.info()

"""*Data Frame* di atas memiliki 271360 baris data dengan keseluruhannya bertipe data object. Jika dilihat dari keseluruhan total data yang ada, terdapat missing values pada kolom book_author (271359 baris), publisher, dan image_url_l. Namun mari kita tangani nanti.

## Memastikan tiap satu ISBN mencakup satu buku
"""

books["isbn"].duplicated().sum()

"""## judul buku unik """

print(f"Banyak data buku yang unik berdasarkan judul : {len(books['book_title'].unique())}")
print(f"Banyak data buku yang unik berdasarkan ISBN : {len(books['isbn'].unique())}")

"""Berdasarkan data di atas, didapatkan beberapa informasi sebagai berikut:  
1. Jumlah seluruh data buku yang unik yakni 271360, hal ini karena menunjukkan bahwa satu ISBN hanya untuk satu buku.
2. Terdapat beberapa judul yang sama dengan ISBN yang berbeda, hal ini kemungkinan menunjukkan bahwa data buku dengan judul yang sama merupakan dua entitas yang berbeda. Seperti, kemungkinan buku berjudul x memiliki sequelnya.
"""

books[books["book_title"].duplicated()].sample(5,axis=0)

"""Data di atas merupakan list buku yang memiliki judul yang duplikat. Selanjutnya mari kita lihat salah satu data pada judul buku di atas."""

books[books["book_title"] == "El Ladron De Cuerpos"]

"""Sudah terlihat, bahwa judul buku sama yang memiliki ISBN berbeda merupakan dua entitas yang berbeda. Jika dilihat pada tahun publikasi di atas, buku dengan judul "El Ladron De Cuerpos" dipublikasi pada tahun yang berbeda. Berdasarkan hal tersebut, dapat disimpulkan bahwa buku dengan judul tersebut memiliki sequel lanjutannya.

## Banyak data user yang unik
"""

users

print(f"Banyak data user unik : {len(users['user_id'].unique())}")

users.info()

"""## Banyak data ratings"""

ratings

print(f"Banyak data rating :{len(ratings)}")
print(f"Jumlah buku yang telah diberi rating : {len(ratings['isbn'].unique())}")
print(f"Jumlah user yang memberikan rating : {len(ratings['user_id'].unique())}")

ratings.info()

"""## Rentang rating"""

ratings.describe().round(3)

"""Rating buku berentang antara 0 - 10 (terendah ke tertinggi)

# Data Preprocessing
"""

print(f"Jumlah seluruh data buku berdasarkan ISBN : {len(books['isbn'].unique())}")
print(f"Jumlah seluruh data buku berdasarkan judul buku : {len(books['book_title'].unique())}")
print(f"Jumlah seluruh users : {len(users['user_id'].unique())}")
print(f"Jumlah seluruh rating : {len(ratings)}")

"""## Menggabungkan data ratings dengan judul buku"""

all_book = ratings
all_book

all_book = pd.merge(all_book, books[["isbn","book_title"]], on="isbn", how="left")
all_book

"""# Data Preparation

## Menangani *Missing Values*
"""

all_book

all_book.isna().sum()

"""Berdasarkan jumlah data rating yang ada (1 juta lebih), *missing value* berjumlah 100 ribu data. Sehingga tidak apa jika kita menghapus *missing value* pada kolom book_title"""

all_book_clean = all_book.dropna()
all_book_clean

"""Setelah menghapus baris data yang mengandung *missing value*, kini data berjumlah 1031136 baris data.

## Membuat dataframe baru yang berisi isbn dan judul buku
"""

preparation = all_book_clean
preparation

book_title, isbn = preparation["book_title"].tolist(), preparation["isbn"].tolist()
print(f"Jumlah data judul buku : {len(book_title)}")
print(f"Jumlah data isbn: {len(isbn)}")

book_new = pd.DataFrame({
    "isbn" : isbn,
    "title" : book_title
})
book_new

"""## Encode kolom user_id dan isbn"""

df = ratings
df

isbn_id = df["isbn"].unique().tolist()
user_id = df["user_id"].unique().tolist()

isbn_encoded = {key:values for values, key in enumerate(isbn_id)}
isbn_decoded = {key:values for key, values in enumerate(isbn_id)}

user_encoded = {key:values for values, key in enumerate(user_id)}
user_decoded = {key:values for key, values in enumerate(user_id)}

"""## Mapping terhadap dataframe"""

df["user_encoded"] = df["user_id"].map(user_encoded)
df["isbn_encoded"] = df["isbn"].map(isbn_encoded)
df

num_users = len(user_encoded)
num_books = len(isbn_encoded)

print(f"Banyak user : {num_users}")
print(f"Banyak buku : {num_books}")

"""## Mengubah tipe data kolom book_rating menjadi float"""

df["book_rating"] = df["book_rating"].values.astype(np.float64)
df

"""## Membagi data latih dan data validasi

### Mengacak dataframe
"""

df = df.sample(frac=1, random_state=99)
df

"""### Membagi data 85% data latih dan 15% data validasi, sekaligus normalisasi kolom rating berkisar 0-1"""

x = df[["user_encoded","isbn_encoded"]]
min = df["book_rating"].min()
max = df["book_rating"].max()
y = df["book_rating"].apply(lambda x:(x-min) / (max-min) )

split = int(0.85 * df.shape[0])
X_train, X_val, Y_train, Y_val = (
    x[:split],
    x[split:],
    y[:split],
    y[split:]
)

"""# Modeling"""

class RecommenderBook(tf.keras.Model):
  def __init__(self, num_users, num_books, embedding_size, **kwargs):
    super(RecommenderBook, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_books=num_books
    self.embedding_size=embedding_size
    self.user_embedding = tf.keras.layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=tf.keras.regularizers.l2(0.00001)
    )
    self.user_bias = tf.keras.layers.Embedding(num_users, 1)
    self.book_embedding = tf.keras.layers.Embedding(
        num_books,
        embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=tf.keras.regularizers.l2(0.00001)
      )
    self.book_bias=tf.keras.layers.Embedding(num_books, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0]) 
    book_vector = self.book_embedding(inputs[:, 1])
    book_bias = self.book_bias(inputs[:, 1]) 
 
    dot_user_book = tf.tensordot(user_vector, book_vector, 2) 
 
    x = dot_user_book + user_bias + book_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

model = RecommenderBook(num_users, num_books, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, epochs=25)

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.title("RMSE metrics plot")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend(["train","test"])
plt.savefig("evaluation.png", dpi=75)
plt.show()

"""# Result

## Memperoleh hasil rekomendasi
"""

book_df = book_new
df = pd.read_csv("/content/drive/MyDrive/datasets/book/Ratings.csv")

user_id = df["User-ID"].sample(1).iloc[0]
readed_book_by_user = df[df["User-ID"] == user_id]

book_not_readed = book_df[~book_df["isbn"].isin(readed_book_by_user["ISBN"].values)]["isbn"]
book_not_readed = list(
    set(book_not_readed).intersection(set(isbn_encoded.keys()))
)
book_not_readed = [[isbn_encoded.get(x)] for x in book_not_readed]
user_encoder = user_encoded.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_readed), book_not_readed)
)

ratings = model.predict(user_book_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_id = [
    isbn_decoded.get(book_not_readed[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Book with high ratings from user')
print('----' * 8)
 
top_book_user = (
    readed_book_by_user.sort_values(
        by = 'Book-Rating',
        ascending=False
    )
    .head(5)
    .ISBN.values
)
 
book_df_rows = book_df[book_df['isbn'].isin(top_book_user)].drop_duplicates()
for row in book_df_rows.itertuples():
    print(row.isbn, ':', row.title)
 
print('----' * 8)
print('Top 10 Book recommendation')
print('----' * 8)
 
recommended_book = book_df[book_df['isbn'].isin(recommended_book_id)].drop_duplicates()
for row in recommended_book.itertuples():
    print(row.isbn, ':', row.title)