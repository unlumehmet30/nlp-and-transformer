# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:52:31 2025

@author: mhmtn
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Veri hazırlığı
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)
word_index = imdb.get_word_index()
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(encoded_review):
    return " ".join(reverse_word_index.get(i, "?") for i in encoded_review)

random_indices = np.random.choice(len(x_train), size=3, replace=False)
for i in random_indices:
    print(f"yorum: {decode_review(x_train[i])}")
    print(f"etiket: {y_train[i]}")

# Transformer block
class Transformer_block(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate=0.3, l2_value=1e-4):
        super(Transformer_block, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * 4, activation="relu", kernel_regularizer=regularizers.l2(l2_value)),
            layers.Dense(embed_size, kernel_regularizer=regularizers.l2(l2_value))
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        attention = self.attention(x, x)
        x = self.norm1(x + self.dropout1(attention, training=training))
        feed_forward = self.feed_forward(x)
        return self.norm2(x + self.dropout2(feed_forward, training=training))

# Transformer model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate, l2_value=1e-4):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=embed_size, 
                                          embeddings_regularizer=regularizers.l2(l2_value))
        self.transformer_blocks = [Transformer_block(embed_size, heads, dropout_rate, l2_value) for _ in range(num_layers)]
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_value))

    def call(self, x, training=None):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.global_avg_pooling(x)
        x = self.dropout(x, training=training)
        return self.fc(x)

# Modeli oluştur
num_layers = 4
embed_size = 64
num_heads = 4
input_dim = 10000
output_dim = 1
dropout_rate = 0.5
maxlen = 100
l2_value = 1e-4

model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate, l2_value)
dummy_input = tf.random.uniform((1, maxlen), minval=0, maxval=input_dim, dtype=tf.int32)
model(dummy_input)

# Modeli derle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Model özeti
model.summary()

# Eğit
history = model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test))

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],marker="*",label="t_loss")
plt.plot(history.history["val_loss"],marker="*",label="val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training and validation loss")
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],marker="*",label="t_accuracy")
plt.plot(history.history["val_accuracy"],marker="*",label="val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("training and validation accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def classify_review(model, review, word_index, maxlen):
    encoded_text = [word_index.get(word, 0) for word in review.lower().split()]
    padded_text = pad_sequences([encoded_text], maxlen=maxlen)
    pred = model.predict(padded_text)
    if pred[0][0] >= 0.6:
        prediction = "positive"
    else:
        prediction = "negative"
    return prediction, pred[0][0]

word_index = imdb.get_word_index()
review = input("write a review: ")

label, score = classify_review(model, review, word_index, maxlen)
print(f"label: {label}\nscore: {score}")
