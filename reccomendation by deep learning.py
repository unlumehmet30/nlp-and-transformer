import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Veri
user_ids = np.array([0,1,2,3,4,0,1,2,3,4])
item_ids = np.array([0,1,2,3,4,1,2,3,4,5])
ratings  = np.array([5,3,4,2,1,5,4,3,2,1])

# Eğitim/Test bölmesi
user_train, user_test, item_train, item_test, ratings_train, ratings_test = train_test_split(
    user_ids, item_ids, ratings, test_size=0.2, random_state=42)

# Model oluşturma fonksiyonu
def create_model(num_user, num_item, emb_dim):
    # Giriş katmanları
    user_input = Input(shape=(1,), name="user")
    item_input = Input(shape=(1,), name="item")

    # Embedding katmanları
    user_embedding = Embedding(input_dim=num_user, output_dim=emb_dim, name="user_embedding")(user_input)
    item_embedding = Embedding(input_dim=num_item, output_dim=emb_dim, name="item_embedding")(item_input)

    # Vektörleştirme
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)

    # Dot product
    dot_product = Dot(axes=1)([user_vec, item_vec])

    # Çıkış
    output = Dense(1)(dot_product)

    # Model tanımı
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

# Modeli oluştur
num_user = 5
num_item = 6
emb_dim = 8

model = create_model(num_user, num_item, emb_dim)

# Eğit
model.fit([user_train, item_train], ratings_train, epochs=100, verbose=1, validation_split=0.1)
loss=model.evaluate([user_test,item_test],ratings_test)
print(f"test:{loss}")
user_id=np.array([0])
item_id=np.array([1])
prediction=model.predict([user_id,item_id])
print(f"by user-{user_id[0]} prediction of item-{item_id[0]} is {prediction[0][0]}")






























































