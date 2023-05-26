import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

with open('/home/rozatk/PycharmProjects/API_Recommend_Products/cliente.json') as f:
    data = json.load(f)

products = set()
for celular in data['resultado']:
    for product in data['resultado'][celular]:
        products.add(product)
product_to_id = {product: id for id, product in enumerate(products)}
num_products = len(products)

user_item_interactions = {}
for celular in data['resultado']:
    user_item_interactions[celular] = np.zeros(num_products)
    for product in data['resultado'][celular]:
        user_item_interactions[celular][product_to_id[product]] = 1.0

X_train, X_test, y_train, y_test = train_test_split(list(user_item_interactions.values()),
                                                    list(
                                                        user_item_interactions.values()),
                                                    test_size=0.2)

user_item_matrix = tf.keras.layers.Input(shape=(num_products,))

user_embedding = tf.keras.layers.Embedding(input_dim=len(user_item_interactions),
                                           output_dim=50)(user_item_matrix)
user_embedding = tf.keras.layers.Flatten()(user_embedding)

item_embedding = tf.keras.layers.Embedding(input_dim=num_products,
                                           output_dim=50)(user_item_matrix)
item_embedding = tf.keras.layers.Flatten()(item_embedding)

dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, item_embedding])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_products,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(num_products, activation='softmax')
])

# Compilando o modelo
model.compile(loss='mse', optimizer='adam')

# Treinando o modelo
history = model.fit(np.array(X_train), np.array(y_train), epochs=300,
                    batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

# Avaliando o modelo
y_pred = model.predict(np.array(X_test))
mae = mean_absolute_error(np.array(y_test), y_pred)
rmse = np.sqrt(mean_squared_error(np.array(y_test), y_pred))
print(f"Erro Absoluto Médio: {mae}")
print(f"Raiz do erro quadrático médio: {rmse}")

user_interactions = user_item_interactions['id_client']

predictions = model.predict(np.array([user_interactions]))

sorted_predictions = sorted(
    enumerate(predictions[0]), key=lambda x: x[1], reverse=True)


# Salvando os pesos do modelo treinado
model.save_weights('recommender.h5')
