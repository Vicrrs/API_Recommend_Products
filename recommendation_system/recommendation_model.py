import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data from JSON file
with open('path_to_json_file') as f:
    data = json.load(f)

# Create dictionary with product IDs
products = {product for celular in data['resultado'] for product in data['resultado'][celular]}
product_to_id = {product: id for id, product in enumerate(products)}
num_products = len(products)

# Create dictionary with user-item interactions
user_item_interactions = {celular: np.zeros(num_products) for celular in data['resultado']}
for celular, product_list in data['resultado'].items():
    for product in product_list:
        user_item_interactions[celular][product_to_id[product]] = 1.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    list(user_item_interactions.values()),
    list(user_item_interactions.values()),
    test_size=0.2
)

# Define input layer for model
user_item_matrix = tf.keras.layers.Input(shape=(num_products,))

# Add user embedding layer
user_embedding = tf.keras.layers.Embedding(input_dim=len(user_item_interactions), output_dim=50)(user_item_matrix)
user_embedding = tf.keras.layers.Flatten()(user_embedding)

# Add item embedding layer
item_embedding = tf.keras.layers.Embedding(input_dim=num_products, output_dim=50)(user_item_matrix)
item_embedding = tf.keras.layers.Flatten()(item_embedding)

# Compute dot product of user and item embeddings
dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, item_embedding])

# Define final model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_products,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(num_products, activation='softmax')
])

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
history = model.fit(
    np.array(X_train),
    np.array(y_train),
    epochs=10,
    batch_size=32,
    validation_data=(np.array(X_test), np.array(y_test))
)

# Evaluate model
y_pred = model.predict(np.array(X_test))
mae = mean_absolute_error(np.array(y_test), y_pred)
rmse = np.sqrt(mean_squared_error(np.array(y_test), y_pred))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Obtendo o vetor de interações do usuário com os produtos
user_interactions = user_item_interactions['id_cliente']

# Fazendo previsões de interação do usuário com todos os produtos
predictions = model.predict(np.array([user_interactions]))

# Ordenando as previsões em ordem decrescente
sorted_predictions = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

# Imprimindo os 10 principais produtos recomendados
for i, (product_id, score) in enumerate(sorted_predictions[:10]):
    product_name = list(products)[product_id]
    print(f"Produtos {i+1}: {product_name}")