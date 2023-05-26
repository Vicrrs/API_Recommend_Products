import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import FastAPI

# Load data from JSON file
with open('/home/victorroza/PycharmProjects/Api_recomenda/Files/cliente.json') as f:
    data = json.load(f)

# Create dictionary with product IDs
products = {product for celular in data['resultado'] for product in data['resultado'][celular]}
product_to_id = {product: id for id, product in enumerate(products)}
num_products = len(products)

# Create dictionary with user-item interactions
user_item_interactions = {celular: np.zeros(num_products) for celular in data['resultado']}
for celular, celular_data in data['resultado'].items():
    for product in celular_data:
        user_item_interactions[celular][product_to_id.get(product, -1)] = 1.0

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

# Create FastAPI app
app = FastAPI()

# Define endpoint for recommendations
@app.get("/recomendacoes/{celular}")
async def recomendacoes(celular: str):
    user_interactions = np.zeros(num_products)
    if celular in data['resultado']:
        for product in data['resultado'][celular]:
            product_id = product_to_id.get(product, -1)
            if product_id != -1:
                user_interactions[product_id] = 1.0

    predictions = model.predict(np.array([user_interactions]))

    sorted_predictions = sorted(
        enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

    produtos_recomendados = [list(products)[sorted_predictions[i][0]] for i in range(10)]

    return {"recommended products": produtos_recomendados}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)