import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from recommendation_system.recommendation_model import *

app = FastAPI()

num_products = 0
with open(r'path_to_json_file') as f:
    data = json.load(f)

products = set()
for celular in data['resultado']:
    for product in data['resultado'][celular]:
        products.add(product)
product_to_id = {product: id for id, product in enumerate(products)}
num_products = len(products)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_products,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(num_products, activation='softmax')
])
model.load_weights(r'path_to_model_weights')


@app.get("/recomendacoes/{celular}")
async def recomendacoes(celular: str):
    user_interactions = np.zeros(num_products)
    if celular in data['resultado']:
        for product in data['resultado'][celular]:
            if product in product_to_id:
                user_interactions[product_to_id[product]] = 1.0

    predictions = model.predict(np.array([user_interactions]))

    sorted_predictions = sorted(
        enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

    produtos_recomendados = [list(products)[sorted_predictions[i][0]] for i in range(10)]

    return {"recommended products": produtos_recomendados}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)