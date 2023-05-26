import unittest
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import FastAPI

class TestRecommendationSystem(unittest.TestCase):
    
    def setUp(self):
        # Load data from JSON file
        with open('path_to_json_file') as f:
            self.data = json.load(f)

        # Create dictionary with product IDs
        self.products = {product for celular in self.data['resultado'] for product in self.data['resultado'][celular]}
        self.product_to_id = {product: id for id, product in enumerate(self.products)}
        self.num_products = len(self.products)

        # Create dictionary with user-item interactions
        self.user_item_interactions = {celular: np.zeros(self.num_products) for celular in self.data['resultado']}
        for celular, celular_data in self.data['resultado'].items():
            for product in celular_data:
                self.user_item_interactions[celular][self.product_to_id.get(product, -1)] = 1.0

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            list(self.user_item_interactions.values()),
            list(self.user_item_interactions.values()),
            test_size=0.2
        )

        # Define input layer for model
        self.user_item_matrix = tf.keras.layers.Input(shape=(self.num_products,))

        # Add user embedding layer
        self.user_embedding = tf.keras.layers.Embedding(input_dim=len(self.user_item_interactions), output_dim=50)(self.user_item_matrix)
        self.user_embedding = tf.keras.layers.Flatten()(self.user_embedding)

        # Add item embedding layer
        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.num_products, output_dim=50)(self.user_item_matrix)
        self.item_embedding = tf.keras.layers.Flatten()(self.item_embedding)

        # Compute dot product of user and item embeddings
        self.dot_product = tf.keras.layers.Dot(axes=1)([self.user_embedding, self.item_embedding])

        # Define final model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_products,)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(self.num_products, activation='softmax')
        ])

        # Compile model
        self.model.compile(loss='mse', optimizer='adam')

        # Train model
        self.history = self.model.fit(
            np.array(self.X_train),
            np.array(self.y_train),
            epochs=10,
            batch_size=32,
            validation_data=(np.array(self.X_test), np.array(self.y_test))
        )

        # Create FastAPI app
        self.app = FastAPI()

    def test_recommendations(self):
        # Define endpoint for recommendations
        @self.app.get("/recomendacoes/{celular}")
        async def recomendacoes(celular: str):
            user_interactions = np.zeros(self.num_products)
            if celular in self.data['resultado']:
                for product in self.data['resultado'][celular]:
                    product_id = self.product_to_id.get(product, -1)
                    if product_id != -1:
                        user_interactions[product_id] = 1.0

            predictions = self.model.predict(np.array([user_interactions]))

            sorted_predictions = sorted(
                enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

            produtos_recomendados = [list(self.products)[sorted_predictions[i][0]] for i in range(10)]

            return {"recommended products": produtos_recomendados}

        # Test endpoint for recommendations
        with self.app.test_client() as client:
            response = client.get('/recomendacoes/celular1')
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), dict)
            self.assertIn('recommended products', response.json())

if __name__ == '__main__':
    unittest.main()