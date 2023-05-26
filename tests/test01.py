import unittest
from unittest.mock import patch
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from sistema_recomendacao.recommendation_model import *

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = FastAPI()
        self.cliente_json = {
            "resultado": {
                "celular1": ["produto1", "produto2"],
                "celular2": ["produto3", "produto4"]
            }
        }
        self.num_products = 2
        self.product_to_id = {"produto1": 0, "produto2": 1}
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_products,)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(self.num_products, activation='softmax')
        ])
        self.model.load_weights(r'path_to_model_weights')

    def test_recomendacoes(self):
        with patch('builtins.open', return_value=json.dumps(self.cliente_json)):
            with patch.object(self.model, 'predict', return_value=np.array([[0.5, 0.5]])):
                with patch('builtins.sorted', return_value=[(1, 0.6), (0, 0.4)]):
                    with patch.object(list, '__getitem__', side_effect=["produto2", "produto1"]):
                        response = self.app.get("/recomendacoes/celular1")
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), {"produtos_recomendados": ["produto2", "produto1"]})

if __name__ == '__main__':
    unittest.main()