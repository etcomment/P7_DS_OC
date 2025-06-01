# test_api.py
import unittest
from main import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_index(self):
        res = self.app.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn('message', res.get_json())

    def test_prediction_valid(self):
        payload = {
            "features": [0.5]*20  # Simule une entrée valide avec 20 features
        }
        res = self.app.post('/predict', json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertIn('score', res.get_json())

    def test_prediction_invalid(self):
        payload = {
            "bad_input": [0.5]*20
        }
        res = self.app.post('/predict', json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.get_json())

if __name__ == '__main__':
    unittest.main()
