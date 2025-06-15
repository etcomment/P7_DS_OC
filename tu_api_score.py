import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import json
import sys
import os

# Ajouter le répertoire parent au path pour importer l'API
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestApiScore(unittest.TestCase):

    def setUp(self):
        """Configuration avant chaque test"""
        # Mock des données de test
        self.mock_df_data = pd.DataFrame({
            'SK_ID_CURR': [100001, 100002, 100003],
            'index': [0, 1, 2],
            'EXT_SOURCE_1': [0.5, 0.7, 0.3],
            'EXT_SOURCE_2': [0.6, 0.8, 0.4],
            'AMT_CREDIT': [150000, 200000, 100000],
            'AMT_INCOME_TOTAL': [50000, 60000, 40000]
        })

        # Mock du modèle
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # [proba_classe_0, proba_classe_1]

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_app_initialization(self, mock_pickle_load, mock_file, mock_read_csv,
                                mock_gdown, mock_requests):
        """Test l'initialisation de l'application"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import de l'API (déclenche l'initialisation)
        import api_score

        # Vérifications
        mock_requests.assert_called_once()
        mock_gdown.assert_called_once()
        mock_pickle_load.assert_called_once()
        self.assertIsNotNone(api_score.app)

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_index_route(self, mock_pickle_load, mock_file, mock_read_csv,
                         mock_gdown, mock_requests):
        """Test la route index"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('message', data)
            self.assertIn('API de scoring bancaire', data['message'])

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_success(self, mock_pickle_load, mock_file, mock_read_csv,
                             mock_gdown, mock_requests):
        """Test une prédiction réussie"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': '100001'})

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('id_client', data)
            self.assertIn('score', data)
            self.assertEqual(data['id_client'], 100001)
            self.assertEqual(data['score'], 0.7)  # Correspond à la proba classe 1

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_missing_id_client(self, mock_pickle_load, mock_file, mock_read_csv,
                                       mock_gdown, mock_requests):
        """Test prédiction sans id_client"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={})

            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertIn('id_client est requis', data['error'])

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_client_not_found(self, mock_pickle_load, mock_file, mock_read_csv,
                                      mock_gdown, mock_requests):
        """Test prédiction avec un client inexistant"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': '999999'})

            self.assertEqual(response.status_code, 404)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertIn('Aucun client trouvé', data['error'])

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_invalid_id_format(self, mock_pickle_load, mock_file, mock_read_csv,
                                       mock_gdown, mock_requests):
        """Test prédiction avec un format d'ID invalide"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': 'abc123'})

            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_model_error(self, mock_pickle_load, mock_file, mock_read_csv,
                                 mock_gdown, mock_requests):
        """Test gestion d'erreur du modèle"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]

        # Mock du modèle qui lève une exception
        mock_model_error = MagicMock()
        mock_model_error.predict_proba.side_effect = Exception("Erreur modèle")
        mock_pickle_load.return_value = mock_model_error

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': '100001'})

            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_predict_get_method_not_allowed(self, mock_pickle_load, mock_file, mock_read_csv,
                                            mock_gdown, mock_requests):
        """Test que la route predict n'accepte que POST"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]
        mock_pickle_load.return_value = self.mock_model

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.get('/predict')
            self.assertEqual(response.status_code, 405)  # Method Not Allowed

    @patch('api_score.requests.get')
    def test_download_failure(self, mock_requests):
        """Test la gestion d'échec de téléchargement"""
        # Simuler un échec de téléchargement
        mock_requests.side_effect = Exception("Erreur de téléchargement")

        with self.assertRaises(Exception):
            # Tenter d'importer le module (ce qui déclenche le téléchargement)
            import api_score

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_multiple_predictions(self, mock_pickle_load, mock_file, mock_read_csv,
                                  mock_gdown, mock_requests):
        """Test plusieurs prédictions consécutives"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_read_csv.side_effect = [self.mock_df_data, self.mock_df_data]

        # Mock du modèle avec différentes réponses
        mock_model_multi = MagicMock()
        mock_model_multi.predict_proba.side_effect = [
            np.array([[0.3, 0.7]]),  # Premier client
            np.array([[0.6, 0.4]]),  # Deuxième client
        ]
        mock_pickle_load.return_value = mock_model_multi

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            # Premier client
            response1 = client.post('/predict', data={'id_client': '100001'})
            self.assertEqual(response1.status_code, 200)
            data1 = json.loads(response1.data)
            self.assertEqual(data1['score'], 0.7)

            # Deuxième client
            response2 = client.post('/predict', data={'id_client': '100002'})
            self.assertEqual(response2.status_code, 200)
            data2 = json.loads(response2.data)
            self.assertEqual(data2['score'], 0.4)


class TestApiScoreEdgeCases(unittest.TestCase):
    """Tests pour les cas limites"""

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_empty_dataframe(self, mock_pickle_load, mock_file, mock_read_csv,
                             mock_gdown, mock_requests):
        """Test avec un DataFrame vide"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        # DataFrame vide
        empty_df = pd.DataFrame(columns=['SK_ID_CURR', 'index'])
        mock_read_csv.side_effect = [empty_df, empty_df]
        mock_pickle_load.return_value = MagicMock()

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': '100001'})
            self.assertEqual(response.status_code, 404)

    @patch('api_score.requests.get')
    @patch('api_score.gdown.download')
    @patch('api_score.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('api_score.pickle.load')
    def test_extreme_probability_values(self, mock_pickle_load, mock_file, mock_read_csv,
                                        mock_gdown, mock_requests):
        """Test avec des valeurs de probabilité extrêmes"""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.content = b'fake_model_data'
        mock_requests.return_value = mock_response

        mock_df = pd.DataFrame({
            'SK_ID_CURR': [100001],
            'index': [0],
            'feature1': [1.0]
        })
        mock_read_csv.side_effect = [mock_df, mock_df]

        # Modèle avec probabilité extrême
        mock_model_extreme = MagicMock()
        mock_model_extreme.predict_proba.return_value = np.array([[0.0, 1.0]])
        mock_pickle_load.return_value = mock_model_extreme

        # Import et test
        import api_score
        with api_score.app.test_client() as client:
            response = client.post('/predict', data={'id_client': '100001'})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['score'], 1.0)


def create_test_suite():
    """Créer une suite de tests complète"""
    suite = unittest.TestSuite()

    # Ajouter tous les tests
    suite.addTest(unittest.makeSuite(TestApiScore))
    suite.addTest(unittest.makeSuite(TestApiScoreEdgeCases))

    return suite


if __name__ == '__main__':
    # Configuration pour les tests
    import sys
    import os

    # Ajouter le répertoire de l'API au path
    api_dir = os.path.dirname(os.path.abspath(__file__))
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)

    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Afficher le résumé
    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ DES TESTS")
    print(f"{'=' * 60}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")

    if result.failures:
        print(f"\nÉchecs détaillés:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")

    if result.errors:
        print(f"\nErreurs détaillées:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")

    # Code de sortie
    sys.exit(0 if result.wasSuccessful() else 1)