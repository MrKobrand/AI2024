import pandas as pd
import numpy as np
from src.models import run_kmeans_pipeline, run_automl_pipeline
from unittest.mock import patch


def test_run_kmeans_pipeline():
    df = pd.DataFrame({
        'COMPANY': ['Company A', 'Company B'],
        'VALUE': [1.0, 2.0]
    })

    with patch('src.models.compute_cosine_distances') as mock_compute:
        mock_compute.return_value = pd.DataFrame([[0, 1], [1, 0]])
        result = run_kmeans_pipeline(df)

        # Проверяем, что результат содержит 2 компании
        assert len(result) == 2

        # Проверяем, что столбец 'Cluster' присутствует
        assert 'Cluster' in result.columns


def test_run_automl_pipeline():
    df = pd.DataFrame({
        'COMPANY': [f'Company {i}' for i in range(1, 31)],
        'VALUE': [i for i in range(1, 31)]
    })

    with patch('src.models.compute_cosine_distances') as mock_compute:
        mock_compute.return_value = pd.DataFrame(np.random.rand(30, 10))

        result = run_automl_pipeline(df)

        # Проверяем, что результат содержит 30 компаний
        assert len(result) == 30

        # Проверяем, что столбец 'Cluster' присутствует
        assert 'Cluster' in result.columns
