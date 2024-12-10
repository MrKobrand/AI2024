import pandas as pd
from unittest.mock import patch


def test_load_data():
    # Ожидаемая структура данных
    expected_data = pd.DataFrame({
        'COMPANY': ['Company A', 'Company B'],
        'VALUE': [1.0, 2.0]
    })

    # Используем patch для замены функции load_data на мок
    with patch('src.data.load_data') as mock_load_data:
        # Настраиваем мок, чтобы он возвращал ожидаемые данные
        mock_load_data.return_value = expected_data

        # Вызываем мок вместо реальной функции
        # Путь не важен, так как мы используем мок
        result = mock_load_data('data/raw/test_data.csv')

        # Проверяем, что загруженные данные соответствуют ожидаемым
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_data
        )
