from src.visualization import plot_dendrogram
from unittest.mock import patch


def test_plot_dendrogram():
    # Создаем фиктивные данные для теста
    linkage_matrix = [[0, 1, 0.5, 2]]  # Примерный формат для дендрограммы

    with patch('matplotlib.pyplot.show') as mock_show:
        plot_dendrogram(linkage_matrix)

        # Проверяем, что функция show была вызвана
        mock_show.assert_called_once()
