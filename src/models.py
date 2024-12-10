import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.features import compute_cosine_distances


def run_kmeans_pipeline(df):
    df1 = df.drop(columns='COMPANY')

    # Расчет косинусного сходства
    co_dist = compute_cosine_distances(df1)

    # Определение конвейера для KMeans
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Стандартизация данных
        ('pca', PCA(n_components=2)),  # Уменьшение размерности
        ('kmeans', KMeans(n_clusters=2))  # Кластеризация
    ])

    # Кластеризация с использованием Pipeline
    labels = pipeline.fit_predict(co_dist)

    # Создание DataFrame с результатами
    elbow_result = pd.DataFrame({'Name': df['COMPANY'], 'Cluster': labels})

    return elbow_result


def run_automl_pipeline(df):
    df1 = df.drop(columns='COMPANY')

    # Расчет косинусного сходства
    co_dist = compute_cosine_distances(df1)

    # Определение конвейера для KMeans
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Стандартизация данных
        ('pca', PCA(n_components=2)),  # Уменьшение размерности
        ('kmeans', KMeans(random_state=42))  # Кластеризация
    ])

    # Определение параметров для GridSearch
    param_grid = {
        # Различное количество кластеров
        'kmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    # Создание пользовательского оценщика для silhouette_score
    def silhouette_scorer(estimator, X):
        labels = estimator.predict(X)
        if len(set(labels)) > 1:  # Убедимся, что есть более одного кластера
            return silhouette_score(X, labels)
        else:
            return -1  # Если только один кластер, возвращаем -1

    # Инициализация GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=silhouette_scorer
    )

    # Обучение GridSearchCV на данных
    grid_search.fit(co_dist)

    # Вывод лучших параметров
    print("Лучшие параметры:", grid_search.best_params_)

    # Получение лучшей модели
    best_model = grid_search.best_estimator_

    # Применение лучшей модели для кластеризации
    labels = best_model.predict(co_dist)

    # Создание DataFrame с результатами
    elbow_result = pd.DataFrame({'Name': df['COMPANY'], 'Cluster': labels})

    # Отображение результатов
    return elbow_result
