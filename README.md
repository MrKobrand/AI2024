# Анализ схожести патентного массива

## Описание
Данный проект направлен на анализ схожести технологий на основе патентной информации с использованием методов машинного обучения.

## Структура проекта
- `data/` - Данные (входные и выходные)
- `notebooks/` - Jupyter ноутбуки для экспериментов
- `src/` - Исходный код проекта
- `tests/` - Тесты
- `requirements.txt` - Список зависимостей
- `main.py` - Главный файл для запуска проекта

## Установка
1. Клонируйте репозиторий.
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск приложения
Запустите основной файл:
```bash
python main.py
```

## Запуск тестов
Запустите файл теста, например:
```bash
pytest ./tests/test_data.py
```

Или все тесты вместе:
```bash
pytest ./tests
```

## Запуск сервера
Выполните следующую команду:
```bash
python server.py
```