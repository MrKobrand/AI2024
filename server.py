from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
from src.models import run_automl_pipeline

app = FastAPI()


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Проверяем, что файл имеет расширение .csv
    if not file.filename.endswith('.csv'):
        return {"error": "File type not supported. Please upload a .csv file."}

    # Читаем CSV файл в DataFrame
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Проверяем, что DataFrame имеет нужную структуру
    if 'COMPANY' not in df.columns:
        return {"error": "Invalid CSV structure. 'COMPANY' column is required."}

    # Применяем функцию run_automl_pipeline к загруженному DataFrame
    result_df = run_automl_pipeline(df)

    # Возвращаем результат в виде словаря
    return result_df.to_dict(orient="records")

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
