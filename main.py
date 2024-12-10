from src.data import load_data
from src.models import run_kmeans_pipeline, run_automl_pipeline


def main():
    df = load_data('data/raw/Medical Ratio.csv')
    df_pipelines = run_kmeans_pipeline(df)
    print(df_pipelines)

    df_automl = run_automl_pipeline(df)
    print(df_automl)


if __name__ == '__main__':
    main()
