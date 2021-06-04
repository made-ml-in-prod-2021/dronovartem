from datetime import timedelta
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

model_dir = Variable.get('model_dir')
model_path = model_dir + '/model.pkl'

with DAG(
        "03_predict_dag",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv",
    )
    wait_for_model = FileSensor(
        task_id='wait-for-model',
        poke_interval=10,
        retries=5,
        filepath=f"{model_path}",
    )
    predict = DockerOperator(
        image="dronovartem/airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }} "
                f"--model-dir {model_dir}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=["/d/Education/made2020/2sem/PROD_ML/airflow_ml_dags/data:/data"]
    )

    [wait_for_data, wait_for_model] >> predict