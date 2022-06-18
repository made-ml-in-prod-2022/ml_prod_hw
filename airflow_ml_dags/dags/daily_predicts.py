import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
        "daily_predicts",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(2),
) as dag:

    start = DummyOperator(task_id="start")

    predict = DockerOperator(
        image="predict-daily",
        command="python predict.py --data /data/raw/{{ ds }}/data.csv --model {{ var.value.model_path }} --predict /data/predictions/{{ ds }}/predictions.csv",
        network_mode="bridge",
        task_id="predict-daily",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/dedperded/experiments/airflow-examples/data/", target="/data", type='bind')]
    )
    
    start >> predict
