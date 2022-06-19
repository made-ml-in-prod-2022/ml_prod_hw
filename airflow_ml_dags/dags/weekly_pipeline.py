import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}


def _wait_for_split_data():
    return os.path.exists("/opt/airflow/data/split/{{ ds }}/x_train.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/x_test.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/y_train.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/y_test.csv")


with DAG(
        "weekly_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(14),
) as dag:

    start = DummyOperator(task_id="start")

    data_prep = DockerOperator(
        image="prep-data",
        command="python preprocess.py --data /data/raw/{{ ds }} --processed /data/processed/{{ ds }}/train_data.csv",
        network_mode="bridge",
        task_id="preprocess-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/dedperded/experiments/airflow-examples/data/", target="/data", type='bind')]
    )

    data_split = DockerOperator(
        image="split-data",
        command="python split.py --data /data/processed/{{ ds }}/train_data.csv --split /data/split/{{ ds }}",
        network_mode="bridge",
        task_id="split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/dedperded/experiments/airflow-examples/data/", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="train-model",
        command="python train.py --split /data/split/{{ ds }} --model /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/dedperded/experiments/airflow-examples/data/", target="/data", type='bind')]
    )

    val = DockerOperator(
        image="validate-model",
        command="python validation.py --split /data/split/{{ ds }} --model /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="validate-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/dedperded/experiments/airflow-examples/data/", target="/data", type='bind')]
    )

    wait_for_train_data = FileSensor(
        task_id="waiting_for_train_data",
        filepath="split/{{ ds }}/x_train.csv",
        fs_conn_id="train_data",
        poke_interval=10
    )

    wait_for_train_target = FileSensor(
        task_id="wait_for_train_target",
        filepath="split/{{ ds }}/y_train.csv",
        fs_conn_id="train_data",
        poke_interval=10
    )

    wait_for_test_data = FileSensor(
        task_id="wait_for_test_data",
        filepath="split/{{ ds }}/x_test.csv",
        fs_conn_id="train_data",
        poke_interval=10
    )

    wait_for_test_target = FileSensor(
        task_id="wait_for_test_target",
        filepath="split/{{ ds }}/y_test.csv",
        fs_conn_id="train_data",
        poke_interval=10
    )

    start >> data_prep >> data_split >> wait_for_train_data >> wait_for_train_target >> train >> wait_for_test_data >> wait_for_test_target >> val

