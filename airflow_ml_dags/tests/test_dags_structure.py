import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


DOWNLOAD_DAG_STRUCTURE = {
    "docker-airflow-download": [],
}

TRAIN_DAG_STRUCTURE = {
    "wait-for-data": ["docker-airflow-preprocess"],
    "wait-for-target": ["docker-airflow-preprocess"],
    "docker-airflow-preprocess": ["docker-airflow-split"],
    "docker-airflow-split": ["docker-airflow-train"],
    "docker-airflow-train": ["docker-airflow-validate"],
    "docker-airflow-validate": [],
}

PREDICT_DAG_STRUCTURE = {
    "wait-for-data": ["docker-airflow-predict"],
    "wait-for-model": ["docker-airflow-predict"],
    "docker-airflow-predict": [],
}


@pytest.fixture(scope='session')
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_import_dags(dag_bag):
    assert len(dag_bag.import_errors) == 0, (
        f"Errors were found while importing: {dag_bag.import_errors}"
    )


@pytest.mark.parametrize(
    ("test_dag_name", "dag_template", "is_correct_structure"),
    [
        ("01_download_dag", DOWNLOAD_DAG_STRUCTURE, True),
        ("02_train_pipeline_dag", TRAIN_DAG_STRUCTURE, True),
        ("03_predict_dag", PREDICT_DAG_STRUCTURE, True),
    ]
)
def test_dag_structure(dag_bag, test_dag_name, dag_template, is_correct_structure):
    assert _dag_has_correct_structure(
        dag_bag, test_dag_name, dag_template) == is_correct_structure


def _dag_has_correct_structure(dag_bag, dag_name, template):
    try:
        dag = dag_bag.dags[dag_name]
        assert len(dag.tasks) == len(template)
        for task_id, downstream_list in template.items():
            assert dag.has_task(task_id)
            task = dag.get_task(task_id)
            assert task.downstream_task_ids == set(downstream_list)
    except AssertionError:
        return False
    return True