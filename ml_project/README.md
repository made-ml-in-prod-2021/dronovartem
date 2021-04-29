Task description:
https://data.mail.ru/blog/topic/view/18519/

Installation:

~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Usage (from ml_project directory):
how to run train pipeline:
~~~
python -m src.train_pipeline --config `path_to_cfg`
~~~
how to run predict pipeline:
~~~
python -m src.predict_pipeline --data `path_to_dataset` --model `path_to_model` --output `output_file_path`
~~~

how to run tests:
~~~
pytest tests
~~~

Project structure is based on:
- [DS cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
- [project example](https://github.com/made-ml-in-prod-2021/ml_project_example)