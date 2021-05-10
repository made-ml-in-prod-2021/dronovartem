Task description:
https://data.mail.ru/blog/topic/view/18768/

# Usage
# python:
(from online_inference dir):
how to run app:
~~~
python -m src.app
~~~
how to run request script:
~~~
python -m src.make_request
~~~
tests:
~~~
pytest tests
~~~

# Docker:
Build command:
~~~
docker build -t dronovartem/inference:v1 .
~~~
Run command:
~~~
docker run -p 8000:8000 dronovartem/inference:v1
~~~
Push command:
~~~
docker push dronovartem/inference:v1
~~~
Pull command:
~~~
docker pull dronovartem/inference:v1
~~~


Project structure is based on:
- [DS cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
- [inference examples](https://github.com/made-ml-in-prod-2021/inference_examples)
- [Fast API docs](https://fastapi.tiangolo.com/)
- [Docker recommendations](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
