FROM python:3.7-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY configs/ ./configs/
COPY models/model.pkl ./models/model.pkl
COPY src/entities/ ./src/entities/
COPY src/features/ ./src/features/
COPY src/models/ ./src/models/
COPY src/app.py ./src/app.py

CMD ["python", "-m", "src.app"]