FROM python:3.10.8

ENV POETRY_VERSION=1.8.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
COPY uk_plate_recognition/pyproject.toml ./uk_plate_recognition/pyproject.toml
COPY uk_plate_recognition/src ./uk_plate_recognition/src
COPY uk_plate_recognition/checkpoints/custom_data_cnn_v2/plate_cnn_final.pt ./uk_plate_recognition/checkpoints/custom_data_cnn_v2/plate_cnn_final.pt

RUN poetry install --without dev --no-root

COPY main.py ./main.py
COPY car_parking ./car_parking

EXPOSE 80

CMD ["poetry", "run", "python", "main.py"]
