# Car Parking ANPR

Car Parking ANPR is a parking API built around a local UK number-plate recognition pipeline.

The project accepts a car image, detects the number plate, reads the plate text, and uses that result to manage parking entry, exit, payment, and parking history. The backend is included so the ML component can be used in a realistic workflow instead of staying as a standalone notebook or script.

The main ML problem is fixed-format UK plate recognition:

```text
LLDDLLL
example: AB12CDE
```

## Navigation

- [Why I Built It](#why-i-built-it)
- [ML Pipeline Overview](#ml-pipeline-overview)
- [Data and Training Utilities](#data-and-training-utilities)
- [Model Evaluation](#model-evaluation)
- [Application Flow](#application-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Using the ANPR Flow](#using-the-anpr-flow)
- [Running Tests](#running-tests)
- [Docker Notes](#docker-notes)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)

## Why I Built It

This repository started from an older team project called **Car Parking**. I later revisited it as a personal project and rebuilt the parts that interested me most:

- a local ANPR runtime package
- a custom CNN recogniser for UK plates
- a detector-to-recogniser inference pipeline
- data cleaning and metadata tools for plate crops
- a FastAPI app that uses the ML pipeline in real parking routes

The backend matters here because it gives the model a useful context: users upload car images, the system reads the plate, and the result drives parking logic.

## ML Pipeline Overview

The runtime pipeline handles full car images, not only cropped plates.

```text
uploaded car image
-> temporary image file
-> local plate detector
-> best plate crop
-> custom CNN recogniser
-> 7-position decoder
-> UK plate format validation
-> parking workflow
```

### 1. Plate Detection

The full-image detector uses `open-image-models` with the model:

```text
yolo-v9-t-256-license-plate-end2end
```

The detector returns bounding boxes for number plates. The app keeps the highest-confidence detection, adds a small amount of padding, and crops the plate before recognition.

Relevant code:

```text
uk_plate_recognition/src/anpr/detection/open_image_models_detector.py
uk_plate_recognition/src/anpr/inference/open_image_models_pipeline.py
```

### 2. Plate Recognition

The recogniser is a custom PyTorch CNN for fixed-format UK plates.

Instead of treating the whole plate as one class, the model predicts each character position separately:

```text
position 0 -> letter, 26 classes
position 1 -> letter, 26 classes
position 2 -> digit, 10 classes
position 3 -> digit, 10 classes
position 4 -> letter, 26 classes
position 5 -> letter, 26 classes
position 6 -> letter, 26 classes
```

This makes the output match the known UK plate structure and keeps the model small enough for local inference.

Relevant code:

```text
uk_plate_recognition/src/anpr/models/plate_cnn.py
uk_plate_recognition/src/anpr/data/encoders.py
uk_plate_recognition/src/anpr/inference/decode.py
```

### 3. Decoding and Validation

The raw model outputs are decoded into a plate string. The prediction is then checked using:

- exact fixed UK format: `LLDDLLL`
- average confidence across all positions
- per-position confidence thresholds

The API runtime currently uses:

```text
minimum overall confidence: 0.95
minimum position confidence: 0.80
```

If the prediction does not pass these checks, the parking route treats the plate as not found.

Relevant code:

```text
uk_plate_recognition/src/anpr/inference/result.py
uk_plate_recognition/src/anpr/validation/uk_plate.py
car_parking/src/services/anpr_service.py
car_parking/src/services/plate_reader.py
```

## Data and Training Utilities

The `uk_plate_recognition` package contains tools for building and testing the recogniser:

- filename-based label parsing for UK plates
- metadata CSV generation
- train/validation/test split assignment
- PyTorch dataset loading
- conservative image augmentation for plate crops
- multi-head cross-entropy loss
- full-plate and per-position evaluation metrics
- scripts for scraping, registering, cleaning, filtering, and deduplicating plate crops

The training code is organized as reusable modules rather than one large training script. This keeps the package easier to test and reuse from notebooks or future CLI scripts.

Important modules:

```text
uk_plate_recognition/src/anpr/data/
uk_plate_recognition/src/anpr/training/
uk_plate_recognition/src/anpr/evaluation/
uk_plate_recognition/src/anpr/scraping/
```


## Model Evaluation

The final recogniser checkpoint was tested on a cleaned cropped-plate test split. This evaluation measures the custom CNN recogniser on already-cropped plate images, not the full end-to-end parking flow. In the full application, final ANPR quality also depends on the detector crop and the quality of the uploaded car image.

Final test run:

```text
final_clean_36x124_10ep
```

| Metric | Value |
|---|---:|
| Test samples | 1,231 |
| Test loss | 0.0517 |
| Full-plate accuracy | 94.96% |
| Regex-valid rate | 100.00% |
| Average confidence | 99.17% |

Per-position accuracy:

| Position | Expected character type | Accuracy |
|---:|---|---:|
| 0 | Letter | 99.68% |
| 1 | Letter | 99.59% |
| 2 | Digit | 99.51% |
| 3 | Digit | 98.70% |
| 4 | Letter | 98.62% |
| 5 | Letter | 97.24% |
| 6 | Letter | 98.70% |

The strongest result is the full-plate accuracy of **94.96%** on the cleaned cropped-plate test set. The lower score at position 5 also gives a useful direction for future dataset balancing and error analysis.

## Application Flow

The FastAPI app uses the ANPR package in the parking routes.

Typical entry flow:

```text
POST /api/parking/enter
-> upload car image
-> read license plate
-> check whether the car is banned
-> create parking session
-> update occupied space count
-> send parking entry email for registered users
```

Typical exit flow:

```text
POST /api/parking/exit
-> upload car image
-> read license plate
-> find active parking session
-> calculate duration and cost
-> send invoice email for registered users
```

The backend also includes user accounts, JWT authentication, admin actions, tariffs, parking availability, email templates, and CSV export. Those features support the parking workflow, but the main technical focus of this repository is the ML pipeline and how it is integrated into the app.

## Tech Stack

### ML and Computer Vision

- PyTorch
- OpenCV
- NumPy
- Albumentations
- Open Image Models

### Backend

- FastAPI
- SQLAlchemy
- Alembic
- Pydantic v1
- PostgreSQL
- Redis for health/infrastructure checks
- FastAPI Mail

### Tooling

- Poetry
- Docker / Docker Compose
- Pytest

## Project Structure

```text
Car-Parking/
|-- car_parking/
|   |-- migrations/                 # Alembic migrations
|   `-- src/
|       |-- routes/                 # FastAPI routes
|       |-- repository/             # database operations
|       |-- services/               # auth, email, ANPR app integration
|       |-- schemas/                # Pydantic schemas
|       |-- database/               # SQLAlchemy models and sessions
|       |-- templates/              # email templates
|       `-- utils/                  # shared backend helpers
|
|-- uk_plate_recognition/
|   |-- src/anpr/
|   |   |-- data/                   # labels, metadata, dataset, transforms
|   |   |-- detection/              # full-image plate detector wrapper
|   |   |-- models/                 # custom CNN recogniser
|   |   |-- inference/              # decoding and runtime pipelines
|   |   |-- training/               # loss and train/eval loops
|   |   |-- evaluation/             # recognition metrics
|   |   |-- validation/             # UK plate validation
|   |   `-- scraping/               # dataset collection/cleaning utilities
|   |-- checkpoints/                # trained recogniser checkpoints
|   |-- data/                       # local metadata and image data
|   `-- tests/                      # ANPR tests
|
|-- docker-compose.yml              # local PostgreSQL and Redis
|-- Dockerfile                      # API image build
|-- main.py                         # FastAPI entrypoint
|-- pyproject.toml
`-- README.md
```

## Setup

Install the project dependencies:

```bash
poetry install
```

Create a `.env` file from `.example.env` and fill in the required values.

For local development with Docker Compose, the database settings normally match:

```env
POSTGRES_DB=car_parking_dev_db
POSTGRES_USER=car_parking_user
POSTGRES_PASSWORD=car_parking_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
SQLALCHEMY_DATABASE_URL=postgresql://car_parking_user:car_parking_password@localhost:5433/car_parking_dev_db
REDIS_URL=redis://localhost:6380/0
```

Start local services:

```bash
docker compose up -d
```

Run database migrations:

```bash
cd car_parking
poetry run alembic upgrade head
cd ..
```

Start the API:

```bash
poetry run python main.py
```

Default local URLs:

```text
API:     http://localhost:80
Swagger: http://localhost:80/docs
```

## Using the ANPR Flow

The easiest way to inspect the project is through Swagger:

```text
POST /api/parking/enter
POST /api/parking/exit
```

Both routes accept an uploaded image file. The app saves the upload temporarily, runs the full-image ANPR pipeline, then removes the temporary file.

The runtime recogniser checkpoint used by the API is:

```text
uk_plate_recognition/checkpoints/custom_data_cnn_v2/plate_cnn_final.pt
```

## Running Tests

The ANPR package has focused tests for the ML/data code:

```bash
cd uk_plate_recognition
poetry install --with dev
poetry run pytest
```

The tests cover label encoding/decoding, UK plate validation, metrics, image preprocessing, loss validation, model output shape, prediction helpers, and train/evaluation loops.

## Docker Notes

The provided `docker-compose.yml` runs PostgreSQL and Redis for local development.

The API image can be built with:

```bash
docker build -t car-parking-api .
```

The Dockerfile copies the runtime ANPR package and the final recogniser checkpoint into the image.

If the API runs inside Docker, use Docker service names instead of `localhost`:

```env
SQLALCHEMY_DATABASE_URL=postgresql://car_parking_user:car_parking_password@car_parking_postgres:5432/car_parking_dev_db
REDIS_URL=redis://car_parking_redis:6379/0
```

## Limitations

- The recogniser is designed for fixed-format UK plates in `LLDDLLL` format.
- Recognition quality depends strongly on image quality and detector crop quality.
- The backend is a portfolio/demo application, not a production parking system.
- Training utilities are present, but this repository does not expose a polished one-command training CLI.
- The reported model evaluation is for the cleaned cropped-plate recogniser test split. Full end-to-end ANPR accuracy can be lower if the uploaded image is low quality or the detector crop is poor.

## Future Improvements

- Add a single reproducible training command.
- Add a reproducible benchmark script for the final checkpoint.
- Add sample images and expected outputs for quick local inspection.
- Improve error reporting for low-confidence plate predictions.
- Add more integration tests around the image upload parking routes.

## Credits

The original Car Parking project began as a team project.

Original team members:

- Kostiantyn Pereimybida (Team Leader)
- Vladyslav Kyryllov (Scrum Master)
- Dmytro Kruhlov (Developer)
- Michael Ivanov (Developer)
- Natalia Semeniuk (Developer)

This version was later revisited, refactored, and maintained by Kostiantyn Pereimybida as a personal continuation with a stronger focus on ANPR and ML engineering.

## License

MIT License.
