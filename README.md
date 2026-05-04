# Car Parking API

A FastAPI backend for a car parking management system.

The application accepts car images, detects a number plate, tracks parking sessions, calculates parking cost, manages users and admin actions, and stores data in PostgreSQL.

## Project background

This repository is based on an older team project called **Car Parking**. The original version was developed as a team project. This version reflects my later personal work on refactoring, cleanup, maintenance, Docker/local development setup, and backend improvements.

The project idea and original feature set came from the team project. The current repository is a later refactored personal version, not the untouched original team submission.

## Main features

- User registration, login, email confirmation, password reset, and logout
- JWT access and refresh token flow
- Role-based access for users and admins
- Admin actions for users, cars, roles, tariffs, and bans
- Car number plate detection from uploaded images
- Parking entry and exit flow
- Parking duration and cost calculation
- Payment confirmation flow
- Parking availability checks
- User parking history
- CSV export for parking history
- Email notifications for account and parking events
- PostgreSQL database with Alembic migrations
- Redis support for local development
- Swagger API documentation
- Docker Compose setup for local PostgreSQL and Redis

## Tech stack

- Python 3.10.8
- FastAPI
- SQLAlchemy
- Alembic
- PostgreSQL
- Redis
- Pydantic v1
- Poetry
- Uvicorn
- OpenCV
- TensorFlow / Keras
- Pillow
- NumPy
- FastAPI Mail
- Docker / Docker Compose

## Project structure

```text
Car-Parking-Project/
├── car_parking/
│   ├── migrations/          # Alembic migrations
│   ├── src/
│   │   ├── conf/            # settings, constants, config helpers
│   │   ├── database/        # SQLAlchemy engine, session, models
│   │   ├── models/          # ML/OCR model files
│   │   ├── repository/      # database operations
│   │   ├── routes/          # FastAPI routes
│   │   ├── schemas/         # Pydantic request/response models
│   │   ├── services/        # auth, email, OCR, roles, domain services
│   │   ├── templates/       # email templates
│   │   └── utils/           # shared helpers
│   ├── alembic.ini
│   └── .example.env
├── docker-compose.yml       # local PostgreSQL and Redis
├── Dockerfile               # API image build
├── main.py                  # app entrypoint
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Local setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Car-Parking-Project
```

### 2. Install dependencies

This project uses Poetry as the main dependency workflow.

```bash
poetry install
```

### 3. Create environment file

Create a local environment file based on the example file.

Depending on your current config path, use the same location expected by `car_parking/src/conf/config.py`.


Do not commit real `.env` files.

## Running locally

### 1. Start PostgreSQL and Redis

```bash
docker compose up -d
```

This starts:

- PostgreSQL on `localhost:5433`
- Redis on `localhost:6380`

Check containers:

```bash
docker ps
```

### 2. Run database migrations

```bash
cd car_parking
poetry run alembic upgrade head
cd ..
```

### 3. Start the API

```bash
poetry run python main.py
```

The API should be available at:

```text
http://localhost:80
```

Swagger documentation:

```text
http://localhost:80/docs
```

Health checks:

```text
GET /health
GET /health/db
```

## Docker usage

The current `docker-compose.yml` is used for local PostgreSQL and Redis.

To reset local containers and volumes:

```bash
docker compose down -v --remove-orphans
docker compose up -d
```

To build the API image:

```bash
docker build -t car-parking-api .
```

If you run the API as a Docker container, do not use `localhost` for database and Redis hostnames inside the container. Use Docker service names instead:

```env
SQLALCHEMY_DATABASE_URL=postgresql://car_parking_user:car_parking_password@car_parking_postgres:5432/car_parking_dev_db
REDIS_HOST=car_parking_redis
REDIS_PORT=6379
REDIS_URL=redis://car_parking_redis:6379/0
```

Then run the API container on the same Docker network as the Compose services.

## API overview

Main API groups:

```text
/auth      authentication, email confirmation, password reset, logout
/users     current user profile and parking information
/parking   parking entry, exit, payment confirmation, availability
/admin     admin operations for users, cars, tariffs, CSV reports
/health    app and database health checks
```

## Database

The project uses PostgreSQL with SQLAlchemy and Alembic.

Common migration command:

```bash
cd car_parking
poetry run alembic upgrade head
cd ..
```

The app also seeds required default parking data on startup, such as default tariffs and parking count data.

## Refactoring notes

This version includes later cleanup and refactoring work, including:

- cleaner local PostgreSQL and Redis setup
- Poetry-based dependency workflow
- safer environment configuration
- improved health checks
- clearer database session handling
- refactored parking flow and parking-count logic
- fixed parking availability calculation
- cleaner route paths for parking entry and exit
- improved auth and refresh-token flow
- logout now invalidates both access-token usage and stored refresh token
- cleaner route-level error handling
- improved admin route structure
- cleaned schemas and response models
- cleaned email service naming and template usage
- repository cleanup and removal of old compatibility wrappers

## Notes and limitations

- This is a backend/API project. It is mainly tested through Swagger docs.
- The OCR and number-plate detection quality depends on the uploaded image.
- This is a portfolio/refactored project, not a production parking system.
- The project still keeps some original structure and naming from the older team project where it makes sense.

## Credits

The original Car Parking project began as a team project.

Original team members:

- Kostiantyn Pereimybida (Team Leader)
- Vladyslav Kyryllov (Scrum Master)
- Dmytro Kruhlov (Developer)
- Michael Ivanov (Developer)
- Natalia Semeniuk (Developer)

This repository/version was later revisited, refactored, and maintained by Kostiantyn Pereimybida as a personal continuation of the original project.

## License

MIT License.