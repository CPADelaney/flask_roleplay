# Lore System

A robust and scalable roleplay lore management system built with Flask and modern async Python.

## Features

- **Efficient Resource Management**
  - Connection pooling
  - Rate limiting
  - Circuit breaker pattern
  - Memory-aware caching

- **Comprehensive Monitoring**
  - Prometheus metrics integration
  - OpenTelemetry tracing
  - Alert management
  - Performance tracking

- **Robust Validation**
  - Schema validation
  - Custom validation rules
  - Parallel validation
  - Error recovery

- **Database Optimization**
  - Connection pooling
  - Query optimization
  - Transaction management
  - Health checks

## Architecture

```
lore/
├── core/
│   ├── resource_manager.py    # Resource management and caching
│   ├── monitoring.py         # System monitoring and metrics
│   ├── validation.py         # Data validation
│   └── database_access.py    # Database operations
├── api/
│   ├── routes.py            # API endpoints
│   └── middleware.py        # Request/response middleware
├── models/
│   ├── npc.py              # NPC data models
│   ├── knowledge.py        # Knowledge management
│   └── relationships.py    # Relationship tracking
└── utils/
    ├── logging.py          # Logging configuration
    └── error_handler.py    # Error handling
```

## Prerequisites

- Python 3.9+
- MySQL 8.0+
- Redis 6.0+
- Prometheus (optional)
- Jaeger (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lore-system.git
cd lore-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Optional: disable pip's version check to speed up installs
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip install -r requirements.txt
```

The first run downloads about **1&nbsp;GB** of transformer weights for the
sentence encoder.

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
flask db upgrade
```

## Configuration

The system can be configured through environment variables or a `.env` file:

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=user
DB_PASSWORD=password
DB_NAME=lore_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=6831

# Security
JWT_SECRET=your-secret-key
API_RATE_LIMIT=100
```

## Usage

1. Start the development server:
```bash
flask run
```

2. Access the API:
```bash
curl http://localhost:5000/api/v1/health
```

3. Monitor the system:
```bash
# Prometheus metrics
curl http://localhost:5000/metrics

# Health check
curl http://localhost:5000/api/v1/health
```

## API Documentation

The API documentation is available at `/docs` when running the server.

### Key Endpoints

- `GET /api/v1/npcs` - List all NPCs
- `POST /api/v1/npcs` - Create a new NPC
- `GET /api/v1/npcs/{id}` - Get NPC details
- `PUT /api/v1/npcs/{id}` - Update NPC
- `DELETE /api/v1/npcs/{id}` - Delete NPC

## Monitoring

### Metrics

The system exposes the following Prometheus metrics:

- `lore_requests_total` - Total number of requests
- `lore_request_duration_seconds` - Request latency
- `lore_errors_total` - Total number of errors
- `lore_memory_usage_bytes` - Memory usage
- `lore_cpu_usage_percent` - CPU usage
- `lore_cache_size_bytes` - Cache size
- `lore_active_connections` - Active connections

### Tracing

Distributed tracing is implemented using OpenTelemetry and Jaeger:

1. Start Jaeger:
```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.22
```

2. Access the Jaeger UI at `http://localhost:16686`

## Development

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run the formatters:
```bash
black .
isort .
```

Run the linters:
```bash
flake8
mypy .
```

### Testing

Run the test suite:
```bash
pytest
```

Generate coverage report:
```bash
pytest --cov=lore tests/
```

## Deployment

### Docker

Build the Docker image:
```bash
docker build -t lore-system .
```

Run the container:
```bash
docker run -d \
  -p 5000:5000 \
  -e DB_HOST=host.docker.internal \
  lore-system
```

### Kubernetes

Deploy to Kubernetes:
```bash
kubectl apply -f k8s/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 