# Contributing to Dementia Detection

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/KinannAlamir/DetectingDementia_2025.git
cd DetectingDementia_2025
```

2. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
uv sync
```

## Code Quality

This project uses Ruff for linting and formatting:

```bash
# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Testing

Run tests before submitting a PR:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/dementia_detection --cov-report=term-missing

# Run verbose
uv run pytest -v
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular
- Add tests for new features

## Questions?

Feel free to open an issue for any questions or concerns!
