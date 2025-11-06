# Contributing to AlphaPredict

Thank you for your interest in contributing to AlphaPredict! We appreciate your time and effort in making this project better.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). By contributing to this project, you agree that your contributions will be licensed under the MPL-2.0 license.

## How to Contribute

1. **Fork the repository** and create your branch from `main`.
2. **Set up the development environment**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```
3. **Make your changes** and ensure tests pass:
   ```bash
   # Run tests
   pytest
   
   # Run type checking
   mypy src
   
   # Check code style
   black --check .
   isort --check-only .
   flake8 .
   ```
4. **Commit your changes** with a clear commit message.
5. **Push** to your fork and submit a pull request.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. Your pull request will be reviewed by the maintainers.

## Reporting Issues

When reporting issues, please include the following:
- A clear and descriptive title
- A description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots

## Development Setup

For local development, you can use the provided `pyproject.toml` to set up the development environment with all necessary dependencies.

## Code Style

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with the following additional guidelines:
- Use type hints for all function signatures
- Include docstrings for all public modules, functions, and classes
- Keep lines under 88 characters (Black's default)
- Use double quotes for strings (enforced by Black)

## Testing

Write tests for all new functionality. Run the test suite with:

```bash
pytest
```

## Documentation

Keep the documentation up to date with any changes to the codebase. This includes:
- README.md
- Docstrings
- Any additional documentation in the `docs/` directory

## Questions?

If you have any questions, feel free to open an issue or contact the maintainers.
