# Contributing to AimGuardian AI

Thank you for your interest in contributing to AimGuardian AI! This document provides guidelines and instructions for contributing to the project.

## 🎯 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🤝 How to Contribute

### 1. Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AimGuardian-AI.git
   cd AimGuardian-AI
   ```
3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/LKZOwner/AimGuardian-AI.git
   ```

### 2. Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### 3. Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards:
   - Use meaningful commit messages
   - Follow PEP 8 style guide
   - Add tests for new features
   - Update documentation

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

4. Check code style:
   ```bash
   flake8 .
   black .
   ```

### 4. Submitting Changes

1. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request:
   - Use the PR template
   - Describe your changes
   - Link any related issues
   - Ensure CI passes

## 📝 Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the README.md if needed
5. The PR will be reviewed by maintainers

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_neural_network.py

# Run with coverage
python -m pytest --cov=aimguardian tests/
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Include both unit and integration tests

## 📚 Documentation

### Code Documentation
- Use docstrings (Google style)
- Include type hints
- Document all public functions
- Add comments for complex logic

### Example:
```python
def process_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Process an image for model input.

    Args:
        image: Input image as numpy array
        target_size: Target size for resizing (width, height)

    Returns:
        Processed image as numpy array

    Raises:
        ValueError: If image is invalid
    """
    # Implementation
```

## 🐛 Bug Reports

When reporting bugs, please include:
1. Clear description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details
6. Screenshots if applicable

## 💡 Feature Requests

When suggesting features:
1. Clear description of the feature
2. Use case/benefit
3. Implementation suggestions (if any)
4. Any relevant examples

## 🔧 Development Tools

### Recommended Tools
- VS Code with Python extension
- PyCharm Professional
- Git for version control
- Black for code formatting
- Flake8 for linting
- Pytest for testing

### VS Code Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

## 📋 Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

### Example
```
feat(network): add attention mechanism

- Implemented self-attention layer
- Added attention visualization
- Updated documentation

Closes #123
```

## 🏆 Recognition

Contributors will be:
1. Added to the README.md
2. Given credit in release notes
3. Invited to join the project team (for significant contributions)

## ❓ Questions?

Feel free to:
1. Open an issue
2. Contact maintainers
3. Join our community discussions

Thank you for contributing to AimGuardian AI! 🎯 