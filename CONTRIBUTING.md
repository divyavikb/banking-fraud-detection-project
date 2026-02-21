# Contributing to Banking ML Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🌟 Ways to Contribute

- **Bug Reports**: Submit detailed bug reports with reproduction steps
- **Feature Requests**: Propose new features or improvements
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add test cases or improve test coverage

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/banking-ml-project.git
   cd banking-ml-project
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## 📝 Development Workflow

### Before Making Changes

1. **Update your fork**
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

### While Developing

1. **Write code** following our style guide
2. **Add tests** for new functionality
3. **Run tests locally**
   ```bash
   make test
   ```

4. **Format code**
   ```bash
   make format
   ```

5. **Check code quality**
   ```bash
   make quality
   ```

### Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

3. **Create Pull Request**
   - Go to GitHub and create a PR
   - Fill in the PR template
   - Link any related issues

## 📋 Code Style

- **Python**: Follow PEP 8
- **Line Length**: Maximum 100 characters
- **Formatting**: Use Black (`make format`)
- **Linting**: Pass Flake8 checks (`make lint`)
- **Type Hints**: Use type hints where appropriate

## 🧪 Testing Guidelines

- Write tests for all new features
- Maintain >80% code coverage
- Use pytest for testing
- Follow AAA pattern (Arrange, Act, Assert)

Example:
```python
def test_model_training():
    # Arrange
    data = create_sample_data()
    
    # Act
    model = train_model(data)
    
    # Assert
    assert model is not None
    assert model.score > 0.8
```

## 📖 Documentation

- Update README.md if adding features
- Add docstrings to all functions
- Update CHANGELOG.md
- Add examples to notebooks/ if relevant

## 🔀 Commit Messages

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

Example: `feat: add SHAP explanations to model predictions`

## 🐛 Reporting Bugs

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Screenshots if applicable

## ✨ Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Any related issues or PRs

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

## ❓ Questions?

Feel free to:
- Open an issue for questions
- Join our discussions
- Contact the maintainers

Thank you for contributing! 🎉
