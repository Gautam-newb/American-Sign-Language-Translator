# Contributing to KOMMUNIK8

Thank you for your interest in contributing to KOMMUNIK8! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in existing issues
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. Check existing feature requests
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the coding style (PEP 8 for Python)
   - Add docstrings to new functions/classes
   - Update documentation if needed
   - Add tests if applicable
4. **Test your changes**:
   - Ensure the application runs without errors
   - Test the new functionality thoroughly
5. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```
   Use clear, descriptive commit messages
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/kommunik8.git
   cd kommunik8
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make your changes and test them

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

### Example Function Documentation

```python
def process_gesture(landmark_list):
    """
    Process hand landmarks to classify gesture.
    
    Args:
        landmark_list (list): List of 21 hand landmark coordinates
        
    Returns:
        str: Classified gesture label or None if no gesture detected
    """
    # Implementation
    pass
```

### Commit Message Format

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- First line should be concise (50 chars or less)
- Add detailed description if needed

Examples:
- `Add: gesture history visualization`
- `Fix: camera initialization on Windows`
- `Update: improve model accuracy with data augmentation`

## Areas for Contribution

### High Priority

- Improve model accuracy
- Add support for more gestures
- Enhance UI/UX
- Performance optimization
- Better error handling

### Medium Priority

- Add unit tests
- Improve documentation
- Add support for multiple languages
- Mobile app development
- Integration with other services

### Low Priority

- Code refactoring
- Additional training data
- UI theme customization
- Advanced filtering options

## Testing

Before submitting a PR, ensure:

1. The application starts without errors
2. Camera feed works correctly
3. Gesture recognition functions properly
4. No console errors or warnings
5. Code follows style guidelines

## Questions?

If you have questions about contributing, please:
- Open an issue with the "question" label
- Check existing documentation
- Review closed issues for similar questions

Thank you for contributing to KOMMUNIK8! 🎉

