# Contributing to AquaGuard AI

Thank you for your interest in contributing to AquaGuard AI! We welcome contributions to improve plastic waste detection in water bodies using computer vision.

Please read through these guidelines to understand how you can participate and make the development process smooth and efficient for everyone.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful, welcoming, and collaborative environment. Please report any unacceptable behavior to the project maintainers.

---

## How to Contribute

### 1. Reporting Bugs or Requesting Features
- Search the issue tracker to check if the bug or feature request has already been reported.
- If not, open a new issue with a clear title, description, and steps to reproduce (for bugs) or clear motivation (for features).

### 2. Working on Issues
- Comment on the issue you would like to work on so we can assign it to you.
- Please do not start working on major changes without discussing them first in an issue.

### 3. Submission Process
1. **Fork the Repository:** Create a personal fork of the project on GitHub.
2. **Clone the Fork:** Clone it to your local machine:
   ```bash
   git clone https://github.com/<your-username>/AquaGuard-AI.git
   ```
3. **Create a Branch:** Create a branch for your changes (see [Branch Naming Conventions](#branch-naming-conventions) below).
4. **Develop:** Implement your changes. Make sure to adhere to standard code formatting rules and add tests.
5. **Run Tests:** Ensure all automated tests pass before committing.
6. **Commit Changes:** Write clear, concise, and descriptive commit messages following the [Commit Message Conventions](#commit-message-conventions).
7. **Push and Open a Pull Request:** Push to your fork and open a Pull Request (PR) against our `main` branch.

---

## Style Guides

### Python (ML Pipeline & FastAPI Backend)
- **Formatters & Linters:** We use `black` for formatting and `ruff` or `flake8`/`mypy` for linting and static typing checks.
- **Style:** Adhere strictly to **PEP 8**.
- **Type Hints:** Use explicit type hints for all function signatures and complex variables.
- **Documentation:** Use Google-style docstrings for functions, classes, and modules.

### TypeScript / React (Frontend)
- **Formatters & Linters:** We use `prettier` for formatting and `eslint` for linting.
- **Style:** Use functional components and React hooks. Keep styling modular.
- **Types:** Avoid the use of `any`. Define interfaces and types clearly.

---

## Branch Naming Conventions

Use prefix naming conventions for branches:
- `feature/<short-description>`: Adding new features.
- `bugfix/<short-description>`: Resolving bugs.
- `docs/<short-description>`: Improving documentation.
- `refactor/<short-description>`: Code changes that neither fix a bug nor add a feature.
- `ci/<short-description>`: CI/CD configuration updates.

*Example: `feature/ml-pipeline-integration` or `bugfix/api-cors-error`*

---

## Commit Message Conventions

We recommend following standard conventional commits:
- `feat: <description>` (for a new feature)
- `fix: <description>` (for a bug fix)
- `docs: <description>` (documentation changes)
- `refactor: <description>` (code optimization/re-structuring)
- `test: <description>` (adding or modifying tests)
- `chore: <description>` (build tasks, dependencies, etc.)

---

## Pull Request Checklist

Before submitting a Pull Request, please ensure:
- [ ] Your code runs locally without errors.
- [ ] You have formatted your code using the designated formatters (`black`, `prettier`).
- [ ] You have written unit/integration tests covering the new changes where applicable.
- [ ] All tests pass successfully.
- [ ] The documentation (README or inline docs) has been updated to reflect your changes.
- [ ] Your branch is rebased with the latest commits from the `main` branch.
