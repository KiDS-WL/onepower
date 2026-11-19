Contributing to OnePower
========================

``onepower`` is an open-source project that thrives on community contributions. You can help by writing tutorials, improving documentation, reporting bugs, suggesting features, or contributing code. Following these guidelines ensures a high-quality codebase and faster reviews.

Reporting Bugs
--------------
#. **Check existing issues** - comment if your bug is already listed.
#. **Create a new issue** if needed, including:
     * Python and ``onepower`` versions
     * Operating system
     * Steps to reproduce
     * Expected vs. actual behavior
     * Minimal code example
#. **Optional**: Submit a pull request with a failing test to expedite the fix.

Suggesting Features
-------------------
#. **Check existing issues** - comment if your idea is already listed.
#. **Create a new issue** if needed, with detailed context and implementation suggestions.

Code Contribution Guidelines
----------------------------
* **Discuss major changes** via issues to gather community feedback.
* **Avoid unrelated changes** - open separate pull requests for unrelated fixes/features.

Getting Started
---------------
#. Fork or branch the repository.
#. Follow the :ref:`INSTALLATION:Developer Install` guide.
#. Run ``pre-commit install`` for code quality checks.
#. Make your changes.
#. **Before submitting**:
     * Add tests for bug fixes.
     * Ensure all new code is tested by running ``pytest``.
     * Document new features (docstrings/tutorials).
#. Submit a pull request.

Code Review Process
-------------------
* The core team reviews pull requests regularly.
* Larger changes may take longer.
* Approval is required for merging.

Release Cycle & Versioning
--------------------------
* **Protected** ``main`` **branch**: All changes require a pull request.
* **Semantic versioning**:
    * **Major**: Breaking changes
    * **Minor**: New features
    * **Patch**: Bug fixes
* Versions are managed via ``setuptools_scm`` and deployed to PyPI automatically upon PR acceptance.
