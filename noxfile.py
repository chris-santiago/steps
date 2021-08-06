import nox

PROJECT = 'steps'


@nox.session(reuse_venv=True)
def tests(session):
    """Run unit tests in current Python environment."""
    session.install('pytest', 'pytest-cov', 'pandas')
    session.install('.')
    session.run('pytest')


@nox.session(python=False)
def lint(session):
    """Lint source code using Pylint, Flake8 and Mypy."""
    session.run('pylint', PROJECT, '--verbose')
    session.run('flake8', PROJECT, '--count', '--statistics', '--select=E9,F63,F7,F82', '--show-source')  # these fail
    session.run('flake8', PROJECT, '--count', '--statistics', '--exit-zero')  # these warn
    session.run('mypy', '-p', PROJECT)


@nox.session(python=False)
def docs(session):
    """Build package documentation."""
    session.run('sphinx-apidoc', PROJECT, '-o', 'docs/source/')
    session.run('sphinx-build', '-b', 'html', 'docs/source/', 'docs/build/html')


@nox.session(python=False)
def qa(session):
    """Run QA code checks."""
    session.run('check-manifest')
    session.run('isort', '.')
    session.run('pre-commit', 'run', 'trailing-whitespace', '--files', '*.py')
    session.run('pre-commit', 'run', 'end-of-file-fixer', '--files', '*.py')
    session.run('pre-commit', 'run', 'check-yaml', '--all-files')
