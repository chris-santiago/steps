.PHONY: lint
lint:
	pylint steps --verbose

.PHONY: tests
tests:
	pytest --cov=steps --cov-report=html --verbose

.PHONY: docs
docs:
	sphinx-apidoc steps -o docs/source/
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: manifest
manifest:
	check-manifest

.PHONY: precommit
precommit:
	pre-commit run trailing-whitespace --all-files --verbose
	pre-commit run end-of-file-fixer --all-files --verbose
	pre-commit run check-yaml --all-files --verbose
	pre-commit run check-added-large-files --all-files --verbose

.PHONY: checks
checks: lint tests docs precommit
