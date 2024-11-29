.DEFAULT_GOAL := check

check:
	pre-commit run -a

changelog:
	cz bump --changelog

setup:
	python -m pip install --upgrade pip
	pip install poetry
	poetry config virtualenvs.create false
	poetry install --with=dev,deploy --no-root

format:
	docformatter --config pyproject.toml --in-place spaceai
	black --config=pyproject.toml spaceai
	pycln --config=pyproject.toml spaceai
	isort spaceai

build: format
	poetry build -v --no-cache --format wheel

verify:
	twine check --strict dist/*

publish-test: build verify
	poetry publish -r test-pypi --skip-existing -v

publish: build verify
	poetry publish --skip-existing -v
