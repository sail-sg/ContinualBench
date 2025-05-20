PY_SOURCE=continual_bench tests

lint:
	@black --check --diff ${PY_SOURCE}
	@ruff check ${PY_SOURCE}

format:
	@autoflake --remove-all-unused-imports -i -r ${PY_SOURCE}
	@isort --project=continual_bench ${PY_SOURCE}
	@black ${PY_SOURCE}
	@ruff check --fix ${PY_SOURCE}
