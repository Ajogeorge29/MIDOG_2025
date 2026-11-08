.PHONY: install download-track2 download-track1-metadata audit test lint clean-reports

install:
	python -m pip install -e ".[dev]"

download-track2:
	python -m midog2025.download --track track2

download-track1-metadata:
	python -m midog2025.download --track track1-metadata

audit:
	python -m midog2025.audit

test:
	python -m pytest

lint:
	python -m ruff check .

clean-reports:
	python -c "import shutil; shutil.rmtree('reports/generated', ignore_errors=True)"
