quality:
	black --check --line-length 119 --target-version py35 .
	isort --check-only --recursive .
	flake8 .

style:
	black --line-length 119 --target-version py35 .
	isort --recursive .
