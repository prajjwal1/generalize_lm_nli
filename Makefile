quality:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check --line-length 119 --target-version py35 .
	isort --check-only --recursive .

style:
	black --line-length 119 --target-version py35 .
	isort --recursive .

