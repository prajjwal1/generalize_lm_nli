quality:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .
	isort --check-only --recursive --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 .


style:
	black .
	isort --recursive --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 .

