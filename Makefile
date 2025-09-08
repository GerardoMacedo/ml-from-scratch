.PHONY: test testv run run-binary

# quiet test run (discovery)
test:
	python -m unittest discover -s tests -p "test*.py"

# verbose test run
testv:
	python -m unittest discover -s tests -p "test*.py" -v

# run CLI (multiclass)
run:
	python -m scratchml.cli

# run CLI (binary)
run-binary:
	python -m scratchml.cli --binary
