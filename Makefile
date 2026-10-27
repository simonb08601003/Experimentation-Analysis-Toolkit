.PHONY: data analysis test
data:
	bash scripts/download_data.sh
analysis:
	python scripts/run_hillstrom.py
	python scripts/run_megafon_uplift.py
	python scripts/run_peeking_sim.py
test:
	python -m pytest -q tests
