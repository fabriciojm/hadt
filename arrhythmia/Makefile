local_run_api:
	uvicorn api.fast:app --host 0.0.0.0 --port 8080 --reload

run_preprocess:
	python -c 'from arrhythmia.ml_logic.main import preprocess; preprocess()'
