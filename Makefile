.PHONY: setup data train sample-train test lint format card bench results drift-demo compare gate ci-gate demo serve docker clean

UV ?= uv
PORT ?= 8000
export PYTHONUTF8 = 1

setup:            ## install Python 3.12 toolchain and dependencies
	$(UV) python install 3.12
	$(UV) sync
	@test -f .env || cp .env.example .env

data:             ## download OpenML creditcard (id 1597) to data/raw/ and rebuild the sample
	$(UV) run python scripts/fetch_data.py

train:            ## train baseline, champion, challenger on the full data (about 1 min on CPU)
	$(UV) run fraudscore train

test:             ## run the test suite
	$(UV) run pytest

lint:             ## ruff lint + format check
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:           ## auto-format
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

card:             ## reports/model_card.md from reports/metrics.json
	$(UV) run python scripts/model_card.py

bench:            ## p95 /score latency against a local uvicorn server -> reports/bench.json
	$(UV) run python scripts/bench.py

results:          ## reports/results.md and the README results table
	$(UV) run python scripts/results_table.py

drift-demo:       ## shifted synthetic stream against the champion -> reports/drift-<date>.md
	$(UV) run python scripts/synth.py --shift --rows 5000 --out data/synthetic/stream_shifted.parquet
	$(UV) run fraudscore drift data/synthetic/stream_shifted.parquet

compare:          ## champion vs challenger on the test split -> reports/eval-*.json, compare.md
	$(UV) run fraudscore compare

gate:             ## regression gate: champion vs the committed sample baseline (needs sample models)
	$(UV) run evalkit gate --baseline reports/baseline-sample.json --candidate reports/ci/eval-champion.json --config evals.yaml --md reports/ci/gate.md

ci-gate:          ## what CI runs: train on the sample, compare, gate
	$(UV) run fraudscore train --data data/sample/creditcard_sample.parquet --models-dir models/ci --reports-dir reports/ci
	$(UV) run fraudscore compare --data data/sample/creditcard_sample.parquet --champion models/ci/champion.joblib --challenger models/ci/challenger.joblib --out-dir reports/ci
	$(MAKE) gate

demo:             ## offline end-to-end demo on the committed sample
	$(UV) run python scripts/demo.py

serve:            ## run the API on http://localhost:$(PORT) (needs models/champion.joblib)
	$(UV) run uvicorn fraudscore.api:app --host 0.0.0.0 --port $(PORT)

docker:           ## build the image (needs models/champion.joblib and ../llm-evalops)
	docker build --build-context llm-evalops=../llm-evalops -t fraud-scoring-service:latest .

clean:
	rm -rf .pytest_cache .ruff_cache models/demo models/ci reports/demo reports/ci
