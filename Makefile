# Every step of the pipeline; see README.md.
CRATE  := old_insider_llm_tests
ILD    := cd $(CRATE) && cargo run --release --quiet --
MODEL  ?= qwen/qwen3.8-27b
PHASE  ?= pilot
SET    ?= original

.PHONY: db db-down data run judge eval report baseline check

db:        ## start PostgreSQL and apply migrations
	docker compose up -d --wait
	$(ILD) migrate

db-down:   ## stop PostgreSQL (data is kept in the volume)
	docker compose down

data:      ## load a stimulus set from ../benchmark into the database
	$(ILD) load --set $(SET)

run:       ## run one model: make run MODEL=... PHASE=smoke|pilot|study [ARGS="--samples 1 --allow-dirty"]
	$(ILD) run --model $(MODEL) --phase $(PHASE) --set $(SET) $(ARGS)

judge:     ## label test awareness of every unjudged episode
	$(ILD) judge-aware

eval: judge

report:    ## print tables and write ../results/<run_id>/
	$(ILD) report

baseline:  ## keyword baseline for test awareness (not implemented yet)
	@echo "baseline: not implemented yet (planned: keyword match vs. judge)"; exit 1

check:     ## format, lint and test
	cd $(CRATE) && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
