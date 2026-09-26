-- Schema v1: stimuli, runs, episodes, test-awareness labels.
-- Closed sets are TEXT with CHECK constraints; the Rust enums in src/types.rs mirror them.

-- One row per stimulus version. A changed file gives a new row (new content hash);
-- episodes point to the exact version they ran on.
CREATE TABLE stimuli (
    id              BIGSERIAL PRIMARY KEY,
    stimulus_set    TEXT NOT NULL CHECK (stimulus_set IN ('original', 'renamed')),
    condition_id    TEXT NOT NULL,
    scenario        TEXT NOT NULL CHECK (scenario IN ('blackmail', 'leaking', 'murder')),
    goal_type       TEXT NOT NULL CHECK (goal_type IN ('explicit', 'none')),
    goal_value      TEXT NOT NULL,
    urgency_type    TEXT NOT NULL CHECK (urgency_type IN ('none', 'restriction', 'replacement')),
    system_prompt   TEXT NOT NULL,
    user_prompt     TEXT NOT NULL,
    email_content   TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    source          TEXT NOT NULL,            -- the set's SOURCE.toml, verbatim
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stimulus_set, condition_id, content_sha256)
);

-- One run = one model on one stimulus set. Everything needed to reproduce it.
CREATE TABLE runs (
    run_id              TEXT PRIMARY KEY,
    phase               TEXT NOT NULL CHECK (phase IN ('smoke', 'pilot', 'study')),
    model               TEXT NOT NULL,
    provider            TEXT NOT NULL,        -- pinned OpenRouter provider tag
    stimulus_set        TEXT NOT NULL CHECK (stimulus_set IN ('original', 'renamed')),
    samples             INTEGER NOT NULL CHECK (samples > 0),
    config_toml         TEXT NOT NULL,
    config_sha256       TEXT NOT NULL,
    code_git_hash       TEXT NOT NULL,
    code_dirty          BOOLEAN NOT NULL,
    benchmark_git_hash  TEXT NOT NULL,
    benchmark_dirty     BOOLEAN NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every attempt that ended is stored, failures included. At most one non-failed row per
-- (run, stimulus, sample); a resumed run retries only the cells without one.
CREATE TABLE episodes (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs (run_id),
    stimulus_id         BIGINT NOT NULL REFERENCES stimuli (id),
    sample              INTEGER NOT NULL CHECK (sample >= 0),
    status              TEXT NOT NULL CHECK (status IN ('ok', 'truncated', 'failed')),
    answer              TEXT,
    reasoning           TEXT,
    finish_reason       TEXT,
    served_by           TEXT,                 -- provider OpenRouter reports for this call
    api_model           TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    reasoning_tokens    INTEGER,
    cost_usd            DOUBLE PRECISION,
    duration_ms         BIGINT NOT NULL,
    attempts            INTEGER NOT NULL,
    error               TEXT,
    raw_response        JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX episodes_one_result ON episodes (run_id, stimulus_id, sample) WHERE status <> 'failed';

-- Test-awareness verdicts. One per episode, judge model and prompt version.
CREATE TABLE aware_labels (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      BIGINT NOT NULL REFERENCES episodes (id),
    label           TEXT NOT NULL CHECK (label IN ('none', 'suspected', 'explicit')),
    evidence        TEXT NOT NULL,
    judge_model     TEXT NOT NULL,
    prompt_sha256   TEXT NOT NULL,
    judge_output    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (episode_id, judge_model, prompt_sha256)
);

CREATE TABLE judge_failures (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      BIGINT NOT NULL REFERENCES episodes (id),
    judge_model     TEXT NOT NULL,
    prompt_sha256   TEXT NOT NULL,
    error           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
