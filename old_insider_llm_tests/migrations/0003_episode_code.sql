-- The code commit that produced each episode, so a resumed run shows which episodes ran on
-- which code. NULL for episodes stored before this migration (their run's commit applies).
ALTER TABLE episodes ADD COLUMN code_git_hash TEXT;
