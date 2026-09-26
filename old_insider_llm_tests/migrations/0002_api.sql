-- Subject models can be served by OpenRouter or Ollama. For OpenRouter, `provider` is the
-- pinned endpoint tag; for Ollama it is the model's manifest digest.
ALTER TABLE runs ADD COLUMN api TEXT NOT NULL DEFAULT 'openrouter' CHECK (api IN ('openrouter', 'ollama'));
ALTER TABLE runs ALTER COLUMN api DROP DEFAULT;
