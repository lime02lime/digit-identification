-- db initialization upon container startup/creation
CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    true_label TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
