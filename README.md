# fingerprint_analyzer

Bitcoin transaction fingerprint analyzer and wallet classifier. Transaction
data is fetched only through an Electrs-compatible API, using mempool.space by
default.

## Install

```bash
python -m pip install -e .
```

## Analyze transactions

```bash
fingerprint_analyzer tx <txid>
fingerprint_analyzer tx <txid-1> <txid-2> --testnet
fingerprint_analyzer tx <txid> --testnet4
```

Mainnet is the default. Reports are JSON. `--testnet` selects testnet3 and
`--testnet4` selects testnet4. Single-transaction reports place
`sequence_set` and `sighash_set_detail` beside the other transaction fields,
not inside a nested details object.

Fingerprint values are `true`, `false`, or `null` when transaction data cannot
establish a property. When multiple transaction IDs are provided, the command
returns one independent report per transaction in a JSON array.

Use `--mempool-url` on transaction-fetching commands (`tx`, `identify`,
`change`, `dataset`, or `collect`) to query another Electrs-compatible mempool
API instead of the network default:

```bash
fingerprint_analyzer tx <txid> --mempool-url https://mempool.example/api
```

## Collect fingerprint usage over time

The `collect` command continuously polls confirmed blocks, analyzes every
transaction in each selected block, and writes one row per transaction to
PostgreSQL. The table
`fingerprint_transactions` contains the transaction ID, block height, and one
nullable `BOOLEAN` column per registered fingerprint. Block timestamps and
hashes are stored once per block in `fingerprint_blocks`. A `NULL` fingerprint
means the transaction did not contain enough data to determine it. On restart,
collection resumes after the highest stored block. The collector does not
detect or recover from chain reorganizations. The collector creates these
tables on startup; use a new, empty database for this schema.
Create the PostgreSQL database first; the collector creates and updates its
tables on startup.

Install the PostgreSQL driver with the package, then provide a connection URL
through `DATABASE_URL` or `--database-url`:

```bash
docker compose up -d db
export DATABASE_URL=postgresql://fingerprint:fingerprint@localhost:5432/fingerprints
fingerprint_analyzer collect
fingerprint_analyzer collect --mempool-url https://mempool.example/api
fingerprint_analyzer collect --start-height 800000 --poll-interval 10
```

The Compose file starts a local PostgreSQL database with a persistent volume.
Its example credentials are for local development only.

By default, collection starts at the current tip and then checks for new blocks
every 10 seconds. `--start-height` enables historical backfill for an empty
database; existing stored blocks take precedence. `--block-interval N` stores
every Nth block, starting from the initial collection height (or the most
recently stored block when resuming); for example, `--block-interval 25` stores
one block for every 25 block heights. At Bitcoin's average rate of 144 blocks
per day, that is about 173 stored blocks per 30-day month, roughly one every
4 hours and 10 minutes. Use `--testnet`/`--testnet4`
to select a test network. Network selection is not stored in PostgreSQL, so use
a separate database for each chain. The configured API must support the
Electrs-compatible block and transaction endpoints.

For example, this query gives the monthly low-R rate among transactions where
the detector had enough signature data to decide:

```sql
SELECT date_trunc('month', b."timestamp")::date AS month,
       count(*) FILTER (WHERE t.low_r_grinding IS TRUE) AS low_r_transactions,
       count(*) FILTER (WHERE t.low_r_grinding IS NOT NULL) AS known_transactions,
       round(
           100.0 * count(*) FILTER (WHERE t.low_r_grinding IS TRUE)
           / nullif(count(*) FILTER (WHERE t.low_r_grinding IS NOT NULL), 0),
           2
       ) AS low_r_percent
FROM fingerprint_transactions AS t
JOIN fingerprint_blocks AS b USING (block_height)
GROUP BY month
ORDER BY month;
```

## Build a fingerprint dataset

Provide a CSV with `txid`, `wallet_label`, and `network` columns. Supported
networks are `mainnet`, `testnet`/`testnet3`, and `testnet4`.

```bash
fingerprint_analyzer dataset transactions.csv > tx_features.csv
```

The generated CSV contains wallet labels and fingerprint features, but no
`txid` column. Features are encoded as `1` (true), `0` (false), and `-1`
(unknown). Progress and errors are written to stderr, leaving stdout suitable
for redirection.

## Train classifiers

```bash
fingerprint_analyzer train
```

All three classifiers (decision tree, random forest, and K-nearest neighbors)
are trained and evaluated using the first `.csv` file in
`src/fingerprint_analyzer/data/`. A readable terminal report shows dataset
statistics, a uniform-random guess baseline (expected accuracy `1 / number of
wallet labels`), deterministic holdout metrics, per-class scores, and
confusion matrices. Confusion-matrix rows are actual labels and columns are
predicted labels. Feature importances are shown where available. Models are
kept in memory only. The `train` command fits each classifier once on the
holdout training split; `identify` instead fits once on the full dataset and
skips holdout evaluation.

## Evaluate decision-tree pruning

The optional standalone script plots training and holdout accuracy against
`ccp_alpha` for the decision tree. It is not part of the
`fingerprint_analyzer` command-line flow.

```bash
python -m pip install -e '.[pruning-eval]'
python scripts/evaluate_pruning.py
```

By default, it uses the first CSV in `src/fingerprint_analyzer/data/`. To
evaluate another fingerprint feature CSV, pass its path. With a non-interactive
matplotlib backend, the plot is saved as `decision_tree_pruning.png` in the
current directory; use `--output` to choose another path or explicitly save
the plot when using an interactive backend.

```bash
python scripts/evaluate_pruning.py path/to/features.csv
python scripts/evaluate_pruning.py --output pruning.png
```

## Identify wallets

```bash
fingerprint_analyzer identify <txid>
fingerprint_analyzer identify <txid> --testnet
fingerprint_analyzer identify <txid-1> <txid-2> --testnet4
```

Identification uses all three classifiers trained on the same first CSV
dataset. Each classifier's top three wallet labels and scores are displayed in
a separate labeled section for each transaction. Scores are not calibrated
probabilities of the predicted wallet being correct. During training, dataset
columns not currently generated by the analyzer are excluded from both
training and identification.

Add fingerprint detector modules under `src/fingerprint_analyzer/fingerprints`
and register them in `fingerprints/registry.py`.
