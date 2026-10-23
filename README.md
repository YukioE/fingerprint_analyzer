# Fingerprint Analyzer

Fingerprint Analyzer is a command-line tool for examining Bitcoin transactions
and identifying wallet software from transaction patterns. It analyzes
transaction scripts and metadata, builds fingerprint features, and can compare
those features with labeled examples using machine-learning classifiers.

## Features

- Analyze one or more transactions and print detailed JSON reports.
- Detect transaction patterns (see [fingerprints](./src/fingerprint_analyzer/fingerprints/)), including:
  - BIP69 input/output ordering and anti-fee-sniping locktimes.
  - Replace-By-Fee signaling, transaction version, output count, and OP_RETURN.
  - Fee-rate choices and ECDSA signature/public-key characteristics.
  - Output address families: Base58, Bech32, and Bech32m.
  - Spent input types, including P2PK, P2PKH, P2SH, P2WPKH, P2WSH, and P2TR.
- Convert labeled transaction lists into feature datasets.
- Train and evaluate decision-tree, random-forest, and K-nearest-neighbor
  classifiers; identify wallets using the decision-tree classifier.
- Inspect likely change outputs by comparing the wallet labels of a transaction
  and the transactions that spend its outputs. This is a heuristic, not a
  guarantee.
- Collect confirmed transactions continuously and store their fingerprints and
  block metadata in PostgreSQL.
- Generate optional classifier evaluation and monthly statistics plots.

Fingerprint results use `true`, `false`, or `null` in transaction reports.
Machine-learning datasets encode these as `1`, `0`, and `-1`, respectively;
`-1` means the feature could not be determined.

## Requirements and dependencies

- Python 3.11 or newer.
- The dependencies installed by the project:
  - `scikit-learn >= 1.2`
  - `psycopg[binary] >= 3.1`
- PostgreSQL is only needed for the `collect` command and monthly statistics
  script. (the repository contains a [docker compose yaml](./compose.yaml) for an easy setup of the database)
- `matplotlib >= 3.7` is optional and needed for the plotting scripts.
- Transaction commands access an Electrs-compatible API, such as the public
  mempool.space API, over the internet.

## Installation

Run these commands from the repository directory.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell does not allow activation scripts, use the virtual environment's
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### macOS and Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install plotting support when needed:

```bash
python -m pip install -e ".[pruning-eval]"
```

The optional extra includes Matplotlib. The package can then be run either as a
command or as a Python module:

```text
fingerprint_analyzer --help
python -m fingerprint_analyzer --help
```

## CLI commands

Use `fingerprint_analyzer <command> --help` for command-specific help.

```text
fingerprint_analyzer tx <TXID> [<TXID> ...] [--testnet | --testnet4] [--mempool-url URL]
fingerprint_analyzer dataset <input_csv> [--mempool-url URL]
fingerprint_analyzer train [--dataset CSV]
fingerprint_analyzer identify <TXID> [<TXID> ...] [--dataset CSV] [--testnet | --testnet4] [--mempool-url URL]
fingerprint_analyzer change <TXID> [--dataset CSV] [--testnet | --testnet4] [--mempool-url URL]
fingerprint_analyzer collect [--testnet | --testnet4] [--mempool-url URL] [--database-url URL] [--start-height HEIGHT] [--poll-interval SECONDS] [--block-interval N]
```

| Command | Purpose |
| --- | --- |
| `tx` (alias: `transaction`) | Fetch and analyze one or more transactions; output JSON. |
| `dataset` | Fetch transactions from a labeled CSV and write fingerprint features as CSV to standard output. |
| `train` | Evaluate all classifiers on a CSV training dataset. |
| `identify` | Predict wallet labels for one or more transactions using a CSV dataset and all classifiers. |
| `change` | Predict a transaction's wallet and inspect its outputs' spenders for likely change using a CSV dataset. |
| `collect` | Continuously collect confirmed transactions into PostgreSQL. |

The `train`, `identify`, and `change` commands accept `--dataset CSV` to choose
the training dataset. When omitted, the first alphabetically sorted CSV in the
package's `data` directory is used.

### Common transaction options

The `tx`, `identify`, and `change` commands accept:

| Option | Description |
| --- | --- |
| `--testnet` | Use Bitcoin testnet3 instead of mainnet. |
| `--testnet4` | Use Bitcoin testnet4 instead of mainnet. |
| `--mempool-url URL` | Use a custom Electrs-compatible API base URL instead of the network default. |

`--testnet` and `--testnet4` cannot be used together. If no network option is
given, mainnet is used. `collect` also accepts the network options, while
`dataset` uses the network value from each CSV row. `dataset` and `collect`
accept `--mempool-url`.

### `collect` options

| Option | Default | Description |
| --- | --- | --- |
| `--database-url URL` | `DATABASE_URL` environment variable | PostgreSQL connection URL. Required if the environment variable is not set. |
| `--start-height HEIGHT` | Current chain tip for a new database | First block to collect. When resuming, collection continues after the latest stored block. |
| `--poll-interval SECONDS` | `10` | Time between checks for new blocks; must be greater than zero. |
| `--block-interval N` | `1` | Store every Nth block; must be greater than zero. |

Stop the collector with Ctrl+C. It will resume from the latest stored block on
the next run.

## Examples

### Analyze transactions

```bash
fingerprint_analyzer tx <TXID>
fingerprint_analyzer tx <TXID_1> <TXID_2> --testnet
fingerprint_analyzer transaction <TXID> --mempool-url https://your-electrs.example/api

fingerprint_analyzer tx b68df4223acd6231908d19af201d710a496cd4832b724b6e1d16711d8d85f862

{
  "txid": "b68df4223acd6231908d19af201d710a496cd4832b724b6e1d16711d8d85f862",
  "version": 1,
  "locktime": 0,
  "confirmed": true,
  "block_height": 970000,
  "input_count": 1,
  "output_count": 3,
  "fee_sats": 413,
  "vbytes": 172,
  "fee_rate_sat_vb": 2.401,
  "input_type_set": [
    "p2wpkh"
  ],
  "output_type_set": [
    "bech32"
  ],
  "sequence_set": [
    "0xffffffff"
  ],
  "sighash_set_detail": [
    1
  ],
  "fingerprints": {
    "anti_fee_sniping": false,
    "bip69": true,
    "compressed_ecdsa_public_keys": true,
    "low_r_grinding": false,
    "manual_fee_rate_setting": true,
    "more_than_2_outputs": true,
    "multi_type_vin": false,
    "op_return_outputs": false,
    "default_opt_in_rbf_set": false,
    "sending_to_base58": false,
    "sending_to_bech32": true,
    "sending_to_bech32m": false,
    "spending_from_p2pk": false,
    "spending_from_p2pkh": false,
    "spending_from_p2sh_all": false,
    "spending_from_p2sh_multisig": false,
    "spending_from_p2sh_p2wpkh": false,
    "spending_from_p2tr": false,
    "spending_from_p2wpkh": true,
    "spending_from_p2wsh_all": false,
    "spending_from_p2wsh_multisig": false,
    "txversion_2": false,
    "uncompressed_ecdsa_public_keys": false
  }
}
```

### Build a feature dataset

Create an input CSV with a `txid`, `wallet`, and `network` column. The `wallet`
column contains the label used to train the classifiers:

```csv
txid,wallet,network
0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef,Example Wallet,mainnet
```

Supported network names include `mainnet`, `bitcoin`, `bitcoin-mainnet`,
`testnet`, `testnet3`, and `testnet4`.

Generate a feature CSV by redirecting standard output:

```bash
fingerprint_analyzer dataset labeled_transactions.csv > features.csv

wallet,anti_fee_sniping,bip69,compressed_ecdsa_public_keys,low_r_grinding,manual_fee_rate_setting,more_than_2_outputs,multi_type_vin,op_return_outputs,default_opt_in_rbf_set,sending_to_base58,sending_to_bech32,sending_to_bech32m,spending_from_p2pk,spending_from_p2pkh,spending_from_p2sh_all,spending_from_p2sh_multisig,spending_from_p2sh_p2wpkh,spending_from_p2tr,spending_from_p2wpkh,spending_from_p2wsh_all,spending_from_p2wsh_multisig,txversion_2,uncompressed_ecdsa_public_keys
Sparrow,1,0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0
Sparrow,1,0,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0
...

Created 100 rows with 23 fingerprint features
```

You can override the API URL with `--mempool-url`. The output contains the wallet label and fingerprint columns, but does not
include the transaction ID. Features are encoded as `1` (true), `0` (false),
and `-1` (unknown).

### Train and identify

```bash
fingerprint_analyzer train

Training dataset: path/to/features.csv
Samples: 100 | Features: 23 | Wallets: 7
Class distribution: Bitbox: 10, Blockstream: 11, Core: 13, Electrum: 13, Sparrow: 18, Trezor: 22, Wasabi: 13
Random baseline (uniform guess): 14.3% (1/7 labels)
Unknown feature values: 54 / 2300

Decision Tree
  Holdout: 79 training / 21 test samples
  Accuracy: 57.1% | Balanced accuracy: 60.7% | Macro F1: 57.8% | Weighted F1: 55.2%
  Per-class one-vs-rest metrics:
    Class            TPr      TNr      FPr      FNr  Precision   Recall       F1
    -----------  -------  -------  -------  -------  ---------  -------  -------
    Bitbox        100.0%   100.0%     0.0%     0.0%     100.0%   100.0%   100.0%
    ...
  Confusion matrix:
    Actual / Predicted |      Bitbox | Blockstream |        Core |    Electrum |     Sparrow |      Trezor |      Wasabi
    -------------------+-------------+-------------+-------------+-------------+-------------+-------------+------------
    Bitbox             |           2 |           0 |           0 |           0 |           0 |           0 |           0
    ...
  Top features:
    anti_fee_sniping: 19.4%
    ...

Random Forest
  Holdout: 79 training / 21 test samples
  Accuracy: 61.9% | Balanced accuracy: 65.5% | Macro F1: 61.8% | Weighted F1: 59.3%
  Per-class one-vs-rest metrics:
    Class            TPr      TNr      FPr      FNr  Precision   Recall       F1
    -----------  -------  -------  -------  -------  ---------  -------  -------
    Bitbox        100.0%   100.0%     0.0%     0.0%     100.0%   100.0%   100.0%
    ...
  Confusion matrix:
    Actual / Predicted |      Bitbox | Blockstream |        Core |    Electrum |     Sparrow |      Trezor |      Wasabi
    -------------------+-------------+-------------+-------------+-------------+-------------+-------------+------------
    Bitbox             |           2 |           0 |           0 |           0 |           0 |           0 |           0
    ...
  Top features:
    anti_fee_sniping: 15.5%
    ...

K-Nearest Neighbors
  Holdout: 79 training / 21 test samples
  Accuracy: 42.9% | Balanced accuracy: 48.8% | Macro F1: 48.4% | Weighted F1: 43.1%
  Per-class one-vs-rest metrics:
    Class            TPr      TNr      FPr      FNr  Precision   Recall       F1
    -----------  -------  -------  -------  -------  ---------  -------  -------
    Bitbox        100.0%   100.0%     0.0%     0.0%     100.0%   100.0%   100.0%
    ...
  Confusion matrix:
    Actual / Predicted |      Bitbox | Blockstream |        Core |    Electrum |     Sparrow |      Trezor |      Wasabi
    -------------------+-------------+-------------+-------------+-------------+-------------+-------------+------------
    Bitbox             |           2 |           0 |           0 |           0 |           0 |           0 |           0
    ...
```

```bash
fingerprint_analyzer identify <TXID>

Wallet identification
Dataset: features.csv

Transaction: b68df4223acd6231908d19af201d710a496cd4832b724b6e1d16711d8d85f862

  [1] Decision Tree
      1. Wasabi (100.0%)
      2. Bitbox (0.0%)
      3. Blockstream (0.0%)
    Reasons: bip69=1, more_than_2_outputs=1, anti_fee_sniping=0

  [2] Random Forest
      1. Bitbox (50.0%)
      2. Wasabi (36.0%)
      3. Trezor (8.0%)
    Reasons: bip69=1, more_than_2_outputs=1, manual_fee_rate_setting=1

  [3] K-Nearest Neighbors
      1. Electrum (40.0%)
      2. Trezor (40.0%)
      3. Bitbox (20.0%)
    Reasons: bip69=1, compressed_ecdsa_public_keys=1, manual_fee_rate_setting=1
```

When `--dataset` is omitted, `train`, `identify`, and `change` use the first
alphabetically sorted CSV in the package's `data` directory. A dataset created
with the `dataset` command can be passed directly with `--dataset`. Extra
columns not produced by the analyzer are ignored.

`train` reports dataset statistics and deterministic holdout metrics for all
three models. `identify` fits the models on the full dataset, uses the
decision-tree prediction as the identified label, and shows each model's top
three labels and scores for comparison. The scores are not calibrated
probabilities that the predicted wallet is correct.

### Check likely change outputs

```bash
fingerprint_analyzer change <TXID>

Dataset:          features.csv
Transaction:      b68df4223acd6231908d19af201d710a496cd4832b724b6e1d16711d8d85f862
Decision-tree wallet: Wasabi

Likely change outputs:
  - Output 2


Output 0
  Status:       Not identified as change
  Spent by:     ded2bc577d46280606e3d8c5f96150904fc521369e4ec8ba733d7eeec9c2d2ba
  Wallet label: Core

Output 1
  Status:       Unspent

Output 2
  Status:       Likely change
  Spent by:     f62341cdacb39b96efc85d2f4ca15f7764128b1135f57a83a984e43dcc68814b
  Wallet label: Wasabi
```

The command checks the wallet label predicted for the original transaction
against the labels predicted for transactions spending its outputs. Outputs
that are unspent or cannot be classified are reported separately.

### Collect data in PostgreSQL

Set the connection URL and start collecting from the current tip:

```bash
# macOS/Linux
export DATABASE_URL="postgresql://user:password@localhost:5432/fingerprints"
fingerprint_analyzer collect
```

```powershell
# Windows PowerShell
$env:DATABASE_URL = "postgresql://user:password@localhost:5432/fingerprints"
fingerprint_analyzer collect
```

Or pass the connection URL directly:

```bash
fingerprint_analyzer collect --database-url "postgresql://user:password@localhost:5432/fingerprints" --start-height 800000 --poll-interval 15 --block-interval 2
```

The tool creates its transaction and block tables if they do not already exist.
It stores transaction fingerprints in `fingerprint_transactions` and block
hashes and timestamps in `fingerprint_blocks`. A `NULL` database feature means
the transaction did not provide enough information to determine that feature.
Collection resumes after the highest stored block, but does not detect or
recover from chain reorganizations. Network selection is not stored in the
database, so use a separate database for each Bitcoin network.

## Plotting scripts

Install the optional plotting dependency first:

```bash
python -m pip install -e ".[pruning-eval]"
```

Evaluate classifier behavior and save plots:

```bash
python scripts/evaluate_models.py
python scripts/evaluate_models.py path/to/features.csv --output pruning.png
```

The evaluation script uses an interactive display when available; otherwise it
saves PNG files in the current directory or beside the path given to `--output`.

Generate monthly charts from collected PostgreSQL data:

```bash
python scripts/plot_monthly_statistics.py --database-url "$DATABASE_URL"
python scripts/plot_monthly_statistics.py --database-url "$DATABASE_URL" --output-dir monthly_statistics
```

On Windows PowerShell, use `$env:DATABASE_URL` in place of `$DATABASE_URL`.

## Development

The source code is under `src/fingerprint_analyzer`. Fingerprint detectors are
kept as small modules in `src/fingerprint_analyzer/fingerprints`; the registry
discovers detector functions from that package. The scripts directory contains
optional analysis and visualization utilities.
