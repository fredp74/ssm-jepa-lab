# First financial corpus

Run `python prepare_finance.py` from the repository root.
The destination `data/finance_seed/` must not exist.
The output uses the existing ssm-jepa-corpus-v1 format.
Preparation does not load or change model checkpoints.

## Scope and provenance

Six original English educational documents cover statements, margins,
valuation, rates, liquidity and portfolio evidence. Four documents train;
two validate. This 8 KB seed checks the preparation pipeline and is not
sufficient for financial specialization. No actual issuer filings are included.
All company figures are fictional. SEC and Federal Reserve references are
conceptual references consulted on 2026-11-14, not copied source text.
Publication dates describe the authored seed, not reference-page dates.

The original seed prose and exercises are dedicated under CC0-1.0:
https://creativecommons.org/publicdomain/zero/1.0/
External reference pages and other repository code are not relicensed.

Document metadata records IDs, groups, splits, dates, periods, units,
rights, rights basis, references and material type. The manifest records
normalized text hashes and the source registry hash.

Exact duplicate text, duplicate IDs and groups crossing partitions are
rejected. Unicode NFC and edge trimming are applied. Internal table text
is unchanged. No HTML/PDF extraction, paraphrase detection, fact checking
or automatic rights verification is implemented.

Partitions hold out topics, not companies or historical periods.
Related concepts occur in both partitions. Real documents must use actual
release dates, preserve tables, units and source fingerprints, and reserve
issuer groups and historical periods before preparation. SEC hosting alone
does not make issuer-authored filings public domain.

## Exercises

corpora/finance/exercises.jsonl contains three evaluation-only exercises:
margin calculation, liquidity interpretation and an interest-rate scenario.
Each has inputs, assumptions, expected criteria, an answer, source IDs and dates.
Exercises are excluded from the language corpus. These are development
checks, not a pristine final test set; their concepts occur in the prose.
No automatic scoring or instruction tuning is implemented in this change.

## Next training gate

Expand beyond this seed before financial specialization. Keep evaluation
answers out of training and prepare separate instruction-training examples.
Preserve the English baseline at 10,034 steps and 1.304136465797244 bpb
on 64 windows. Specialization should use a separate checkpoint copy.
Measure retained English loss and generation alongside financial results.
No training command is prescribed for this tiny seed.
No financial or investment ability has been demonstrated by preparation.

Run all checks with `python -m unittest discover -s tests -v`.
