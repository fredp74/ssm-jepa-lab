# CSV structural behavior checks

## Scope

This experiment extends the CSV tool-discovery diagnostics from cell
contents to relations between tables.

The agent and checks are symbolic. No SSM, JEPA, or neural training is used.

## Frozen probe-bank evaluation

The fixed and contrast banks were evaluated without modification on eight
additional handcrafted transformations, applied to four base configurations.

Evaluation used 100 new tables, distinct from the earlier evaluation
tables and the complete probe tables.

Both banks:
- Correctly recognized all 48 known configurations.
- Detected 24/32 outside-family environments.
- Missed all four row-sorting environments.
- Missed all four empty-row-removal environments.

For the missed cases, exact output accuracy was:
- Row sorting: 50/100 tables.
- Empty-row removal: 96/100 tables.

High aggregate prediction accuracy did not establish correct behavior.

On additional environments, mean calls were 3.375 for the fixed bank
and 4.5 for the contrast bank. Both allowed eight calls and 24 cells.

These behaviors were designed by the experiment author.
This was not an independent blind evaluation.

## Relational regression checks

After inspecting those failures, six relations were implemented:
- Reverse rows.
- Reverse columns.
- Duplicate rows.
- Append an empty row.
- Prepend an empty row.
- Transpose the table.

A composite probe first identifies a known CSV format or rejects the
current family. When a format survives, outputs are decoded using that
format and compared through the expected table transformations.

## Results

| Measurement | Result |
|---|---:|
| Known configurations incorrectly rejected | 0/48 |
| Additional outside-family environments detected | 32/32 |
| Detected by initial configuration probe | 24/32 |
| Detected through relational checks | 8/32 |

Row sorting violates the row-reversal relation.
Empty-row removal violates empty-row insertion relations.

For known configurations, the complete check uses eight calls and
44 cells. The previous banks used at most 24 cells.

Across additional environments, the relational procedure averages
2.75 calls and 17 cells because it stops early when the initial probe
rejects the family. Once relational checking begins, all six relations
are evaluated.

## Interpretation

Cell contrasts and table relations cover different failure modes.
The relational checks repair the observed structural blind spots.

The 32/32 result is a regression result on previously inspected
transformations. It does not establish general unknown-behavior detection.

Rejecting a behavioral hypothesis does not reconstruct the tool.
Passing the relations does not certify arbitrary future behavior.

## Limitations

- Handwritten relations and known CSV decoding.
- Synthetic configurations and transformations.
- Related base configurations per behavior.
- Different cell and text budgets between procedures.
- Previously inspected behaviors in the relational experiment.
- No independent estimate of real-world failure frequency.

## Decision

Freeze the contrast and relational procedures.
Do not claim superiority from the 32/32 result without accounting for
the larger input budget and retrospective design.

A subsequent comparison should use new behaviors and a common,
prespecified interaction and payload budget.

## Artifacts

Scripts are under autonomous_lab/. Detailed local runs remain under
autonomous_lab/runs/, excluded from Git. The companion JSON preserves
configurations, summaries, and source hashes.
