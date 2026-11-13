# CSV composition: persistence and verification controls

Targeted verification inputs rejected incomplete programs that had
passed earlier checks. The composed repertoire retained 37.5% answer
coverage in v3, with zero observed errors among its answers.

This remains a development study within a predefined program language.

## Retrospective audit

The initial pilot's persistence field compared two reads of the same
file. A separate audit performed a new serialization of existing
memories and compared payloads and prediction digests before saving
and after reloading.

All 48 of
48 memories passed that new roundtrip.
This does not retroactively validate the original persistence check.

The audit also reused existing verification responses to filter
candidates in the 16 previously ambiguous composed environments.
Diagnostic consensus answers increased from
1,141 to
2,252 out of
3,200, with
0 observed errors after
filtering and no new tool calls.

These were retrospective measurements on reused evaluation tables.
The original agent had abstained on those environments.

## Revised procedure

Version 2 separates observations used for candidate filtering from
subsequent verification. Candidates are frozen before verification.

The resource allocation is:

- Catalogue discovery: up to four calls, eight cells and 256 bytes.
- Two learning observations: eight cells and up to 128 bytes.
- Two verification calls: eight cells and up to 128 bytes.

The total ceiling remains eight calls, 24 cells and 512 input-text
UTF-8 bytes. This reallocates the original discovery budget, so changes
cannot be attributed to filtering alone.

A provisional memory may answer only when every retained program agrees.
Verification disagreement causes abstention; incompatible responses
revoke the proposal.

Versions 2 and 3 compare payloads and predictions before saving and
after reloading, using catalogue and verification inputs. Evaluation
predictions are then computed from reloaded memories.

## Verification redesign

Version 2 accepted incomplete single-operation programs in environments
that required a transformation followed by row sorting. Its verification
inputs did not adequately distinguish the missing sorting behavior.

Version 3 changes the verification inputs to include:

- Unsorted rows.
- Values that collide after truncation.
- Unicode normalization variants.
- Whitespace-only rows and a leading-zero value.

These inputs were deliberately designed after examining v2 failures.
They are development-informed controls, not evidence of a general
verification guarantee.

## Results

Each version evaluates 200 tables in each of the same 24 development
environments. Evaluation tables differ between versions.

| Version | Candidate language | Provisional environments | Answers | Correct | Errors | Coverage |
|---|---|---:|---:|---:|---:|---:|
| v2 | At most one operation | 11/24 | 2,200 | 1,294 | 906 | 45.8% |
| v2 | At most two operations | 9/24 | 1,800 | 1,800 | 0 | 37.5% |
| v3 | At most one operation | 3/24 | 600 | 600 | 0 | 12.5% |
| v3 | At most two operations | 9/24 | 1,800 | 1,800 | 0 | 37.5% |

Each arm has 4,800 evaluation questions per version.

In v3, all eight single-arm environments requiring a transformation
followed by sorting were revoked. The 906 errors observed in v2 did not
recur on v3's new tables. This is not a question-by-question comparison.

The composed arm retained nine provisional environments in both
versions. Its other 15 environments remained verification-ambiguous.

Its successful environments cover casefold followed by deduplication,
rstrip followed by empty-row removal, and one CSV configuration for
Unicode normalization followed by rstrip.

## Resources and persistence

The composed arm used 120 calls and 576 cells in each version:
five calls and 24 cells per environment. The cell budget was exhausted
before the call ceiling.

Input-text consumption was
3,864 bytes in v2 and
4,200 bytes in v3.

Corrected persistence checks passed for
48/48 memories
in v2 and 48/48
in v3.

## Interpretation and limitations

Verification input quality affects whether incomplete programs are
accepted. More candidate programs alone do not fix inadequate checks.

Zero observed errors does not prove general correctness. In the
composed arm, the true simulated program is included in the candidate
language. The single-operation arm may not represent the true behavior.

The environments and elementary operations are predefined and reused.
There is no neural training, no invention of new primitive operations,
and no demonstrated SSM/JEPA advantage.

Evaluation tables are new between versions, but the same tables are
shared across related environments. The 4,800 questions are not
independent experimental replications.

The next comparison should freeze the procedure before evaluation,
rather than repeatedly tuning checks to these six compositions.

## Reproduction

After the initial composition pilot:

1. python autonomous_lab/audit_csv_compositions.py
2. python autonomous_lab/make_csv_compositions_v2.py
3. python autonomous_lab/acquire_csv_compositions_v2.py
4. python autonomous_lab/make_csv_compositions_v3.py
5. python autonomous_lab/acquire_csv_compositions_v3.py
6. python autonomous_lab/summarize_csv_composition_controls.py

Generators and experiments refuse to overwrite their destinations.
Raw results remain local under autonomous_lab/runs.
The companion JSON records source hashes, protocols and summaries.
