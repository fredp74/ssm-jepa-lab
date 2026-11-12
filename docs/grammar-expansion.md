# Grammar expansion: finite-domain control

## Result

The symbolic agent progressively expands a programmed hypothesis family
when its observations contradict every current candidate.

| Required family | Verified functions |
|---|---:|
| Initial grammar | 28 |
| Degree-two extension | 100 |
| Degree-three extension | 128 |
| Total | 256 |

All 256 functions were queried on all eight inputs: 2,048 questions.
The saved polynomial representations were reloaded and checked.

All 128 expansions to degree three occurred after the eighth observation.
These cases reconstruct a fully observed function; they do not demonstrate
prediction of previously unobserved behavior.

## What this establishes

- Contradictions trigger expansion.
- Observations remain constraints after expansion.
- Verified rules can be stored as executable representations.

## What this does not establish

- The extensions are programmed, not invented.
- The final family covers every function in this tiny domain.
- There is no query saving in this experiment.
- Exhaustive verification does not scale to large domains.
- Results assume a deterministic environment without noise.
- No SSM or JEPA mechanism is involved.

## Next protocol: budgeted generalization

Move to five input bits, giving 32 possible inputs.

Agent budget:
- At most 12 discovery questions.
- At most four additional verification questions.
- No access to evaluator-only answers.
- Freeze the proposed rule before final evaluation.

Evaluate on every input not queried by the agent. These evaluator calls
are separate from the agent's interaction budget.

Test both functions inside and outside the declared hypothesis family.
Report abstentions and contradictions separately from accepted hypotheses.
A hypothesis surviving four checks remains provisional.

Primary measurements:
- Exact accuracy on unqueried inputs.
- Acceptance coverage.
- False acceptance rate for out-of-family functions.
- Number of discovery and verification questions.

This will test budgeted generalization, not autonomous invention of a
rule language. Hypothesis families and test generation must be recorded
before running the evaluation.

## Local artifacts

Detailed results are under autonomous_lab/runs/grammar_expansion_01/.
Run directories and model artifacts remain excluded from Git.
