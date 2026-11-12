# Symbolic consolidation and changing environments

## Scope

This experiment tests persistent reuse of interaction history.
It is symbolic: no SSM, JEPA objective, or neural training is involved.

The agent knows a programmed grammar of 28 three-bit Boolean functions.
It asks questions until exactly one compatible rule remains.
The resulting rule signature and encounter count are stored in a library.

Verification is exact only within the assumed grammar. It does not
establish correctness for arbitrary functions outside that grammar.

## Strategies

- Cold: discard encounter history between episodes.
- Cumulative: weight each candidate by one plus its previous encounter count.
- Discounted: multiply encounter counts by 0.95 before each episode,
  then update the identified rule.

The pseudocount of one keeps every candidate available.
Discounting changes search priorities; it does not remove rules.

## Stationary pilot

One 60-episode sequence favored three recurring rules.

- Cold agent: 296 questions.
- Persistent agent: 252 questions.
- Saving: 44 questions, or 14.9%.

Repeated encounters saved 47 questions; first encounters cost three extra.
The library was saved and reloaded between episodes.

## Changing-environment experiment

Five sequences contain 60 episodes each. After episode 30, the three
frequent rules change to three different rules. The switch is not announced
to the agent. With probability 0.8 the environment selects a frequent rule;
otherwise it samples uniformly from the entire grammar.

All strategies receive the same environment sequence for each seed.

| Strategy | Questions over 300 episodes | Saving against cold |
|---|---:|---:|
| cold | 1477 | 0.00% |
| cumulative | 1355 | 8.26% |
| discounted | 1344 | 9.00% |

Every inferred rule was correct, and all unqueried outputs were correct.

### Questions by sequence

| Seed | Cold | Cumulative | Discounted |
|---|---:|---:|---:|
| 72001 | 297 | 284 | 278 |
| 72002 | 290 | 267 | 266 |
| 72003 | 299 | 266 | 263 |
| 72004 | 299 | 273 | 278 |
| 72005 | 292 | 265 | 259 |

Discounting saved 11 questions relative to cumulative history overall,
but performed worse on one of the five sequences.

### Adaptation after the switch

| Period | Cold mean questions | Cumulative | Discounted |
|---|---:|---:|---:|
| Episodes 31-40 | 4.94 | 4.76 | 4.72 |
| Episodes 41-60 | 4.90 | 4.46 | 4.44 |

## Interpretation

Persistent history reduced interaction cost on these sequences.
Discounting added a small aggregate benefit without consistent dominance.

This demonstrates adaptation of question-selection priorities within a
known grammar, not autonomous discovery of a new rule language.

## Limitations

- Small programmed grammar and synthetic recurrence patterns.
- Five changing-environment sequences and one stationary pilot.
- No statistical significance claim.
- Unique compatibility within the grammar is not proof outside it.
- No claim of neural or language-model capability.

## Decision and next question

Keep cumulative history as the simple reference strategy.
Preserve discounting as an experimental alternative.

Before expanding the grammar, test how reliably the agent detects
observations incompatible with its current hypotheses. An out-of-grammar
function may remain indistinguishable from a known rule after only a
few questions, so detection cannot be assumed to be guaranteed.

## Artifacts

Scripts are under autonomous_lab/. Detailed runs and libraries remain
local under autonomous_lab/runs/, which is excluded from Git.

The companion JSON contains configurations, source hashes, aggregate
scores, and per-sequence question counts.
