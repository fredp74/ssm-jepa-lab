# V13 experiment: repeated-character training

## Question

Does adding consecutive repeated characters to copy training improve
accuracy beyond an equal amount of ordinary training?

## Protocol

All runs started from assistant_longcopy_v10.pt.

- Two branches: ordinary-copy control and repetition-focused training.
- Three seeds: 130013, 130014, 130015.
- 200 training steps per run.
- Learning rate: 0.00005.
- Replay weight: 0.3.
- Other instruction tasks remained in both branches.
- In the repetition branch, approximately half of copy examples of
  length 6 or greater were generated with exactly one adjacent duplicate.

The branches used the same training budget and seed values, but their
generated training examples diverged.

## Final evaluation

The final datasets were generated after training all six runs.
Recorded training targets and available earlier dataset values were
excluded during generation.

Instruction templates remained familiar. This evaluates new strings
under existing templates, not generalization to unseen instructions.

The general set contains 208 new-code cases and 20 reused missing-code
cases. Every model passed the 20 reused cases; they are excluded below.

| Model | New-code cases /208 | Copy diagnostic /288 |
|---|---:|---:|
| Initial v10 | 201 | 265 |
| Control, seed 130013 | 201 | 266 |
| Repetition, seed 130013 | 201 | 271 |
| Control, seed 130014 | 200 | 271 |
| Repetition, seed 130014 | 201 | 271 |
| Control, seed 130015 | 203 | 271 |
| Repetition, seed 130015 | 199 | 268 |

A successful case requires exact answers and end markers at every turn.

## Results across seeds

| Method | Mean new-code score /208 | Mean copy score /288 |
|---|---:|---:|
| Control | 201.33 | 269.33 |
| Repetition | 200.33 | 270.00 |

Repetition-focused training did not show a consistent overall advantage.
Its small average gain on the copy diagnostic came with a small average
loss on the general set.

The repetition run with seed 130015 had led on the earlier general
validation set, but did not retain that lead on the final set.

## Retention

Mean language-model scores after training, measured on 200 samples
per domain. Lower bits per byte is better.

| Method | A | B | C |
|---|---:|---:|---:|
| Initial v10 | 2.23297 | 1.97020 | 2.37003 |
| Control, mean of three seeds | 2.22766 | 1.96621 | 2.35913 |
| Repetition, mean of three seeds | 2.22808 | 1.96590 | 2.35937 |

Both methods slightly improved retention scores relative to v10.
Differences between the methods were small.

## Conclusion

This experiment does not establish repetition-focused training as
a superior method.

Control seed 130015 had the best observed balance on these final sets.
This ranking is descriptive; selecting a checkpoint using final scores
does not establish its superiority on future data.

## Limitations

- Only three training seeds.
- Small synthetic datasets and familiar instruction templates.
- The same evaluation cases were shared across runs, so scores must
  not be interpreted as independent new test examples for each seed.
- No statistical significance claim is made.
- Exclusion checks rely on recorded values, not a complete history of
  every string ever encountered.
- Final evaluation datasets must remain excluded from future training.

## Artifacts

- build/experiment_v13_control.py
- build/experiment_v13_repetition.py
- build/make_experiment_v13.py
- build/prepare_repetition_test.py
- build/make_final_evaluation.py
- build/prepare_final_general.py
- build/prepare_final_repetitions.py
- build/final_evaluation/general.jsonl
- build/final_evaluation/repetitions.jsonl

Checkpoints remain local and are excluded from Git.
