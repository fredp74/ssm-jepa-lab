# Contributing

This repository develops a language model based on selective SSM layers
and a JEPA-inspired latent objective.

Contributions should focus on the model, training and recovery,
language-data preparation, recurrent inference, continual learning
or meaningful language evaluation.

Before submitting a change:

1. Run python -m unittest discover -s tests -v.
2. Describe the problem, implementation and validation.
3. Include reproduction commands and seeds for experimental results.
4. Report actual data, compute and checkpoint requirements.
5. Distinguish improved prediction loss from coherent generation.
6. Document failures and limitations.

Keep changes focused. Discuss substantial architectural changes in an
issue before investing in a large experiment.

Do not commit model weights, downloaded corpora, generated datasets,
local runs or credentials. Use only data you are entitled to use.

Use English for shared code documentation and reports.
Do not add unrelated benchmark families or symbolic-agent experiments.

Check the repository's current license before reuse or redistribution.
This guide does not grant additional license rights.
