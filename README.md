# On the limited utility of parallel data for learning shared multilingual representations
This repository contains the source code for running the evaluation experiments of the article "On the limited utility of parallel data for learning shared multilingual representations".

In the study, we train four transformer language models of 1.4 billion parameters on multilingual corpora with varying proportions of parallel data: 0%, 1%, 2%, and 5%. We then evaluate the level of cross-lingual alignment in the models using principal component projections, projection weighted canonical correlation analysis (PWCCA), language-specific neuron analysis, and cross-lingual control vectors.

## Notes
The source code for training the language models is based on [the source code for training the OLMo models](https://github.com/allenai/OLMo) by AI2 and modified for the training setup and infrastructure of this study. The models are trained using the LUMI HPC cluster, powered by AMD Instinct™ MI250X accelerators.

The PWCCA code in the evaluation experiments is based on [the source code of article "Insights on representational similarity in neural networks with canonical correlation"](https://github.com/google/svcca).

The cross-lingual control vector code in the evaluation experiments is based on [the source code of article "Controlling Language and Style of Multi-lingual Generative Language Models with Control Vectors"](https://github.com/shiftleino/crosslingual-control-vectors).

## Acknowledgements
The research was supported by the Technology Industries of Finland Centennial Foundation.
