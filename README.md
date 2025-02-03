# TSE2025SemArc
This repository contains dataset for the paper "Software Architecture Recovery Augmented with Semantics".
The folders is organized as follows:
````
└─ground-truth
    ├─collected
    └─labeled
└─semantic_analysis
└─SemArc
````
## ground-truth
The dataset consists of eight ground-truth datasets sourced from existing research and seven additional datasets created by us. These datasets cover three different programming languages and vary in sizes, providing a comprehensive set of real-world examples for software architecture recovery tasks.
- **collected**: This folder contains datasets sourced from existing research, providing a reliable foundation for evaluating architecture recovery methods.
- **labeled**: This folder includes the datasets created by us, where the ground-truth architecture has been manually labeled and validated. These datasets are designed to cover a wide range of system sizes and programming languages, ensuring a diverse set of examples.

## semantic_analysis
This module utilizes Large Language Models (LLMs) to identify both code semantics and architectural semantics within the project. The analysis results are automatically saved into two JSON files, which are named after the project as follows:

- **project_name_ArchSem.json**: This file contains the architectural semantics identified in the project.
- **project_name_CodeSem.json**: This file contains the code semantics identified in the project.

### Usage
To run the semantic analysis, use the following command:

```bash
python semantic_analysis.py [project folder]
