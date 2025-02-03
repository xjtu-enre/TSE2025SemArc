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
This module utilizes Large Language Models (LLMs) to identify both code semantics and architectural semantics within the project. Before use this you should setup **API_KEY** and choose **LLM_MODEL** in **config.py**. The analysis results are automatically saved into two JSON files, which are named after the project as follows:

- **[project name]_ArchSem.json**: This file contains the architectural semantics identified in the project.
- **[project name]_CodeSem.json**: This file contains the code semantics identified in the project.

### Usage
To run the semantic analysis, use the following command:

```bash
python semantic_analysis.py [project folder]
```

## SemArc
### Usage
```bash
python SemArc.py [-h] [-g  [...]] [-o] [--cache_dir] [-s  [...]] [-a  [...]] [-c  [...]] [-r] [-n] datapath [datapath ...]
```
positional arguments:
````
  datapath              path to the input project folder

options:
  -h, --help            show this help message and exit
  -g  [ ...], --gt  [ ...]
                        path to the ground truth json file
  -o , --out_dir        path to the result folder
  --cache_dir           cache path
  -s  [ ...], --stopword_file  [ ...]
                        paths to external stopword lists
  -a  [ ...], --archsem_file  [ ...]
                        paths to architecture semantic file
  -c  [ ...], --codesem_file  [ ...]
                        paths to code semantic file
  -r , --resolution     resolution parameter, affecting the final cluster size.
  -n, --no_fig          prevent figure generation
````
### Example

We have provided a **demo** folder that contains the source code, ground-truth architecture and semantic files of **bash-4.2**. You can run the architecture recovery process on this example project using the following command:

```bash
python .\SemArc.py .\demo\bash-4.2 -s .\stopwords.txt -a .\semantic_analysis\bash-4.2_ArchSem.json -c .\semantic_analysis\bash-4.2_CodeSem.json -g .\demo\bash-4.2-GT.json
```

