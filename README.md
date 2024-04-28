# Auto-wikiHow

## Description
Exploring Multimodal LLM to generate or enhance wikiHow.

## Environments
```sh
conda create -n multilingual-WikiHowQA python=3.10
conda activate multilingual-WikiHowQA
pip install -r requirements.txt
```

## Dataset
dataset from: [multilingual-wikihow-qa-16k](https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k)

### raw dataset download
```sh
git lfs install
git clone https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k
```

### dataset split
```sh
python data_splitter.py multilingual-wikihow-qa-16k/data/train-00000-of-00001-0bdf6bc5b4b507e0.parquet 42 ./dataset 0.8 0.1 0.1
```

### Language categories with count
| Language | Training Set | Validation Set | Test Set |
|----------|--------------|----------------|----------|
| English  | 1596         | 199            | 200      |
| Russian  | 1646         | 206            | 206      |
| Portuguese | 1595       | 199            | 200      |
| Dutch    | 1613         | 202            | 202      |
| Italian  | 1768         | 221            | 221      |
| French   | 1724         | 216            | 216      |
| Spanish  | 1672         | 209            | 209      |
| German   | 1841         | 230            | 231      |

### Language and Relations
```
Indo-European Language Family
│
├── Germanic Branch
│   ├── English(en)
│   ├── Dutch(nl)
│   └── German(de)
│
├── Romance Branch
│   ├── Portuguese(pt)
│   ├── Italian(it)
│   ├── French(fr)
│   └── Spanish(es)
│
└── Slavic Branch
    └── Russian(ru)
```

### Language Pair for Eval:
* Intra-lang-branch:
  * en -> nl, en -> de
  * es -> pt, es -> it, es -> fr
* Inter-lang-branch:
  * en -> es, en -> ru
  * es -> en, es -> ru
  * ru -> en, ru -> es
