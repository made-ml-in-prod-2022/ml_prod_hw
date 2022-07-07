# ML Project
---
[![build status](https://github.com/made-ml-in-prod-2022/ml_prod_hw/actions/workflows/main.yml/badge.svg)](https://github.com/made-ml-in-prod-2022/ml_prod_hw/actions/workflows/main.yml)

__Installation:__
Start your venv.
Then install all dependencies with:
```
pip install -r requirements.txt
```
__Usage:__
* Start train pipeline:
```
python projectname/train_pipeline.py
```
* Start predict pipeline:
```
python projectname/predict_pipeline.py predict_params.path_to_data={your data} predict_params.path_to_model={model to use} predict_params.output_path={to store results}
```
---
# Project Organization
```bash
.
├── README.md
├── configs
│   └── config.yaml
├── .github
│   └── workflows
│       └── main.yml
├── models
├── notebooks
│   └── hw1_prod.ipynb
├── projectname
│   ├── all_dataclasses
│   │   ├── __init__.py
│   │   ├── pathes_params.py
│   │   ├── predict_params.py
│   │   ├── splitting_params.py
│   │   ├── train_predict_pipeline_params.py
│   │   └── training_params.py
│   ├── build_ds
│   │   ├── __init__.py
│   │   └── build_dataset.py
│   ├── data
│   ├── logs
│   ├── model_code
│   │   ├── __init__.py
│   │   └── train_predict_eval.py
│   ├── predict_pipeline.py
│   └── train_pipeline.py
└── requirements.txt
```