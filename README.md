# ChessWinnerPrediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

<a target="_blank" href="https://mlflow.org/">
    <img src="https://img.shields.io/badge/Mlflow-Tracking-0194E1?logo=mlflow" />
</a>

<a target="_blank" href="https://optuna.org/">
    <img src="https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-48D1CC?logo=optuna" />
</a>

<a target="_blank" href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-Web%20app%20framework-FF4B4B?logo=streamlit" />
</a>

<a target="_blank" href="https://github.com/niklasf/python-chess">
    <img src="https://img.shields.io/badge/Chess-Python%20Library-000000?logo=python" />
</a>

<a target="_blank" href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn" />
</a>

<a target="_blank" href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-Data%20Manipulation-150B33?logo=pandas" />
</a>

<a target="_blank" href="https://numpy.org/">
    <img src="https://img.shields.io/badge/Numpy-Scientific%20Computing-013243?logo=numpy" />
</a>

<a target="_blank" href="https://plotly.com/">
    <img src="https://img.shields.io/badge/Plotly-Data%20Visualization-3F4F75?logo=plotly" />
</a>

<a target="_blank" href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/Seaborn-Data%20Visualization-3776AB?logo=seaborn" />
</a>

<a target="_blank" href="https://pypi.org/project/zstandard/">
    <img src="https://img.shields.io/badge/Zstandard-Compression-FF9933?logo=python" />
</a>

<a target="_blank" href="https://jupyter.org/">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter" />
</a>

---

## Installation and Running a Demo

To get started with the Chess Winner Prediction project, follow the steps below:

#### Clone the Repository

```bash
git clone https://github.com/Tikhon-Radkevich/Chess-Winner-Prediction.git
cd Chess-Winner-Prediction
```

#### Set Up the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

#### Run the Demo

```bash
streamlit run ./demo/main.py
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for chesswinnerprediction
│                         and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── chesswinnerprediction                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes chesswinnerprediction a Python module
    │
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

