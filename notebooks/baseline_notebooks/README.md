# Chess Winner Prediction Baselines

![log](/imgs/baseline_log.jpg)  

This part contains baseline models and their implementations for predicting the winner of chess games based on player ratings and game details.

## Content

- [Baseline Structure](#baseline-structure)
- [Dataset Overview](#dataset-overview)
- [Logistic Regression and Naive Elo Model](#logistic-regression-and-naive-elo-model)
  - [Naive Elo Model](#naive-elo-model)
  - [Logistic Regression](#logistic-regression)
- [KNN](#knn)
- [Decision Tree](#decision-tree)
- [Ensembles](#ensembles)
- [Double Stage Model](#double-stage-model)

## Baseline Structure

```plaintext
/root/  
└── chesswinnerprediction/  
    └── baseline/  
        ├── models  
        │   └── naive_model.py  
        │   └── custom_knn.py   
        │   └── double_stage_wrapper.py   
        ├── utils.py 
        └── constants.py

└── notebooks/  
    └── baseline_notebooks/  
        ├── README.md  
        ├── logistic_regression.ipynb  
        ├── decision_tree.ipynb  
        ├── knn.ipynb  
        ├── ensembles.ipynb  
        └── double_stage.ipynb 
```


## Dataset Overview

The dataset includes features related to player ratings and game specifics.   
Below is a preview of the dataset:

| Event                   | EloDiff | MeanElo | WhiteElo | BlackElo | BaseTime | IncrementTime | ZeroIncrementTime |
|-------------------------|---------|---------|----------|----------|----------|---------------|-------------------|
| Rated Bullet tournament | -100    | 1550    | 1500     | 1600     | 60       | 3             | False             |
| Rated Blitz game        | +200    | 1700    | 2200     | 2000     | 120      | 0             | True              |

- **EloDiff**: Calculated as `WhiteElo - BlackElo`
- **Event**: For some models, one-hot encoding of the event type was used
- All numerical features were normalized using `StandardScaler`
---

## Logistic Regression and Naive Elo Model
Notebook: [logistic_regression.ipynb](/notebooks/baseline_notebooks/logistic_regression.ipynb)

Here you can find implementation of two ideas: Naive prediction by elo diff, and logistic regression model.  
The notebook includes the code and performance metrics such as the confusion matrix, precision and recall for each class. 
Additionally, you can find the feature importance analysis.


### Naive Elo Model
Model: [naive_model.py](/chesswinnerprediction/baseline/models/naive_model.py)

This model predicts the winner based on the Elo difference between players. The player with the higher Elo rating is predicted to win:

- **If WhiteElo > BlackElo**: White wins
- **Else**: Black wins

**Naive Solution Accuracy**:
- Excluding draw games, the model achieves **60.09% accuracy** when predicting wins based solely on EloDiff.
- When including draws and using `balanced_accuracy_score`, the model achieves **40.06% balanced accuracy**.


### Logistic Regression

After fitting Logistic Regression model primarily relies on the `EloDiff` feature. 

- **No draws prediction**: Achieves similar results to the naive model, with approximately **60% accuracy**.
- **Including draws**: A multinomial multiclass solution was used (scikit-learn==1.5.1), achieving a **46.6% balanced accuracy score**.

---
## KNN
Notebook: [knn.ipynb](/notebooks/baseline_notebooks/knn.ipynb)  
Model: [custom_knn.py](/chesswinnerprediction/baseline/models/custom_knn.py)

The main challenge with knn was class imbalance (only 2% of games are draws). 
I used dataset resampling to oversample draws and undersample black win and white win games.  
Optuna was used to tune KNN model hyperparameters and number of samples for each class.

KNN achieved a **46% balanced accuracy score**, witch comparable to the logistic regression model.

---
## Decision Tree
Notebook: [decision_tree.ipynb](/notebooks/baseline_notebooks/decision_tree.ipynb)

RandomizedSearchCV from sklearn was used to achieve better performance.  
The best model got a **44.63% balanced accuracy score**.

---
## Ensembles
Notebook: [ensembles.ipynb](/notebooks/baseline_notebooks/ensembles.ipynb)

This notebook covers both random forest and gradient boosting models.  
RandomizedSearchCV from sklearn was used to tune hyperparameters.

- **Random Forest**: Achieved **46.6% balanced accuracy score**.
- **Gradient Boosting**: Achieved **46.93% balanced accuracy score**.

Ensemble models outperformed the decision tree, logistic regression, and KNN models.  
Gradient Boosting achieved the best performance among all models.

---
## Double Stage Model
Notebook: [double_stage.ipynb](/notebooks/baseline_notebooks/double_stage.ipynb)  
Model code path: [double_stage_wrapper.py](/chesswinnerprediction/baseline/models/double_stage_wrapper.py)

The idea is to predict the winner in two stages:

- **Stage 1**: Predict the game result (draw or win) using `win_to_draw_splitter_model`.
- **Stage 2**: Predict the winner (White or Black) using `black_to_white_splitter_model`.

**This implementation supports the following models:**
- Decision Tree
- Random Forest
- Gradient Boosting
- Logistic Regression
- Custom KNN

More details can be found in `double_stage_wrapper.py`.

Optuna was used to tune model types and hyperparameters.

---
## Summary
The best performance was achieved by the Gradient Boosting model, with a **46.93% balanced accuracy score**.

