# 🚢 Titanic Survival Prediction Using Decision Tree

A Machine Learning classification project that predicts whether a passenger survived the **Titanic disaster** using a **Decision Tree Classifier**.

The project demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, outlier handling, feature scaling, model training, and hyperparameter optimization.

## 🎯 Project Objective

The objective is to predict the `Survived` target variable:

* `0` → Did not survive
* `1` → Survived

The project uses passenger information such as age, gender, family information, fare, and embarkation point to build the predictive model.

## 📊 Dataset

The project uses the **Titanic: Machine Learning from Disaster** dataset from Kaggle.

### Features

The original dataset contains:

* `PassengerId`
* `Name`
* `Sex`
* `Age`
* `SibSp`
* `Parch`
* `Ticket`
* `Fare`
* `Cabin`
* `Embarked`

### Target

```text
Survived
```

## 🔄 Machine Learning Workflow

```text
Titanic Dataset
       ↓
Data Exploration
       ↓
Data Preprocessing
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Outlier Treatment
       ↓
Feature Scaling
       ↓
Decision Tree
       ↓
GridSearchCV
       ↓
Model Evaluation
```

## 🧹 Data Preprocessing

Several preprocessing techniques were applied to prepare the dataset.

### Categorical Encoding

`Sex` and `Embarked` were converted into numerical values using **LabelEncoder**.

### Irrelevant Features

The following columns were removed:

```text
PassengerId
Name
Ticket
Cabin
```

### Missing Values

Missing `Age` values were handled using the mean.

## ⚙️ Feature Engineering

Two additional features were created:

### FamilySize

```python
FamilySize = SibSp + Parch + 1
```

This represents the total number of family members traveling with the passenger.

### IsAlone

Passengers were categorized based on whether they were traveling alone:

```text
FamilySize = 1 → IsAlone = 1
FamilySize > 1 → IsAlone = 0
```

These engineered features help the model capture additional relationships within the passenger data.

## 📈 Feature Transformation

### Fare Transformation

The `Fare` feature was log-transformed to reduce skewness.

### Outlier Handling

Outliers in `Fare` were capped using the **IQR method**.

### Feature Scaling

`StandardScaler` was applied to numeric features.

## 🌳 Model

The primary algorithm used is:

**Decision Tree Classifier**

Decision Trees recursively split the dataset based on feature values to make classification decisions.

## 🔧 Hyperparameter Tuning

`GridSearchCV` was used to search for an effective combination of Decision Tree hyperparameters.

The parameters explored included:

```python
criterion = ['gini', 'entropy', 'log_loss']

max_depth = [None, 3, 5, 7, 10]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

max_features = [None, 'sqrt', 'log2']
```

### Best Hyperparameters

The repository's current experiment identifies:

```text
criterion        = entropy
max_depth        = 7
max_features     = sqrt
min_samples_leaf = 4
min_samples_split = 5
```

## 📌 Key Observations

* Feature engineering with **FamilySize** and **IsAlone** provides additional predictive information.
* Log transformation helps reduce the skewness of `Fare`.
* IQR-based capping helps control the effect of extreme fare values.
* Hyperparameter tuning provides a more optimized Decision Tree configuration.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## 📂 Project Structure

```text
Decision_tree/
│
├── Decision_tree(titanic).ipynb
├── README.md
└── ...
```

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/Rj-Ahsan/Decision_tree.git
cd Decision_tree
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the notebook

```bash
jupyter notebook "Decision_tree(titanic).ipynb"
```

Run the notebook cells sequentially to reproduce the preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

## 🔮 Future Improvements

* Compare Decision Tree with Random Forest and Gradient Boosting
* Add confusion matrix visualization
* Evaluate precision, recall, and F1-score
* Perform feature importance analysis
* Add cross-validation comparisons
* Deploy the trained model as an API or web application

## 👨‍💻 Author

**Ahsan Tanveer**

BS Artificial Intelligence | AI/ML Engineer

GitHub: [@Rj-Ahsan](https://github.com/Rj-Ahsan)

---

⭐ If you find this project useful, consider giving the repository a star!
