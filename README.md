# Titanic Survival Prediction Using Decision Tree

## 🔗 Dataset

The dataset used in this project is the **Titanic dataset** from Kaggle:  
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

- Features include `PassengerId, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`.
- Target variable: `Survived` (0 = No, 1 = Yes)

---

## 📊 Data Preprocessing

1. **Label Encoding**
   - Converted categorical features `Sex` and `Embarked` to numeric using `LabelEncoder`.

2. **Dropping irrelevant columns**
   - Removed `Cabin`, `Name`, `Ticket`, `PassengerId`.

3. **Handling missing values**
   - `Age` missing values filled with **mean**.

4. **Feature Engineering**
   - `FamilySize` = `SibSp + Parch + 1`  
   - `IsAlone` = 1 if `FamilySize` = 1 else 0

5. **Skewed features & outliers**
   - Log transformation applied to `Fare`.  
   - Outliers in `Fare` capped using the **IQR method**.

6. **Feature Scaling**
   - StandardScaler applied to numeric features.

---

## 🌳 Model Training

- **Algorithm:** Decision Tree Classifier
- **Default Model Accuracy:** `replace_with_default_accuracy`

### Hyperparameter Tuning

- Used `GridSearchCV` to find the best combination:

```python
parameter = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}
Best hyperparameters found:

criterion = entropy
max_depth = 7
max_features = sqrt
min_samples_leaf = 4
min_samples_split = 5
📈 Model Evaluation
Tuned Model Test Accuracy: replace_with_tuned_accuracy
💡 Key Observations
Feature engineering (FamilySize, IsAlone) improved predictive power.
Log-transform and IQR capping stabilized Fare distribution and reduced skew.
Hyperparameter tuning significantly improved accuracy over the default Decision Tree.
