# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ========== 1) Data Preprocessing ==========
data = pd.read_csv("train.csv")
print(data.shape)

# Extract Title
data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Create FamilySize
data['FamilySize'] = data['SibSp'] + data['Parch']

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Title', 'Embarked'], drop_first=True)

# Drop irrelevant columns
data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Standardize features
scaler = StandardScaler()
data[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(data[['Age', 'Fare', 'FamilySize']])

# ========== 2) Data Splitting ==========
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split into Training (80%) and Evaluation (20%)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Evaluation Function ==========
def evaluate_model(name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_eval, y_pred), 4))
    print("Precision:", round(precision_score(y_eval, y_pred), 4))
    print("Recall:", round(recall_score(y_eval, y_pred), 4))
    print("F1 Score:", round(f1_score(y_eval, y_pred), 4))
    print("Classification Report:\n", classification_report(y_eval, y_pred, digits=4))

# ========== 3 & 4) Model Implementation and Training ==========

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# ========== 5) Model Tuning (GridSearchCV on Evaluation Set) ==========

print("\n--- Grid Search Tuning: SVM ---")
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_svm.fit(X_eval, y_eval)
best_svm = grid_svm.best_estimator_


print("Best Parameters for SVM:", grid_svm.best_params_)

# ========== 6) Model Evaluation ==========
evaluate_model("Logistic Regression", logreg, X_eval, y_eval)
evaluate_model("Tuned SVM (GridSearchCV)", best_svm, X_eval, y_eval)
evaluate_model("Random Forest", rf, X_eval, y_eval)