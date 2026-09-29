import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("german_credit_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2. SELECT TARGET COLUMN
# ==========================================

TARGET_COLUMN = "Risk"  # CHANGE THIS

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column not found. Available columns: {df.columns.tolist()}"
    )

# Remove rows where target is missing
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Separate input features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ==========================================
# 3. DATA CLEANING
# ==========================================

# Remove duplicate rows
combined = pd.concat([X, y], axis=1)
combined = combined.drop_duplicates()

X = combined.drop(columns=[TARGET_COLUMN])
y = combined[TARGET_COLUMN]

# Remove columns with no useful variation
X = X.loc[:, X.nunique(dropna=False) > 1]

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str))

print("\nTarget Classes:")
print(label_encoder.classes_)

print("\nFeature Columns:")
print(X.columns.tolist())

# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical Features:", numerical_cols)
print("\nCategorical Features:", categorical_cols)

# ==========================================
# 5. PREPROCESSING
# ==========================================

# Numerical:
# Fill missing values with median
# Scale numerical features

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical:
# Fill missing values with most frequent value
# Convert categories into numerical representation

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# 7. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []
trained_models = {}

# ==========================================
# 8. TRAIN AND EVALUATE
# ==========================================

for name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.astype(str),
        zero_division=0
    ))

# ==========================================
# 9. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# Save comparison results
results_df.to_csv("model_comparison.csv", index=False)

# ==========================================
# 10. PERFORMANCE GRAPH
# ==========================================

results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Credit Scoring Model Comparison")
plt.ylabel("Performance Score")
plt.xlabel("Machine Learning Algorithm")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

# Demonstration using Random Forest
model = trained_models["Random Forest"]

y_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================
# 12. ROC-AUC (BINARY CLASSIFICATION)
# ==========================================

if len(label_encoder.classes_) == 2:

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print("\nROC-AUC Score:", round(auc, 4))

# ==========================================
# 13. NEW APPLICANT PREDICTION
# ==========================================

# Demonstration using one existing test record.
# Replace this with actual applicant input later.

new_applicant = X_test.iloc[[0]]

prediction = model.predict(new_applicant)[0]

predicted_class = label_encoder.inverse_transform(
    [prediction]
)[0]

print("\nNEW APPLICANT PREDICTION")
print("Predicted Credit Category:", predicted_class)

print("\nApplicant Details:")
print(new_applicant)