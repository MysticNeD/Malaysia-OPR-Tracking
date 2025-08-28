"""
Train and compare multiple classifiers on OPR movement prediction (up/same/down)
using the feature generation from prepare_dataset().
"""
from train import prepare_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# -------------------------------
# Step 1: Load data
# -------------------------------
_, Xdf = prepare_dataset()  # reuse your prepare_dataset
features = ["myor_mean", "myor_last", "overnight_mean", "overnight_last",
            "m1_mean", "vol_mean", "vol_sum", "myor_minus_opr",
            "myor_vol_mean", "myor_vol_last", "myor_diff", "overnight_diff"]

X = Xdf[features]
y = Xdf["label"]

# -------------------------------
# Step 1.5: Encode labels for models that require numeric
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'down','same','up' -> 0,1,2

# -------------------------------
# Step 2: Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# Step 2.5: Safe SMOTE
# -------------------------------
counts = Counter(y_train)
min_count = min(counts.values())
if min_count > 1:
    k = min(5, min_count - 1)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_train, y_train = sm.fit_resample(X_train, y_train)
else:
    print("[warn] Not enough samples for SMOTE, skipping oversampling")

# -------------------------------
# Step 3: Define models
# -------------------------------
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=42))
    ])
}

# -------------------------------
# Step 4: Train & evaluate
# -------------------------------
results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 转回原始标签
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)
    
    cr = classification_report(y_test_labels, y_pred_labels)
    print(f"{name} accuracy: {acc:.4f}")
    print(cr)
    results[name] = acc

# -------------------------------
# Step 5: Plot comparison
# -------------------------------
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Comparison on OPR Movement Prediction")
plt.show()
