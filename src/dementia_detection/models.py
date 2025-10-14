"""Machine learning models for dementia detection."""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def get_models() -> dict:
    """Return dictionary of ML models for dementia detection."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def get_tuned_models(X_train, y_train) -> dict:
    """Return tuned models using GridSearchCV."""
    # Logistic Regression tuning
    lr_params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
    lr = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_params,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    # Random Forest tuning
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42), rf_params, cv=3, scoring="f1", n_jobs=-1
    )

    # SVM tuning
    svm_params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]}
    svm = GridSearchCV(
        SVC(probability=True, random_state=42), svm_params, cv=3, scoring="f1", n_jobs=-1
    )

    # Fit all models
    print("Tuning Logistic Regression...")
    lr.fit(X_train, y_train)
    print("Tuning Random Forest...")
    rf.fit(X_train, y_train)
    print("Tuning SVM...")
    svm.fit(X_train, y_train)

    return {
        "Logistic Regression (Tuned)": lr.best_estimator_,
        "Random Forest (Tuned)": rf.best_estimator_,
        "SVM (Tuned)": svm.best_estimator_,
    }


def create_ensemble(models: dict) -> VotingClassifier:
    """Create voting ensemble from trained models."""
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting="soft")


def train_and_evaluate(model, X_train, y_train, X_val, y_val, fit: bool = True) -> dict:
    """Train model and return evaluation metrics."""
    if fit:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, target_names=["Control", "Dementia"]),
    }
