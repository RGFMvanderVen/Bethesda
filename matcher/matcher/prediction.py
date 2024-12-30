import re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class Prediction:
    """A very minimal and simple pipeline for prediction"""

    def __init__(self, path: str = "results.xlsx"):
        # Load and preprocess data
        self.raw_data = pd.read_excel(path)
        self.data = self._preprocess_data()

        # Split data
        self.labeled_data = self.data.loc[
            (~self.data.Bethesda.isnull()) & (self.data.Bethesda != 0), :
        ]
        self.unlabeled_data = self.data.loc[self.data.Bethesda.isnull(), :]

    def split_data(self):
        """Split data into train, test and eval"""
        X, y = (
            self.labeled_data.Conclusie.values.tolist(),
            self.labeled_data.Bethesda.values.tolist(),
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, train_size=0.5
        )
        X_eval, X_test, y_eval, y_test = train_test_split(
            X_test, y_test, random_state=0, train_size=0.5
        )
        print(f"Training size: {len(X_train),len(y_train)}")
        print(f"Test size: {len(X_test),len(y_test)}")
        print(f"Eval size: {len(X_eval),len(y_eval)}")
        return X, y, X_train, y_train, X_test, y_test, X_eval, y_eval

    def _preprocess_data(self):
        """Preprocess data"""
        self.data = self.raw_data.copy()
        names = ["Bethesda", "Betehesda"]
        for name in names:
            self.data.Conclusie = self.data.Conclusie.apply(
                lambda row: row.split(name)[0], 1
            )
        self.data.Conclusie = (
            self.data.Conclusie.str.replace("_x000D_", " ")
            .str.replace("\n", " ")
            .str.strip()
        )
        self.data.Conclusie = self.data.Conclusie.apply(
            lambda row: re.sub(" +", " ", row), 1
        )
        return self.data

    def evaluate(
        self,
        pipes,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ):
        """Evaluate multiple pipelines using a 5-fold cross-validation
        and also derive a classification report for specifics on the performance of the model.


        Arguments:
            pipes: A list of scikit learn pipelines
            X: A numpy array of features
            y: A numpy array of labels
            X_test: A numpy array of features for testing
            y_test: A numpy array of labels for testing
            verbose: A boolean to print the results of the evaluation

        Returns:
            cv_scores: A list of cross-validation scores
            reports: A list of classification reports
        """
        cv_scores = []
        reports = []
        displays = []
        for pipe in pipes:
            cv_score, report, disp = self.evaluate_pipeline(
                pipe, X, y, X_test, y_test, verbose
            )
            cv_scores.append(cv_score)
            reports.append(report)
            displays.append(disp)

        return cv_scores, reports, displays

    def evaluate_pipeline(
        self,
        pipe,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ):
        """Evaluate a single pipeline using a 5-fold cross-validation and also derive a classification
        report for specifics on the performance of the model.

        Arguments:
            pipe: A scikit learn pipeline
            X: A numpy array of features
            y: A numpy array of labels
            X_test: A numpy array of features for testing
            y_test: A numpy array of labels for testing
            verbose: A boolean to print the results of the evaluation

        Returns:
            cv_score: A list of cross-validation scores
            report: A classification report
        """
        clf_name = pipe.steps[-1][-1].__class__.__name__
        X_train, y_train = X, y
        if "xgb" in clf_name.lower():
            y_train = [int(val - 1) for val in y]
            y_test = [int(val - 1) for val in y_test]

        # Cross-validation
        f1_macro = cross_val_score(
            pipe, X_train, y_train, cv=5, verbose=1, n_jobs=-1, scoring="f1_macro"
        )
        f1_micro = cross_val_score(
            pipe, X_train, y_train, cv=5, verbose=1, n_jobs=-1, scoring="f1_micro"
        )
        cv_scores = {"f1_macro": f1_macro, "f1_micro": f1_micro}

        # Classification reprot
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5, 6]
        )

        if verbose:
            print(f"### {clf_name} ###")
            print(
                f"F1 macro scores: {f1_macro} \n F1 micro scores: {f1_micro} \n\n {report}\n"
            )

        return cv_scores, report, disp

    def add_predictions(self, pipe, X, y):
        """Add predictions to the data"""
        # Prepare data
        updated_data = self.raw_data.copy()
        updated_data["Bethesda_Predict"] = np.nan
        updated_data["Bethesda_Predict_Probability"] = np.nan
        to_predict = updated_data.loc[updated_data.Bethesda.isnull(), "Conclusie"]

        # Fit pipeline and predict
        pipe.fit(X, y)
        bethesda_predicted = pipe.predict(to_predict)
        bethesda_proba = pipe.predict_proba(to_predict)

        # Add predictions and probs
        updated_data.loc[updated_data.Bethesda.isnull(), "Bethesda_Predict"] = (
            bethesda_predicted
        )
        updated_data.loc[
            updated_data.Bethesda.isnull(), "Bethesda_Predict_Probability"
        ] = np.max(bethesda_proba, 1)
        updated_data = updated_data.drop("Unnamed: 0", 1)
        return updated_data

    def create_pipeline(self, clf):
        """Create training pipeline"""
        vectorizer = TfidfVectorizer()
        pipe = Pipeline([("vectorizer", vectorizer), ("clf", clf)])
        return pipe
