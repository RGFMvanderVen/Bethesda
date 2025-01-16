"""
Pipeline for extracting Bethesda score from PALGA Conclusie texts.
Essentially the same code as in `/notebooks`

Usage:
    python main.py --path "my_data.csv" --output "my_results.csv"
"""

import argparse
import pandas as pd

from rapidfuzz import fuzz
from polyfuzz.models import RapidFuzz
from sklearn.linear_model import LogisticRegression

from matcher import MatchEntity, EntityCollection, Prediction
from matcher import preprocess


parser = argparse.ArgumentParser(description="Bethesda")
parser.add_argument("-p", "--path", dest="path", help="Path to the data", type=str)
parser.add_argument(
    "-o", "--output", dest="output", help="Save path", type=str, default="results.csv"
)
parser.add_argument(
    "-pr", "--predict", dest="predict", help="Save path", type=bool, default=True
)
args = parser.parse_args()


if __name__ == "__main__":
    # Load data
    if ".xlsx" in args.path:
        df = pd.read_excel(args.path)
    elif ".csv" in args.path:
        df = pd.read_csv(args.path)
    else:
        raise ValueError("Make sure that the data is either a `.csv` or a `.xlsx` file")

    # Check if a `Conclusie` column exists
    if "Conclusie" not in df.columns:
        raise ValueError("Make sure that the data contains a column called `Conclusie`")

    # Preprocess the documents
    docs = preprocess.preprocess_docs(df)

    # Define matching procedure
    main_matching_alg = RapidFuzz(n_jobs=1, scorer=fuzz.QRatio)
    to_match = [
        MatchEntity(
            search_term="bethesda klasse 1", ngram=3, matcher=main_matching_alg
        ),
        MatchEntity(search_term="klasse", ngram=2, matcher=main_matching_alg),
        MatchEntity(search_term="bethesda II", ngram=2, matcher=main_matching_alg),
        MatchEntity(
            search_term="bethesda classificatie II", ngram=3, matcher=main_matching_alg
        ),
        MatchEntity(
            search_term="bethesda categorie II", ngram=3, matcher=main_matching_alg
        ),
        # New, used for 'landelijk'
        MatchEntity(
            search_term="bethesda schildklier categorie 1",
            ngram=4,
            matcher=main_matching_alg,
        ),
        MatchEntity(
            search_term="bethesdacategorie 1", ngram=2, matcher=main_matching_alg
        ),
        MatchEntity(
            search_term="bethesdaklasse II", ngram=2, matcher=main_matching_alg
        ),
        MatchEntity(
            search_term="bethesda classificatie cat 2",
            ngram=4,
            matcher=main_matching_alg,
        ),
        MatchEntity(search_term="bethesda cat 2", ngram=4, matcher=main_matching_alg),
        MatchEntity(
            search_term="bethesda classificatie categorie II",
            ngram=4,
            matcher=main_matching_alg,
        ),
    ]
    entity_collection = EntityCollection(to_match)

    # Match entities
    matches = entity_collection.match(docs)
    df["Bethesda"] = preprocess.extract_bethesda_score(matches)

    # Save data
    if ".xlsx" in args.output:
        df.to_excel(args.output)
    elif ".csv" in args.output:
        df.to_csv(args.output)
    else:
        raise ValueError(
            "Make sure that the output path is either `.csv` or a `.xlsx` file"
        )

    # Predict pipeline
    if args.predict:
        # Prepare data
        predictor = Prediction(args.output)
        X, y, X_train, y_train, X_test, y_test, X_eval, y_eval = predictor.split_data()

        # Train model
        clf = LogisticRegression(
            random_state=0, max_iter=500, dual=True, solver="liblinear"
        )
        pipeline = [predictor.create_pipeline(clf)]
        cv_scores, reports, displays = predictor.evaluate(
            pipeline, X_train, y_train, X_test, y_test
        )

        # Predict unlabeled data
        results = predictor.add_predictions(pipeline[0], X, y)

        # Save data
        if ".xlsx" in args.output:
            df.to_excel(args.output)
        elif ".csv" in args.output:
            df.to_csv(args.output)
