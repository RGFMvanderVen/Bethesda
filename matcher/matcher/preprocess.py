import re
import pandas as pd
from typing import List


def preprocess_docs(data: pd.DataFrame) -> List[str]:
    """Preprocess the documents in a dataframe that
    has the `Conclusie` column

    Arguments:
        data: Dataframe with the `Conclusie` column

    Returns:
        docs: The preprocessed documents
    """
    docs = [
        (
            doc.replace("(", " ")
            .replace(")", " ")
            .replace(":", " ")
            .replace(".", " ")
            .replace("|", " ")
            .replace("-", " ")
            .replace("_x000D_", " ")
            .replace("\n", " ")
            .strip()
        )
        for doc in data.Conclusie.tolist()
    ]
    docs = [" ".join(doc.split()) for doc in docs]
    docs = [re.sub(" +", " ", doc) for doc in docs]

    return docs


def extract_bethesda_score(matches_df: pd.DataFrame) -> pd.Series:
    """Extract the Bethesda score from a set of matches

    Arguments:
        matches_df: Dataframe with the following columns:
                     * Doc_ID
                     * Search_term
                     * Match
                     * Score

    Returns:
        selection.Bethesda_Score: The Bethesda score
    """
    score_map = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
    }
    scores = sorted(list(score_map.keys()), key=len)[::-1]

    # Extract best matches
    selection = (
        matches_df.sort_values("Score", ascending=False).groupby("Doc_ID").first()
    )
    unique_matches = selection.loc[selection.Score > 0.8, "Match"].unique()

    # Map matches to Bethesda Score
    extracted = {}
    for match in unique_matches:
        for score in scores:
            if score in match:
                extracted[match] = score_map[score]
                break

    # Map Bethesda Score to a numerical value
    selection["Bethesda_Score"] = selection["Match"].replace(extracted)
    selection.loc[
        ~selection.Bethesda_Score.isin([0, 1, 2, 3, 4, 5, 6]), "Bethesda_Score"
    ] = None

    return selection.Bethesda_Score
