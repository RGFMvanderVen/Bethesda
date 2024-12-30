import nltk
import pandas as pd

from tqdm import tqdm
from typing import List
from dataclasses import dataclass

from polyfuzz import PolyFuzz
from polyfuzz.models import BaseMatcher


@dataclass
class MatchEntity:
    """Class for defining a single entity to match.

    Usage:

    ```python
    from polyfuzz.models import RapidFuzz
    entity = MatchEntity(search_term="bethesda klasse 1",
                         ngram=3,
                         matcher=RapidFuzz(n_jobs=1, scorer=fuzz.QRatio))
    ```
    """

    search_term: str
    ngram: float
    matcher: BaseMatcher = None


class EntityCollection:
    """Collection of Entities for extraction

    There is room for improvement computation-wise:
        * Paralellize search between documents
        * Create a vocabulary before-hand
            * Create a similarity matrix between vocab and search terms
            * Match highly similar matches with words found in documents

    Usage:

    ```python
    # Define matching procedure
    to_match = [
        {"search_term": "bethesda klasse 1", "ngram": 3, "matcher": RapidFuzz(n_jobs=1, scorer=fuzz.QRatio)},
        {"search_term": "klasse", "ngram": 2, "matcher": RapidFuzz(n_jobs=1, scorer=fuzz.QRatio)},
    ]

    # Create a Collection of Matching Entities
    entity_collection = EntityCollection(to_match)
    ```
    """

    def __init__(self, entities: List[MatchEntity]):
        """Initialize EntityCollection

        Arguments:
            entities: a MatchEntity describing the procedure for matching certain words

        """
        self.entities = entities

    def match(self, docs) -> pd.DataFrame:
        """Match search terms with terms found in documents

        Usage:

        ```python
        # Define matching procedure
        to_match = [
            {"search_term": "bethesda klasse 1", "ngram": 3, "matcher": RapidFuzz(n_jobs=1, scorer=fuzz.QRatio)},
            {"search_term": "klasse", "ngram": 2, "matcher": RapidFuzz(n_jobs=1, scorer=fuzz.QRatio)},
        ]

        # Create a Collection of Matching Entities
        entity_collection = EntityCollection(to_match)
        results = entity_collection.match(docs)
        ```

        """
        matches = []

        for doc_id, doc in enumerate(tqdm(docs)):
            for entity in self.entities:
                if entity.matcher is None:
                    model = PolyFuzz("EditDistance")
                else:
                    model = PolyFuzz(entity.matcher)

                ngrams = [
                    " ".join(ngram) for ngram in nltk.ngrams(doc.split(), entity.ngram)
                ]
                model.match([entity.search_term], ngrams, top_n=1)

                match = model.get_matches().To.values[0]
                similarity = model.get_matches().Similarity.values[0]

                matches.append((doc_id, entity.search_term, match, similarity))

        matches = pd.DataFrame(
            matches, columns=["Doc_ID", "Search_term", "Match", "Score"]
        )
        return matches
