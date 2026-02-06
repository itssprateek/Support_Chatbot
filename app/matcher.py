from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQMatcher:
    def __init__(self, faq_items):
        """
        faq_items: list of dicts with keys like:
        Intent, User Question, Bot Response
        """
        self.faq_items = faq_items

        # Build searchable text (question + intent helps)
        self.questions = [
            f"{x.get('Intent','')} :: {x.get('User Question','')}".strip()
            for x in faq_items
        ]

        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def top_k(self, query: str, k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            results.append((self.faq_items[idx], float(score)))
        return results