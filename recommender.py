import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class Recommender:
    def __init__(self, data_dir="data", min_ratings=20):
        movies = pd.read_csv(f"{data_dir}/movies.csv")
        ratings = pd.read_csv(f"{data_dir}/ratings.csv")
        tags = pd.read_csv(f"{data_dir}/tags.csv")

        # a few titles repeat in the dataset, keep the first one
        self.movies = movies.drop_duplicates("title").reset_index(drop=True)
        self.title_to_idx = pd.Series(self.movies.index, index=self.movies["title"])
        self.titles = sorted(self.movies["title"].tolist())

        self._build_content(tags)
        self._build_collab(ratings, min_ratings)
        links = pd.read_csv(f"{data_dir}/links.csv")
        self.tmdb_map = (
        links.dropna(subset=["tmdbId"])
        .set_index("movieId")["tmdbId"]
        .astype(int)
        .to_dict()
    )

    # ---------- content based ----------
    def _build_content(self, tags):
        tag_text = tags.groupby("movieId")["tag"].apply(
            lambda t: " ".join(t.astype(str))
        )
        tag_col = self.movies["movieId"].map(tag_text).fillna("")
        genre_col = self.movies["genres"].str.replace("|", " ", regex=False)

        text = (genre_col + " " + tag_col).str.lower()
        vectorizer = TfidfVectorizer(stop_words="english")
        self.content_matrix = vectorizer.fit_transform(text)

    def content_based(self, title, n=10):
        i = self.title_to_idx[title]
        scores = cosine_similarity(self.content_matrix[i], self.content_matrix).ravel()

        order = scores.argsort()[::-1]
        top = [j for j in order if j != i][:n]

        return pd.DataFrame({
    "movieId": self.movies.loc[top, "movieId"].values,
    "Title": self.movies.loc[top, "title"].values,
    "Genres": self.movies.loc[top, "genres"].values,
    "Similarity": scores[top].round(3),
})

    # ---------- collaborative ----------
    def _build_collab(self, ratings, min_ratings):
        counts = ratings["movieId"].value_counts()
        keep = counts[counts >= min_ratings].index
        r = ratings[ratings["movieId"].isin(keep)]

        movie_cat = r["movieId"].astype("category")
        user_cat = r["userId"].astype("category")

        # rows = movies, columns = users, values = ratings
        self.rating_matrix = csr_matrix(
            (r["rating"], (movie_cat.cat.codes, user_cat.cat.codes))
        )
        self.collab_ids = list(movie_cat.cat.categories)
        self.id_to_row = {m: k for k, m in enumerate(self.collab_ids)}

        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn.fit(self.rating_matrix)

    def collaborative(self, title, n=10):
        movie_id = self.movies.loc[self.title_to_idx[title], "movieId"]
        if movie_id not in self.id_to_row:
            return None  # not enough ratings for this movie

        row = self.id_to_row[movie_id]
        dist, ind = self.knn.kneighbors(self.rating_matrix[row], n_neighbors=n + 1)

        ids = [self.collab_ids[k] for k in ind[0][1:]]  # first one is the movie itself
        sims = 1 - dist[0][1:]

        found = self.movies.set_index("movieId").loc[ids]
        return pd.DataFrame({
    "movieId": ids,
    "Title": found["title"].values,
    "Genres": found["genres"].values,
    "Similarity": sims.round(3),
})