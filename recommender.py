# recommender.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse as sp


class MoodFlixRecommender:
    """
    Sistema di raccomandazione content-based basato su IMDb:

    - TF-IDF su (generi + descrizione sintetica)
    - runtime normalizzato come feature numerica
    - mapping mood -> generi preferiti
    - filtri per piattaforma, durata, generi (mood + genere scelto)
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.movies = pd.read_csv(csv_path)

        print("📂 Colonne in movies.csv:", list(self.movies.columns))
        print(f"🎬 Numero totale di film: {len(self.movies)}")

        # Pulizia base
        for col in ["genres", "description", "platforms"]:
            if col not in self.movies.columns:
                self.movies[col] = ""
            self.movies[col] = self.movies[col].fillna("")

        if "runtime" not in self.movies.columns:
            raise ValueError("Nel CSV manca la colonna 'runtime'.")
        self.movies["runtime"] = self.movies["runtime"].fillna(
            self.movies["runtime"].median()
        )

        # Testo combinato (generi + descrizione)
        self.movies["text_features"] = (
            self.movies["genres"].astype(str).str.replace(",", " ", regex=False)
            + " "
            + self.movies["description"].astype(str)
        )

        print("✅ Colonna 'text_features' creata.")

        # TF-IDF su text_features (sparse)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["text_features"]
        )

        # Normalizzazione runtime
        self.scaler = MinMaxScaler()
        runtime_scaled = self.scaler.fit_transform(self.movies[["runtime"]])
        runtime_sparse = sp.csr_matrix(runtime_scaled)

        # Feature matrix: [tfidf | runtime] (sparse)
        self.feature_matrix = sp.hstack(
            [self.tfidf_matrix, runtime_sparse], format="csr"
        )

    # -------------------------------
    # Mapping mood → priorità generi
    # -------------------------------
    @staticmethod
    def mood_to_genres(mood: str) -> List[str]:
        mood = mood.lower()
        if mood in ["triste", "down", "low", "bassa energia"]:
            return ["Comedy", "Family", "Animation", "Romance"]
        if mood in ["felice", "happy", "alta energia", "carico"]:
            return ["Action", "Adventure", "Sci-Fi", "Thriller"]
        if mood in ["riflessivo", "pensieroso", "neutro"]:
            return ["Drama", "Biography", "Documentary"]
        if mood in ["stressato", "ansioso", "stress"]:
            return ["Animation", "Family", "Documentary"]
        # default
        return ["Drama", "Comedy"]

    @staticmethod
    def duration_range(choice: str) -> Tuple[int, int]:
        choice = choice.replace(" ", "").lower()
        if choice == "<60":
            return (0, 60)
        if choice == "60-90":
            return (60, 90)
        if choice == "90-120":
            return (90, 120)
        if choice == ">120":
            return (120, 10000)
        return (0, 10000)

    def filter_by_platform(self, platform: Optional[str]) -> pd.DataFrame:
        if (
            platform is None
            or platform.strip() == ""
            or platform.lower() == "qualsiasi"
        ):
            return self.movies.copy()
        mask = self.movies["platforms"].str.contains(
            platform, case=False, na=False
        )
        return self.movies[mask].copy()

    @staticmethod
    def _has_genre(genres_str: str, target_genres: List[str]) -> bool:
        genres_list = [g.strip().lower() for g in str(genres_str).split(",")]
        target_lower = [g.lower() for g in target_genres]
        return any(g in target_lower for g in genres_list)

    def recommend(
        self,
        mood: str,
        duration_choice: str,
        platform: Optional[str],
        n_recs: int = 5,
        desired_genre: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Restituisce un dataframe con i migliori n_recs film
        per il mood specificato, durata stimata, piattaforma e (opzionale) genere desiderato.
        """

        mood_genres = self.mood_to_genres(mood)
        min_d, max_d = self.duration_range(duration_choice)

        desired_genre_norm = (
            desired_genre.strip().lower()
            if isinstance(desired_genre, str) and desired_genre.strip()
            else None
        )

        # 1) Filtro piattaforma
        base_candidates = self.filter_by_platform(platform)

        # 2) Filtro durata
        duration_mask = (
            (base_candidates["runtime"] >= min_d)
            & (base_candidates["runtime"] <= max_d)
        )
        candidates = base_candidates[duration_mask].copy()

        if candidates.empty:
            print(
                "⚠️ Nessun film nel range di durata scelto. Uso tutti i film della piattaforma."
            )
            candidates = base_candidates.copy()

        # 3) Filtro per GENERE DESIDERATO (se specificato)
        if desired_genre_norm is not None and not candidates.empty:
            print(f"🎯 Filtro anche per genere desiderato: {desired_genre_norm}")
            genre_pref_mask = candidates["genres"].fillna("").apply(
                lambda g: self._has_genre(g, [desired_genre_norm])
            )
            pref_candidates = candidates[genre_pref_mask]
            if not pref_candidates.empty:
                candidates = pref_candidates
            else:
                print(
                    "⚠️ Nessun film con il genere desiderato nei filtri attuali. "
                    "Mantengo i candidati senza il filtro sul genere."
                )

        # 4) Filtro per generi coerenti col MOOD (se possibile)
        if not candidates.empty:
            mood_mask = candidates["genres"].fillna("").apply(
                lambda g: self._has_genre(g, mood_genres)
            )
            mood_candidates = candidates[mood_mask]
            if not mood_candidates.empty:
                candidates = mood_candidates

        if candidates.empty:
            return pd.DataFrame(
                columns=self.movies.columns.tolist() + ["score"]
            )

        # 5) Profilo utente (testo + runtime target)
        profile_tokens = list(mood_genres)
        if desired_genre_norm is not None:
            profile_tokens.append(desired_genre_norm)
        pseudo_text = " ".join(profile_tokens)

        user_vec_text = self.vectorizer.transform([pseudo_text])

        runtime_target = np.array([[(min_d + max_d) / 2.0]])
        runtime_target_scaled = self.scaler.transform(runtime_target)
        runtime_target_sparse = sp.csr_matrix(runtime_target_scaled)

        user_vec = sp.hstack(
            [user_vec_text, runtime_target_sparse], format="csr"
        )

        # 6) Similarità coseno solo sui candidati
        candidate_indices = candidates.index.values
        candidate_features = self.feature_matrix[candidate_indices]

        sims = cosine_similarity(user_vec, candidate_features)[0]

        candidates = candidates.copy()
        candidates["score"] = sims
        candidates = candidates.sort_values("score", ascending=False)

        return candidates.head(n_recs)