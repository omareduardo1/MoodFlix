import os
import hashlib

import pandas as pd


RAW_DIR = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join("data", "movies.csv")


def assign_platforms(tconst: str) -> str:
    """
    Assegna piattaforme "finte" ma deterministiche a partire dall'ID IMDb.
    Serve solo per avere il filtro piattaforma funzionante nel progetto.
    """
    platforms_list = ["Netflix", "Prime", "Disney+", "HBO Max"]
    h = int(hashlib.sha256(tconst.encode("utf-8")).hexdigest(), 16)
    p1 = platforms_list[h % len(platforms_list)]
    p2 = platforms_list[(h // len(platforms_list)) % len(platforms_list)]
    if p1 == p2:
        return p1
    return f"{p1},{p2}"


def main():
    os.makedirs("data", exist_ok=True)

    basics_path = os.path.join(RAW_DIR, "title.basics.tsv.gz")
    ratings_path = os.path.join(RAW_DIR, "title.ratings.tsv.gz")

    if not os.path.exists(basics_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Non trovo i file IMDb in {RAW_DIR}. "
            f"Assicurati di aver scaricato 'title.basics.tsv.gz' e 'title.ratings.tsv.gz'."
        )

    print("📥 Carico title.basics.tsv.gz...")
    basics = pd.read_csv(
        basics_path,
        sep="\t",
        na_values="\\N",
        dtype={"tconst": str},
        low_memory=False,
    )

    print("📥 Carico title.ratings.tsv.gz...")
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        na_values="\\N",
        dtype={"tconst": str},
        low_memory=False,
    )

    print("🔍 Filtro solo i film veri (titleType == 'movie')...")
    movies = basics[basics["titleType"] == "movie"].copy()

    # Pulizia runtime e anno
    movies["runtimeMinutes"] = pd.to_numeric(
        movies["runtimeMinutes"], errors="coerce"
    )
    movies["startYear"] = pd.to_numeric(
        movies["startYear"], errors="coerce"
    )

    # Teniamo solo film con runtime, anno e generi
    movies = movies.dropna(
        subset=["runtimeMinutes", "startYear", "genres", "primaryTitle"]
    )

    # Solo film dal 1970 in poi
    movies = movies[movies["startYear"] >= 1970]

    print(f"🎬 Film dopo primi filtri: {len(movies)}")

    # Merge con ratings
    print("🔗 Unisco con rating IMDb...")
    ratings["averageRating"] = pd.to_numeric(
        ratings["averageRating"], errors="coerce"
    )
    ratings["numVotes"] = pd.to_numeric(
        ratings["numVotes"], errors="coerce"
    )

    movies = movies.merge(ratings, on="tconst", how="left")

    # Teniamo solo film con rating e almeno 5000 voti
    movies = movies.dropna(subset=["averageRating", "numVotes"])
    movies = movies[movies["numVotes"] >= 5000]

    print(f"⭐ Film con rating e >=5000 voti: {len(movies)}")

    # Ordiniamo per popolarità e limitiamo per tenere il dataset gestibile
    movies = movies.sort_values("numVotes", ascending=False).head(50000)
    print(f"📦 Prendo i top {len(movies)} film per popolarità (numVotes).")

    # Costruiamo il DataFrame finale
    out = pd.DataFrame()
    out["movie_id"] = movies["tconst"]
    out["title"] = movies["primaryTitle"]
    out["year"] = movies["startYear"].astype(int)
    out["genres"] = movies["genres"]
    out["runtime"] = movies["runtimeMinutes"].astype(int)
    out["rating"] = movies["averageRating"]
    out["num_votes"] = movies["numVotes"].astype(int)

    # Piattaforme sintetiche
    out["platforms"] = out["movie_id"].apply(assign_platforms)

    # Descrizione sintetica (IMDb non fornisce la trama in questi file)
    out["description"] = out.apply(
        lambda row: f"{row['title']} ({row['year']}) — {row['genres']} movie, IMDb rating {row['rating']:.1f}.",
        axis=1,
    )

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Salvato {OUTPUT_PATH} con {len(out)} film.")


if __name__ == "__main__":
    main()