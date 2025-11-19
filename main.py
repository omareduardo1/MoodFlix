from recommender import MoodFlixRecommender


def choose_option(prompt: str, options: dict) -> str:
    """
    Mostra un piccolo menu numerato e restituisce il valore scelto.
    options: { "1": "felice", "2": "triste", ... }
    """
    print(prompt)
    for key, label in options.items():
        print(f"  {key}) {label}")
    while True:
        choice = input("Seleziona un'opzione: ").strip()
        if choice in options:
            return options[choice]
        print("Scelta non valida, riprova.")


def main():
    print("======================================")
    print("        🎬 MoodFlix Recommender       ")
    print("          (powered by IMDb)          ")
    print("======================================\n")

    # Inizializza il recommender
    try:
        rec = MoodFlixRecommender("data/movies.csv")
    except FileNotFoundError:
        print("❌ Non trovo 'data/movies.csv'.")
        print("   1) Scarica da https://datasets.imdbws.com/")
        print("      - title.basics.tsv.gz")
        print("      - title.ratings.tsv.gz")
        print("   2) Mettili in data/raw/")
        print("   3) Esegui: python3 prepare_imdb_dataset.py")
        return

    # 1) Mood
    mood_options = {
        "1": "felice",
        "2": "triste",
        "3": "stressato",
        "4": "riflessivo",
        "5": "neutro",
    }
    mood = choose_option("Come ti senti oggi?", mood_options)
    print(f"Hai scelto: {mood}\n")

    # 2) Durata
    duration_options = {
        "1": "<60",
        "2": "60-90",
        "3": "90-120",
        "4": ">120",
    }
    duration_choice = choose_option("Quanto tempo hai?", duration_options)
    print(f"Hai scelto: {duration_choice} minuti\n")

    # 3) Piattaforma
    platform_options = {
        "1": "Qualsiasi",
        "2": "Netflix",
        "3": "Prime",
        "4": "Disney+",
        "5": "HBO Max",
    }
    platform = choose_option("Su quale piattaforma vuoi cercare?", platform_options)
    print(f"Hai scelto: {platform}\n")

    # 4) Genere desiderato (opzionale)
    print("Vuoi specificare un genere di film (es. Action, Comedy, Drama, Horror, Sci-Fi, Romance, Animation, Thriller, Documentary)?")
    desired_genre = input("Scrivi il genere oppure premi Invio per nessuna preferenza: ").strip()
    if not desired_genre:
        desired_genre = None
        print("Nessuna preferenza di genere.\n")
    else:
        print(f"Hai scelto il genere: {desired_genre}\n")

    # 5) Numero di raccomandazioni
    while True:
        try:
            n_recs = int(input("Quanti film vuoi che ti consigli? [default 5]: ") or "5")
            if n_recs <= 0:
                raise ValueError
            break
        except ValueError:
            print("Inserisci un numero intero positivo, per favore.")

    platform_arg = platform if platform != "Qualsiasi" else "qualsiasi"

    # Chiamata al recommender
    results = rec.recommend(
        mood=mood,
        duration_choice=duration_choice,
        platform=platform_arg,
        n_recs=n_recs,
        desired_genre=desired_genre,
    )

    print("\n======================================")
    print("             RISULTATI 🎥             ")
    print("======================================\n")

    if results.empty:
        print(
            "Nessun film trovato con questi filtri. Prova a cambiare durata, piattaforma o genere."
        )
        return

    for _, row in results.iterrows():
        year_str = f" ({int(row['year'])})" if "year" in row and not pd.isna(row["year"]) else ""
        print(f"- Titolo: {row['title']}{year_str}")
        if "rating" in row:
            print(f"  Rating IMDb: {row['rating']:.1f} ({int(row['num_votes'])} voti)")
        print(f"  Generi: {row['genres']}")
        print(f"  Durata: {int(row['runtime'])} min")
        print(f"  Piattaforme (sintetiche): {row['platforms']}")
        print(f"  Score raccomandazione: {row['score']:.3f}")

        desc = str(row.get("description", "")).strip()
        if desc:
            short_desc = (desc[:160] + '...') if len(desc) > 160 else desc
            print(f"  Descrizione: {short_desc}")
        print()

    print("Buona visione! 🍿")


if __name__ == "__main__":
    import pandas as pd  # usato in main per pd.isna
    main()