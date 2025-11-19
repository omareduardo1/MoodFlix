# 🎬 MoodFlix  
### _A Mood-Based Movie Recommender System (CLI – Powered by IMDb & Machine Learning)_

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Content--Based-yellow" />
  <img src="https://img.shields.io/badge/IMDb-Dataset-orange?logo=imdb" />
  <img src="https://img.shields.io/badge/Platform-CLI-lightgrey?logo=terminal" />
</p>

---

## 🌟 Overview

**MoodFlix** is a command-line movie recommender system that selects films for you based on:

- 🎭 **Your mood**
- ⏱️ **The time you have available**
- 📺 **The platform you prefer** (synthetic Netflix / Prime / Disney+ / HBO Max tags)
- 🎞️ **Optionally, a specific genre you want** (Action, Drama, Sci-Fi, Romance...)

Under the hood, MoodFlix uses a **content-based recommendation model** built on:

- TF-IDF embeddings  
- Cosine similarity  
- IMDb official *non-commercial* datasets  

---

## ⚠️ Legal Notice (Important)

This repository **DOES NOT** include any IMDb data.  
IMDb data must be:

1. Downloaded manually  
2. Used **only** for personal / non-commercial purposes  
3. Processed locally via `prepare_imdb_dataset.py`

You must download the official datasets from:

👉 https://datasets.imdbws.com/

Required files:

- `title.basics.tsv.gz`  
- `title.ratings.tsv.gz`

Place them in:
data/raw/

---

# 🧠 How It Works

## 1. Data Preparation

`prepare_imdb_dataset.py` loads the IMDb TSV files and:

- Keeps only **real movies** (`titleType = "movie"`)
- Filters:
  - runtime available  
  - year ≥ 1970  
  - rating present  
  - at least **5000 votes**  
- Merges basics + ratings  
- Generates a clean `movies.csv` with the following columns:
movie_id, title, year, genres, runtime, rating, num_votes,
platforms (synthetic), description (short)

Synthetic streaming availability (Netflix / Prime / Disney+ / HBO Max)  
is assigned deterministically from movie IDs for demo purposes.

---

## 2. Feature Engineering

For each movie:

- A combined text field is created:  
genres + description

- TF-IDF vectorization (up to 10,000 features)
- Runtime is normalized
- All features are stored in a **sparse matrix** for efficiency

---

## 3. User Profile

MoodFlix builds a profile vector using:

- The **selected mood**
- The **desired genre** (optional)
- The **target duration range**

Moods map to preferred genres:

| Mood        | Preferential Genres                               |
|-------------|---------------------------------------------------|
| felice      | Action, Adventure, Sci-Fi, Thriller               |
| triste      | Comedy, Romance, Family, Animation                |
| stressato   | Animation, Family, Documentary                    |
| riflessivo  | Drama, Biography, Documentary                     |
| neutro      | Drama, Comedy                                     |

---

## 4. Ranking Algorithm

Movies are filtered by:

- Platform  
- Runtime range  
- Desired genre  
- Mood-compatible genres  

Then ranked using **cosine similarity** between:

- user profile vector  
- movie feature vectors  

---

# 📂 Project Structure
Moodflix/
│── main.py                   # CLI interface
│── recommender.py            # Recommendation engine
│── prepare_imdb_dataset.py   # Generates movies.csv from IMDb data
│── requirements.txt
│── README.md
│── .gitignore
└── data/
├── raw/                  # IMDb TSV files (NOT included)
└── movies.csv            # Generated dataset (NOT included)
