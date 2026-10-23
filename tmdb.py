import requests
import streamlit as st

IMG_BASE = "https://image.tmdb.org/t/p/w342"


@st.cache_data(show_spinner=False)
def get_details(tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    try:
        r = requests.get(url, params={"api_key": api_key}, timeout=8)
        r.raise_for_status()
        d = r.json()
    except requests.RequestException:
        return None

    poster = d.get("poster_path")
    return {
        "poster": IMG_BASE + poster if poster else None,
        "overview": d.get("overview", ""),
        "rating": d.get("vote_average"),
        "year": (d.get("release_date") or "")[:4],
    }