import streamlit as st
from recommender import Recommender
from tmdb import get_details

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_resource
def load_model():
    return Recommender("data")


rec = load_model()

try:
    api_key = st.secrets["TMDB_API_KEY"]
except Exception:
    api_key = ""


def fetch(movie_id):
    tmdb_id = rec.tmdb_map.get(int(movie_id))
    if api_key and tmdb_id:
        return get_details(tmdb_id, api_key)
    return None


def show_cards(result, per_row=5):
    for start in range(0, len(result), per_row):
        cols = st.columns(per_row)
        chunk = result.iloc[start:start + per_row]
        for col, (_, row) in zip(cols, chunk.iterrows()):
            info = fetch(row["movieId"])
            with col:
                if info and info["poster"]:
                    st.image(info["poster"], use_container_width=True)
                else:
                    st.write("No poster available")
                st.markdown(f"**{row['Title']}**")
                st.caption(row["Genres"].replace("|", " · "))
                st.caption(f"Match: {row['Similarity']:.0%}")
                if info and info["rating"]:
                    st.caption(f"TMDB rating: {info['rating']:.1f}/10")
                if info and info["overview"]:
                    with st.expander("Overview"):
                        st.write(info["overview"])


st.title("Movie Recommendation System")
st.write("Pick a movie you like and get similar ones.")

if not api_key:
    st.info("TMDB key not found, so posters are turned off. Add it in .streamlit/secrets.toml.")

method = st.sidebar.radio(
    "Recommendation method",
    ["Content-based", "Collaborative (user ratings)"],
)
n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

default = rec.titles.index("Toy Story (1995)")
title = st.selectbox("Select a movie", rec.titles, index=default)

# details of the selected movie
selected_id = rec.movies.loc[rec.title_to_idx[title], "movieId"]
selected = fetch(selected_id)
left, right = st.columns([1, 4])
with left:
    if selected and selected["poster"]:
        st.image(selected["poster"], use_container_width=True)
with right:
    st.subheader(title)
    if selected and selected["overview"]:
        st.write(selected["overview"])

if st.button("Recommend"):
    if method == "Content-based":
        result = rec.content_based(title, n)
    else:
        result = rec.collaborative(title, n)

    if result is None:
        st.warning("Too few ratings for collaborative filtering on this movie, showing content-based results instead.")
        result = rec.content_based(title, n)

    st.subheader("You might also like")
    show_cards(result)

st.markdown("---")
st.caption("Data: MovieLens (GroupLens). Posters and details: TMDB. "
           "This product uses the TMDB API but is not endorsed or certified by TMDB.")