import streamlit as st
import pandas as pd
import joblib
from music_mapper import load_mapping, recommend_songs
from features import log_transform

@st.cache_resource # calling a model once
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data # calling data once
def get_mapping():
    return load_mapping()

model = load_model()
mapping_df = get_mapping()

st.title("🧠 EEG emotion based music recommendation") 
st.write("We recommend you a playlist based on EEG feature CSV file you upload!")

mode = st.radio("Choose data", ["Try with sample data", "Upload my own CSV"])

if mode == "Try with sample data":
    sample = st.selectbox("Select a sample", ["positive", "neutral", "negative"])
    uploaded = f"data/sample_{sample}.csv"
else:
    uploaded = st.file_uploader("Upload your CSV file", type="csv")
    st.caption("Requires the same feature columns as the birdy654 Kaggle EEG emotion dataset.")

if uploaded: # run only if a file is uploaded
    df = pd.read_csv(uploaded).drop(columns=["label"], errors="ignore")
    st.write(f"uploaded data: {len(df)} rows")
    
    missing = set(model.feature_names_in_) - set(df.columns)

    if missing:
        st.error(f"{len(missing)} columns are missing. This doesn't look like a supported EEG feature file. "
                 "Please use the same format as the birdy654 Kaggle EEG emotion dataset.")
        st.stop()

    x = log_transform(df[model.feature_names_in_])
    preds = model.predict(x) 
    counts = pd.Series(preds).value_counts() 
    emotion = counts.idxmax()

    st.subheader(f"detected emotion: {emotion}")
    st.bar_chart(counts) 

    refresh = st.button("🔄 get another song recommendation") # true when press button, false otherwise
    if refresh or st.session_state.get("emotion") != emotion:
        st.session_state.emotion = emotion
        st.session_state.songs = recommend_songs(emotion, mapping_df, n=3)

    songs = st.session_state.songs
    if songs.empty:
        st.warning("There is no song matching this emotion.")
    else:
        st.write("song recommendation:")
        st.dataframe(songs, hide_index=True)
