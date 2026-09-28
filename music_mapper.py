import pandas as pd

def load_mapping(csv_path="data/track_mapping.csv"):
    df = pd.read_csv(csv_path)
    return df

def recommend_songs(emotion_label, df, n=3):
    filtered = df[df['emotion_label'] == emotion_label]
    if filtered.empty:
        return pd.DataFrame(columns=['track_name', 'artist']) # print error when it is empty

    # select n number of songs randomly
    recommended = filtered.sample(n=min(n, len(filtered)))
    return recommended[['track_name', 'artist']]

# test
if __name__ == "__main__":
    df = load_mapping()
    result = recommend_songs("NEGATIVE", df, n=3)
    print(result)