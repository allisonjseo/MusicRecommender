import pandas as pd
import numpy as np
import re 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

from recommend import RecommendationModel

##Example 1

# playlist_ids = ['6wn61Fzx9XMxQmieLpoIhW', '1D4PL9B8gOg78jiHg3FvBb', '5P4wWhUYWM0IaVYLuZxdar']
# playlist_ids = ['7BqBn9nzAq8spo5e7cZ0dJ', '3RH9idbxUAMcUldet2ormp', '4HlFJV71xXKIGcU3kRyttv']
# number is number of recommendations
def pred(song_id1, song_id2, song_id3, number):
    #Instantiate the model
    model = RecommendationModel()
    
    # Provide a list of song IDs from your dataset
    playlist_ids = [song_id1, song_id2, song_id3]

    # Create playlist dataframe
    playlist_df = model.create_necessary_outputs(playlist_ids)

    # Generate playlist feature vector and non-playlist features
    features, nonplaylist_features = model.generate_playlist_feature(model.complete_feature_set, playlist_df)

    # Generate recommendations with top_k=1 to get only the top recommendation
    recommendations = model.generate_playlist_recos(features, nonplaylist_features, top_k=int(number))

    # View the top recommendation
    # print(recommendations[['id', 'artists_upd', 'name', 'sim']])

    print(recommendations[['id', 'artists_upd', 'name', 'sim']].to_string(index=False))

    # return recommendations[['id', 'artists_upd', 'name', 'sim']].to_string(index=False)
    # return recommendations[['artists_upd', 'name']].to_string(index=False)
    return formatHelper(recommendations[['artists_upd', 'name', 'sim']])

def formatHelper(df):
    # Make sure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    
    # Initialize the result list
    ans = ""

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract artist names and join them if there are multiple artists
        artist_names = ', '.join(row['artists_upd'])  # Join all artist names with a comma and a space
        song_name = row['name']
        similarity = str(row['sim'])[:4]

        # Combine into a string
        result = f"{artist_names} - {song_name} ({similarity})"

        # Print and add to the result list
        print(result)
        ans += result + "\n"
    
    return ans[:-1]

import pandas as pd
import re
from difflib import get_close_matches

def find_song_ids(song_name1, artist_name1, song_name2, artist_name2, song_name3, artist_name3, number):
    """
    Finds the song IDs associated with given song names and artist names from the data.csv file.

    Parameters:
    - song_name1, song_name2, song_name3 (str): The song names to find IDs for.
    - artist_name1, artist_name2, artist_name3 (str): The corresponding artist names.
    - number (int): An additional parameter as per your requirement.

    Returns:
    - The result of the pred function called with the found song IDs and the 'number' parameter.
    """
    input_songs = [
        {'song_name': song_name1, 'artist_name': artist_name1},
        {'song_name': song_name2, 'artist_name': artist_name2},
        {'song_name': song_name3, 'artist_name': artist_name3}
    ]

    # Read the data.csv file into a DataFrame
    df = pd.read_csv('data.csv')

    # Preprocess the 'name' and 'artists' columns in the DataFrame
    def preprocess(text):
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    df['name_clean'] = df['name'].apply(preprocess)
    df['artists_clean'] = df['artists'].apply(preprocess)

    # Initialize a dictionary to store the results
    song_ids = {}

    # For each input song and artist name, find the best match
    for item in input_songs:
        song_name_clean = preprocess(item['song_name'])
        artist_name_clean = preprocess(item['artist_name'])

        # Filter the DataFrame to rows that contain the artist name
        df_artist_filtered = df[df['artists_clean'].str.contains(artist_name_clean, na=False)]

        if not df_artist_filtered.empty:
            # Create a list of preprocessed song names from the filtered DataFrame
            song_names_list = df_artist_filtered['name_clean'].tolist()
            # Find close matches using difflib
            matches = get_close_matches(song_name_clean, song_names_list, n=1, cutoff=0.6)
            if matches:
                best_match = matches[0]
                # Retrieve the song ID for the best match
                song_row = df_artist_filtered[df_artist_filtered['name_clean'] == best_match]
                if not song_row.empty:
                    song_id = song_row['id'].iloc[0]
                    key = f"{item['song_name']} by {item['artist_name']}"
                    song_ids[key] = song_id
                else:
                    key = f"{item['song_name']} by {item['artist_name']}"
                    song_ids[key] = None
            else:
                key = f"{item['song_name']} by {item['artist_name']}"
                song_ids[key] = None
        else:
            # If artist is not found, attempt to find the song in the entire DataFrame
            song_names_list = df['name_clean'].tolist()
            matches = get_close_matches(song_name_clean, song_names_list, n=1, cutoff=0.6)
            if matches:
                best_match = matches[0]
                song_row = df[df['name_clean'] == best_match]
                if not song_row.empty:
                    song_id = song_row['id'].iloc[0]
                    key = f"{item['song_name']} by {item['artist_name']}"
                    song_ids[key] = song_id
                else:
                    key = f"{item['song_name']} by {item['artist_name']}"
                    song_ids[key] = None
            else:
                key = f"{item['song_name']} by {item['artist_name']}"
                song_ids[key] = None

    print(song_ids)
    return pred(
        song_ids.get(f"{song_name1} by {artist_name1}"),
        song_ids.get(f"{song_name2} by {artist_name2}"),
        song_ids.get(f"{song_name3} by {artist_name3}"),
        number
    )
