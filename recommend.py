import pandas as pd
import numpy as np
import re 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

class RecommendationModel():
    def __init__(self):
        # Data Preparation
        spotify_df = pd.read_csv('data.csv')
        data_w_genre = pd.read_csv('data_w_genres.csv')
        data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
        spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
        spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
        spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'])
        spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis=1)
        spotify_df.sort_values(['artists_song','release_date'], ascending=False, inplace=True)
        spotify_df.drop_duplicates('artists_song', inplace=True)
        artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')
        artists_exploded_enriched = artists_exploded.merge(
            data_w_genre, how='left', left_on='artists_upd', right_on='artists')
        artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]
        artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
        artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(
            lambda x: list(set(list(itertools.chain.from_iterable(x)))))
        spotify_df = spotify_df.merge(
            artists_genres_consolidated[['id','consolidates_genre_lists']], on='id', how='left')
        spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
        float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
        spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
        spotify_df['genre'] = spotify_df['consolidates_genre_lists'].apply(
            lambda d: d if isinstance(d, list) else [])
        
        # Feature Engineering
        def ohe_prep(df, column, new_name): 
            tf_df = pd.get_dummies(df[column])
            feature_names = tf_df.columns
            tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
            tf_df.reset_index(drop=True, inplace=True)    
            return tf_df
        
        # def create_feature_set(df, float_cols):
        #     tfidf = TfidfVectorizer()
        #     tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
        #     genre_df = pd.DataFrame(tfidf_matrix.toarray())
        #     genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        #     genre_df.reset_index(drop=True, inplace=True)
        #     year_ohe = ohe_prep(df, 'year','year') * 0.5
        #     popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15
        #     floats = df[float_cols].reset_index(drop=True)
        #     scaler = MinMaxScaler()
        #     floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2
        #     final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)
        #     final['id'] = df['id'].values
        #     return final
        def create_feature_set(df, float_cols):
            tfidf = TfidfVectorizer()
            # Ensure 'consolidates_genre_lists' contains lists
            df['consolidates_genre_lists'] = df['consolidates_genre_lists'].apply(lambda x: x if isinstance(x, list) else [])
            # Apply TF-IDF Vectorizer
            tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
            genre_df = pd.DataFrame(tfidf_matrix.toarray())
            genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
            genre_df.reset_index(drop=True, inplace=True)
            
            # One-Hot Encoding for 'year' and 'popularity_red'
            year_ohe = ohe_prep(df, 'year', 'year') * 0.5
            popularity_ohe = ohe_prep(df, 'popularity_red', 'pop') * 0.15
            
            # Scale float columns
            floats = df[float_cols].reset_index(drop=True)
            scaler = MinMaxScaler()
            floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2
            
            # Concatenate all features
            final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)
            final['id'] = df['id'].values
            return final


        complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)
        self.spotify_df = spotify_df
        self.complete_feature_set = complete_feature_set

    def create_necessary_outputs(self, playlist_ids):
        playlist = pd.DataFrame()
        for ix, song_id in enumerate(playlist_ids):
            if song_id in self.spotify_df['id'].values:
                song_data = self.spotify_df[self.spotify_df['id'] == song_id].iloc[0]
                playlist.loc[ix, 'artist'] = song_data['artists_upd'][0]
                playlist.loc[ix, 'name'] = song_data['name']
                playlist.loc[ix, 'id'] = song_id
                playlist.loc[ix, 'date_added'] = pd.to_datetime('today') - pd.Timedelta(days=ix)
        playlist['date_added'] = pd.to_datetime(playlist['date_added'])
        playlist = playlist[playlist['id'].isin(self.spotify_df['id'].values)].sort_values('date_added', ascending=False)
        return playlist

    def generate_playlist_feature(self, complete_feature_set, playlist_df, weight_factor=1.09):
        complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
        complete_feature_set_playlist = complete_feature_set_playlist.merge(
            playlist_df[['id','date_added']], on='id', how='inner')
        complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
        playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending=False)
        most_recent_date = playlist_feature_set.iloc[0]['date_added']
        for ix, row in playlist_feature_set.iterrows():
            playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row['date_added'].to_pydatetime()).days / 30)
        playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(
            lambda x: weight_factor ** (-x))
        playlist_feature_set_weighted = playlist_feature_set.copy()
        playlist_feature_set_weighted.update(
            playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight, 0))
        playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
        return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist

    def generate_playlist_recos(self, features, nonplaylist_features, top_k=10):
        non_playlist_df = self.spotify_df[self.spotify_df['id'].isin(nonplaylist_features['id'].values)]
        non_playlist_df = non_playlist_df.reset_index(drop=True)
        nonplaylist_features = nonplaylist_features.reset_index(drop=True)
        non_playlist_df['sim'] = cosine_similarity(
            nonplaylist_features.drop('id', axis=1).values, features.values.reshape(1, -1))[:,0]
        non_playlist_df_top_k = non_playlist_df.sort_values('sim', ascending=False).head(top_k)
        return non_playlist_df_top_k
