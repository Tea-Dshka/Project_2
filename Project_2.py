import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import datetime


movies = pd.read_csv(r'C:\Temp\movies.csv')
ratings = pd.read_csv(r'C:\Temp\ratings.csv')


movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])
genre_sim_matrix = cosine_similarity(genre_matrix)


def build_matrix(ratings_df):
    user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_matrix.fillna(0, inplace=True)
    item_similarity = cosine_similarity(user_movie_matrix.T)
    return pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

item_similarity_df = build_matrix(ratings)

def load_profile(user_id):
    path = f'C:\\Temp\\user_{user_id}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'user_id': user_id, 'ratings': {}, 'favorites': [], 'history': []}

def save_profile(user):
    with open(f'C:\\Temp\\user_{user["user_id"]}.json', 'w') as f:
        json.dump(user, f, indent=2)


def retrain_similarity(new_ratings_df):
    global item_similarity_df
    new_df = pd.concat([ratings, new_ratings_df], ignore_index=True)
    item_similarity_df = build_matrix(new_df)

def recommend(user_data, include_genres=None, exclude_genres=None, top_n=5):
    favorites = user_data['favorites']
    if not favorites:
        print("Add your liked films for recommendation.")
        return pd.DataFrame()

    fav_ids = []
    for fav in favorites:
        match = movies[movies['title'].str.lower() == fav.lower()]
        if not match.empty:
            fav_ids.append(match.movieId.values[0])

    scores = pd.Series(0, index=item_similarity_df.index, dtype='float64')
    for fid in fav_ids:
        if fid in item_similarity_df.columns:
            scores += item_similarity_df[fid]


    if include_genres or exclude_genres:
        genre_filter = movies.copy()
        if include_genres:
            genre_filter = genre_filter[genre_filter['genres'].str.contains('|'.join(include_genres), case=False)]
        if exclude_genres:
            genre_filter = genre_filter[~genre_filter['genres'].str.contains('|'.join(exclude_genres), case=False)]
        scores = scores[scores.index.isin(genre_filter.movieId)]

    scores = scores.drop(labels=fav_ids, errors='ignore')
    top_ids = scores.sort_values(ascending=False).head(top_n).index
    return movies[movies['movieId'].isin(top_ids)][['title', 'genres']]


def ex_sess(user_data, recommendations, filename):
    export = {
        "user_id": user_data["user_id"],
        "favorites": user_data["favorites"],
        "filters": user_data.get("filters", {}),
        "recommendations": recommendations,
        "timestamp": datetime.datetime.now().isoformat()
    }
    export_path = f'C:\\Temp\\{filename}'
    with open(export_path, 'w', encoding='utf-8') as f:
        if filename.endswith('.json'):
            json.dump(export, f, indent=2, ensure_ascii=False)
        else:
            f.write(json.dumps(export, indent=2, ensure_ascii=False))
    print(f"Title export in: {export_path}")


def cli():
    user_id = input("Your ID: ")
    user = load_profile(user_id)

    while True:
        print("\n[1] Add score\n[2] Add liked\n[3] Add genres\n[4] Recommendations\n[5] Export film\n[6] Out")
        choice = input("")

        if choice == '1':
            movie_title = input("Move title : ")
            rating = float(input("Score (0-5): "))
            match = movies[movies['title'].str.lower() == movie_title.lower()]
            if not match.empty:
                movie_id = int(match.movieId.values[0])
                user['ratings'][str(movie_id)] = rating
                print("Score added.")
            else:
                print("Film is not found.")

        elif choice == '2':
            fav = input("Add film title of your favourite genre: ")
            user['favorites'].append(fav)

        elif choice == '3':
            include = input("Turn on genres: ").split(',')
            exclude = input("Turn off genres: ").split(',')
            user['filters'] = {"include": [g.strip() for g in include if g.strip()],
                               "exclude": [g.strip() for g in exclude if g.strip()]}

        elif choice == '4':
            inc = user.get('filters', {}).get('include', [])
            exc = user.get('filters', {}).get('exclude', [])
            recs = recommend(user, include_genres=inc, exclude_genres=exc)
            if not recs.empty:
                print("\nRecomendations titles:")
                print(recs.to_string(index=False))
                user['history'].append({
                    "time": datetime.datetime.now().isoformat(),
                    "input": user['favorites'],
                    "filters": user.get('filters', {}),
                    "results": recs['title'].tolist()
                })

        elif choice == '5':
            filename = input("Name file for export : ")
            inc = user.get('filters', {}).get('include', [])
            exc = user.get('filters', {}).get('exclude', [])
            recs = recommend(user, include_genres=inc, exclude_genres=exc)
            if not recs.empty:
                ex_sess(user, recs['title'].tolist(), filename)
            else:
                print ("No recommendations to export.")
        elif choice == '6':
            save_profile(user)
            print("Profile saved. Out.")
            break

        else:
            print("None. Try again.")


if __name__ == "__main__":
    cli()
