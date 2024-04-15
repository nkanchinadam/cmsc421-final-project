import pandas as pd
import json

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1').dropna().drop_duplicates(subset=['imdbId'])
    genre_labels = set()
    for imdbId in df['imdbId']:
        row = df[df['imdbId'] == imdbId]
        genres = row['Genre'].values[0].split('|')
        for genre in genres:
            genre_labels.add(genre)
    genre_labels = sorted(list(genre_labels))
    with open('./data/genreLabels.json', 'w') as f:
        json.dump(genre_labels, f)

if __name__ == '__main__':
    main()