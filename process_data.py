import pandas as pd
import requests
import shutil
from PIL import Image
import numpy as np
import os
import json

#put your name here, so each file is different
CSV_FILENAME = 'Nitin'

def get_image(link, imdbId):
    res = requests.get(link, stream=True)
    if res.status_code == 404:
        return np.array([])
    with open('./movies/' + str(imdbId) + '.png', 'wb') as f:
        shutil.copyfileobj(res.raw, f)
    
    image = None
    try:
        image = np.asarray(Image.open('./movies/' + str(imdbId) + '.png'))
    except:
        image = np.array([])
    os.remove('./movies/' + str(imdbId) + '.png')
    return image

def main():
    df = pd.read_csv('./movies/' + CSV_FILENAME + '.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df = df.drop_duplicates(subset=['imdbId'])

    genre_labels = json.load(open('./data/genreLabels.json'))

    pixel_data = {}
    genre_data = {}
    i = 0
    for imdbId in df['imdbId']:
        print(i)
        i += 1
        row = df[df['imdbId'] == imdbId]
        image = get_image(row['Poster'].values[0], imdbId)
        if not np.array_equal(image, np.array([])):
            pixel_data[imdbId] = image.tolist()
            movie_genres = row['Genre'].values[0].split('|')
            genre_data[imdbId] = [1.0 if label in movie_genres else 0.0 for label in genre_labels]

    with open('./data/pixelData' + CSV_FILENAME + '.json', 'w') as f:
        json.dump(pixel_data, f)
    with open('./data/genreData' + CSV_FILENAME + '.json', 'w') as f:
        json.dump(genre_data, f)

if __name__ == "__main__":
    main()