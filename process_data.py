import pandas as pd
import requests
import shutil
from PIL import Image
import numpy as np
import os
import json

def get_image(link, imdbId):
    res = requests.get(link, stream=True)
    if res.status_code == 404:
        return np.array([])
    with open('./movies/' + str(imdbId) + '.png', 'wb') as f:
        shutil.copyfileobj(res.raw, f)
    image = np.asarray(Image.open('./movies/' + str(imdbId) + '.png'))
    os.remove('./movies/' + str(imdbId) + '.png')
    return image

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df = df.drop_duplicates(subset=['imdbId'])
    pixelData = {}
    for imdbId in df['imdbId']:
        row = df[df['imdbId'] == imdbId]
        image = get_image(row['Poster'].values[0], imdbId)
        if not np.array_equal(image, np.array([])):
            pixelData[imdbId] = image

    with open('pixelData.json', 'w') as f:
        json.dump(pixelData, f)
    
if __name__ == "__main__":
    main()