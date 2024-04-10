import pandas as pd
import requests
import shutil
from PIL import Image
import numpy as np
import os

def get_image(link, imdbId):
    res = requests.get(link, stream=True)
    if res.status_code == 404:
        return None
    with open('./movies/' + str(imdbId) + '.png', 'wb') as f:
        shutil.copyfileobj(res.raw, f)
    image = np.asarray(Image.open('./movies/' + str(imdbId) + '.png'))
    os.remove('./movies/' + str(imdbId) + '.png')
    return image

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df = df.drop_duplicates(subset=['imdbId'])
    print(len(df))
    print(df)
    #throws a KeyError when i == 137, not sure why
    for i in range(len(df)):
        print(i)
        get_image(df['Poster'][i], df['imdbId'][i])
    
if __name__ == "__main__":
    main()