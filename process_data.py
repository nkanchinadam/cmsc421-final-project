import pandas as pd
import requests
import shutil

def get_image(link, imdbId):
    res = requests.get(link, stream=True)
    if res.status_code == 404:
        return None
    with open('./movies/posters/' + str(imdbId) + '.png', 'wb') as f:
        shutil.copyfileobj(res.raw, f)

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df = df.drop_duplicates(subset=['imdbId'])
    df = df.apply(lambda x : x)
    print(len(df))
    print(df)
    #throws a KeyError when i == 137, not sure why
    for i in range(len(df)):
        print(i)
        get_image(df['Poster'][i], df['imdbId'][i])
    
if __name__ == "__main__":
    main()