import pandas as pd
import requests
import shutil
from PIL import Image
import numpy as np
import os
import json
import pprint
import random
import copy

#put your name here, so each file is different
CSV_FILENAME = 'MovieGenre'

def get_image_http(link, imdbId):
    res = requests.get(link, stream=True)
    if res.status_code == 404:
        return False
    with open('./images/' + str(imdbId) + '.png', 'wb') as f:
        shutil.copyfileobj(res.raw, f)
    
    try:
        np.asarray(Image.open('./images/' + str(imdbId) + '.png'))
        return True
    except:
        os.remove('./images/' + str(imdbId) + '.png')
        return False

def get_image_filedir(imdbId):
    try:
        Image.open('./images/' + str(imdbId) + '.png')
        return True
    except:
        return False

def main():
    df = pd.read_csv('./movies/' + CSV_FILENAME + '.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df = df.drop_duplicates(subset=['imdbId'])

    genre_labels = json.load(open('./data/genreLabels.json'))
    count = {}
    genre_data = {}
    single_data = {}
    single_count = {}
    NUM_CLASSES = 6
    # accepting_labels = ['Crime', 'Action', 'Romance', 'Comedy', 'Drama']
    accepting_labels = ['Drama', 'Documentary', 'Comedy', 'Action', 'Thriller', 'Horror']
    
    for label in genre_labels:
        count[label] = 0
        single_count[label] = 0

    i = 0
    for imdbId in df['imdbId']:
        # i += 1
        row = df[df['imdbId'] == imdbId]
        if get_image_filedir(imdbId):
            movie_genres = row['Genre'].values[0].split('|')
            
            bool = False
            for genre in movie_genres:
                if count[genre] > 2850:
                    bool = True
                    break
            if bool:
                continue
            
            label_vector = [1.0 if label in movie_genres else 0.0 for label in accepting_labels]
            if label_vector != [0,0,0,0,0,0]:
                genre_data[imdbId] = label_vector
                temp_labels = copy.deepcopy(label_vector)
                one_indices = [i for i, val in enumerate(temp_labels) if val == 1]
    
                if one_indices:
                    # Choose a random index from one_indices
                    random_index = random.choice(one_indices)

                    # Set all indices with 1 to 0, except the randomly chosen index
                    for i in range(len(temp_labels)):
                        if i != random_index:
                            temp_labels[i] = 0
                    single_count[accepting_labels[random_index]] += 1
                single_data[imdbId] = temp_labels
                
                i+=1
            
            # label_vector = [1.0 if label in movie_genres else 0.0 for label in genre_labels]
            # if label_vector.count(0.0) == len(genre_labels) - 1:
            #     genre_data[imdbId] = label_vector
            
            for genre in movie_genres:
                count[genre] += 1
            
                
    print("number of distict posters:", i)
    pprint.pp(dict(sorted(count.items(), key=lambda item: item[1])))
    print("********************************************************")
    pprint.pp(dict(sorted(single_count.items(), key=lambda item: item[1])))
    with open('./new_data/Top6GenreData.json', 'w') as f:
        json.dump(genre_data, f)
    with open('./new_data/SingleGenreData.json', 'w') as f:
        json.dump(single_data, f)

if __name__ == "__main__":
    main()