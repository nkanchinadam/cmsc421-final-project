import pandas as pd 
import json 
from sklearn.model_selection import train_test_split
from PIL import Image

genre_data = json.load(open('./new_data/Top5GenreData.json', 'r'))
file_path = []
classes = []
count = 0
for id in genre_data.keys():
  count += 1
  print(count)
  try:
    Image.open('./images/' + id + '.png').convert('RGB').save("./images/" + id + ".png", )
    file_path.append("./images/" + id + ".png")
    classes.append(genre_data[id])
  except:
    continue

file_path_train, file_path_test, classes_train, classes_test = train_test_split(file_path, classes, test_size=0.2, random_state=1)
file_path_train, file_path_val, classes_train, classes_val = train_test_split(file_path_train, classes_train, test_size=0.25, random_state=1)



df1 = pd.DataFrame(data={"path": file_path_train, "label": classes_train})
df2 = pd.DataFrame(data={"path": file_path_val, "label": classes_val})
df3 = pd.DataFrame(data={"path": file_path_test, "label": classes_test})
df1.to_csv("PathAndClassTrain.csv")
df2.to_csv("PathAndClassVal.csv")
df3.to_csv("PathAndClassTest.csv")
