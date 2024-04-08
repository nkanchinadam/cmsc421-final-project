import pandas as pd

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1')
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['imdbId'], inplace=True)
    
if __name__ == "__main__":
    main()