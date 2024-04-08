import pandas as pd

def main():
    df = pd.read_csv('./movies/MovieGenre.csv', encoding='utf-8')
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['imdbId'], inplace=True)

if __name__ == "__main__":
    main()