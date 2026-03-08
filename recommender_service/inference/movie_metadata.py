import pandas as pd
from pathlib import Path

class MovieMetadata:

    def __init__(self, movie_file_path):
        movie_file_path_x = Path(movie_file_path)
        self.movies_df = pd.read_csv(
            movie_file_path_x,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"]
        )

        self.movie_dict = self.movies_df.set_index("movie_id")["title"].to_dict()

    def get_title(self, movie_id):

        return self.movie_dict.get(movie_id, "Unknown Movie")