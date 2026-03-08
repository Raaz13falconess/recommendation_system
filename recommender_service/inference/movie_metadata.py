import pandas as pd


class MovieMetadata:

    def __init__(self, movie_file_path):

        self.movies_df = pd.read_csv(
            movie_file_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"]
        )

        self.movie_dict = self.movies_df.set_index("movie_id")["title"].to_dict()

    def get_title(self, movie_id):

        return self.movie_dict.get(movie_id, "Unknown Movie")