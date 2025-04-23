import pandas as pd


class Preprocessor:
    def __init__(self, data):
        """
        Initialize the Preprocessor with the data.

        Args:
        data (pd.DataFrame): DataFrame containing the loaded photon and segment data.
        """
        self.data = data
        self.data = self.filter_ndwi()

    def separate_beams(self):
        """
        Separate the DataFrame into individual beams identified by their unique combination of rgt, cycle, and gtxx.
        Normalize the x_ph values on a per-profile basis.

        Returns:
        dict: Dictionary where keys are (rgt, cycle, gtxx) tuples and values are DataFrames.
        """
        beam_groups = self.data.groupby(["rgt", "cycle", "spot"])
        separated_beams = {}
        for key, group in beam_groups:
            group = group.copy()
            group["x_ph"] = group["x_ph"] - group["x_ph"].min()
            separated_beams[key] = group
        return separated_beams

    def filter_ndwi(self):
        """
        Check if there are NDWI values different from the default and filter out non-water data.

        Returns:
        pd.DataFrame: Filtered DataFrame, if needed.
        """
        if "ndwi" in self.data.columns and not (self.data["ndwi"] == 0).all():
            print("Filtering NDWI values to retain only water-related data.")
            return self.data[self.data["ndwi"] > 0]
        return self.data


# Example usage
if __name__ == "__main__":
    # Assuming `data` is the DataFrame loaded by the data loader
    data = pd.read_csv("path_to_csv_file")

    preprocessor = Preprocessor(data)
    separated_beams = preprocessor.separate_beams()

    for key, df in separated_beams.items():
        print(f"Beam {key}:")
        print(df.head())
