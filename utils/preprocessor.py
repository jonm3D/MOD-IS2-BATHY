import pandas as pd


class Preprocessor:
    def __init__(self, data):
        """
        Initialize the Preprocessor with the data.

        Args:
        data (pd.DataFrame): DataFrame containing the loaded photon and segment data.
        """
        self.data = data

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

