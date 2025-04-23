class BaseDataLoader:
    """
    Base class for loading and processing ICESat-2 data.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, beam):
        """
        Method to load data for a specific beam.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
