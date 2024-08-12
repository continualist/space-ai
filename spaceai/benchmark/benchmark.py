import os


class Benchmark:

    def __init__(self, run_id: str, exp_dir: str):
        """Initializes a new benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
        """
        self.run_id = run_id
        self.exp_dir = exp_dir

    @property
    def run_dir(self) -> str:
        """Returns the directory where the results of this run are stored."""
        return os.path.join(self.exp_dir, self.run_id)
