from ..base import ProjectBase


class DatasetBase(ProjectBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def get_deps(self):
        return []

    def load(self):
        """Do nothing, since the generator doesn't read in data.

        This is part of the ProjectBase functions so this dummy
        implementation should suffice for most generators."""
        pass

    def save(self):
        import torch

        # mark the top-level for the generated data.
        (self.path / "data.root").touch(exist_ok=True)
        save_data = self.data_sd
        self.save_lambda_alt(self.outdir / "dataset.pt", save_data, torch.save)
