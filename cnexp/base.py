import abc
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


class ProjectBase(abc.ABC):
    """The abstract base class for this project.

    The function defines the stubs for the functions that are expected
    to be implemented by the subclasses.  Additionally, the class
    defines the classmethod `from_string` that will instantiate the
    correct subclass, based on the string."""

    def __init__(self, path, random_state=None):
        self.path = Path(path)

        self.indir = self.path.parent
        self.outdir = self.path / "out"

        if isinstance(random_state, np.random._generator.Generator):
            self.random_state = random_state
        elif random_state is None:
            self.random_state = np.random.default_rng(0xDEADBEEF)
        elif isinstance(random_state, int) and random_state > 0:
            self.random_state = np.random.default_rng(random_state)
        else:
            raise ValueError(
                f"Expected random_state or seed, but got {random_state = }"
            )

        # will also create self.path
        self.outdir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def get_deps(self):
        """Returns a list of files that need to be present and up-to-date.

        This function is used to keep track of what files need to be
        updated in order to keep the state of the generated artifacts
        up to date."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self):
        """Access the file system and load the files into memory.

        Ideally this should use the variables from self.get_datadeps()
        in order to make the dependency explicit and not cause a
        difference between what's declared a dependency and what is
        actually loaded."""
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()

    def __call__(self):
        self.load()
        d = self.compute()
        self.save()
        return d

    @staticmethod
    def save_lambda(fname, data, function, openmode="wb"):
        """Saves the file via the specified lambda function.

        The rationale for this function is that if we write to the
        files that we hardlinked to, `redo` will notice a change in
        the source file and error out.  This way we write to a
        tempfile and atomically move it to a location that we will
        hardlink to.  This way we get a new inode that is not linked
        to by the target that we want to create at a later point.

        Parameters
        ----------

        fname : str or path-like
        The name that the file will be written to in the end.
        data
        This will be passed to the parameter `function` and will
        through this be written to the disk.
        function : function or lambda
        The function that will write to the opened file.  Use
        `np.save` if you want to save a numpy array.
        mode : str
        The mode for opening the file.  Will be passed onto
        `NamedTemporaryFile` as is for opening."""
        return save_lambda(fname, data, function, openmode)

    @classmethod
    def save_lambda_alt(cls, fname, data, function, openmode="wb"):
        """Swap the file and data arguments for save_lambda.

        For now I am only using this or torch.save, where the order is
        the other way around.
        """
        return cls.save_lambda(
            fname, data, lambda f, d: function(d, f), openmode=openmode
        )


def save_lambda(fname, data, function, openmode="wb"):
    """Saves the file via the specified lambda function.

    The rationale for this function is that if we write to the
    files that we hardlinked to, `redo` will notice a change in
    the source file and error out.  This way we write to a
    tempfile and atomically move it to a location that we will
    hardlink to.  This way we get a new inode that is not linked
    to by the target that we want to create at a later point.

    Parameters
    ----------

    fname : str or path-like
    The name that the file will be written to in the end.
    data
    This will be passed to the parameter `function` and will
    through this be written to the disk.
    function : function or lambda
    The function that will write to the opened file.  Use
    `np.save` if you want to save a numpy array.
    mode : str
    The mode for opening the file.  Will be passed onto
    `NamedTemporaryFile` as is for opening."""

    fname = Path(fname)
    with NamedTemporaryFile(
        openmode, dir=fname.parent, suffix=fname.suffix, delete=False
    ) as tempf:
        function(tempf, data)
        return Path(tempf.name).replace(fname)
