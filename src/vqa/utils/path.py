import os
import re
import typing


class Path:
    """Utility class to handle strings

    Methods:
        __str__: absolute path as string
        __fspath__: absolute path as path-like
        __repr__: same as __str__

    Attributes:
        _rel_path: relative path from ml-v3 directory
        _abs_path: absolute path on local machine

    """

    _is_initiated_cwd = True

    def __init__(
        self,
        path_like: typing.Union[str, typing.List[str], "os.PathLike"],
        mkdir: bool = False,
    ):
        if Path.init_cwd():
            if isinstance(path_like, str):
                if os.path.sep in path_like:
                    path_like = path_like.split(os.path.sep)
                elif os.path.altsep in path_like:
                    path_like = path_like.split(os.path.altsep)
                else:
                    path_like = [path_like]
            elif isinstance(path_like, os.PathLike):
                path_like = [path_like.__fspath__()]
            elif not isinstance(path_like, list):
                raise FileNotFoundError(
                    "Cannot resolve path! Please provide path either as str or list."
                )

            self._rel_path = os.path.join(*path_like)
            self._abs_path = os.path.join(*([os.getcwd()] + path_like))

            if mkdir and not os.path.isfile(self._abs_path):
                os.makedirs(self._abs_path, exist_ok=True)

    def __str__(self):
        return self._abs_path

    def __repr__(self):
        return self._abs_path

    def __fspath__(self):
        return self._abs_path

    @staticmethod
    def init_cwd() -> bool:
        """Initialize the current working directory

        Returns:
            True if working directory could be set or is already set

        """
        if Path._is_initiated_cwd:
            return True
        else:
            try:
                cwd = re.split(r"src", os.getcwd())
                os.chdir(os.path.join(*cwd[:-1]))
                Path._is_initiated_cwd = True
                return True
            except FileNotFoundError:
                return False
