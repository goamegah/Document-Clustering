import warnings


class Jar(dict):
    """
    Container object exposing keys as attributes.
    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.
    Examples
    --------
    # >>> from src.utils import Jar
    # >>> b = Bunch(a=1, b=2)
    # >>> b['b']
    # 2
    # >>> b.b
    # 2
    # >>> b.a = 3
    # >>> b['a']
    # 3
    # >>> b.c = 6
    # >>> b['c']
    # 6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

        # Map from deprecated key to warning message
        self.__dict__["_deprecated_key_to_warnings"] = {}

    def __getitem__(self, key):
        if key in self.__dict__.get("_deprecated_key_to_warnings", {}):
            warnings.warn(
                self._deprecated_key_to_warnings[key],
                FutureWarning,
            )
        return super().__getitem__(key)

    def _set_deprecated(self, value, *, new_key, deprecated_key, warning_message):
        """Set key in dictionary to be deprecated with its warning message."""
        self.__dict__["_deprecated_key_to_warnings"][deprecated_key] = warning_message
        self[new_key] = self[deprecated_key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass
