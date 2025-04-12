import json
import dpath.util
from .default_config import DEFAULT_CONFIG

class Config:
    """Configuration management for experiments using dpath for nested access."""

    def __init__(self, config_dict=None):
        """Initialize with default or provided configuration."""
        self.config = config_dict or {}
        self._set_defaults()

    def _set_defaults(self):
        """
        Merge the default configuration into self.config using dpath.

        The merge is performed with overwrite=False, meaning that only missing keys
        from DEFAULT_CONFIG will be added; existing keys in self.config remain unchanged.
        """
        dpath.util.merge(self.config, DEFAULT_CONFIG, overwrite=False)

    def get(self, path, sep='.'):
        """
        Retrieve a configuration value from a nested dictionary using a unified path.

        Parameters:
            path (str): A dot-separated string (or list/tuple) representing the nested key path,
                        e.g. 'database.credentials.user'.
            sep (str): Separator used when path is a string (default is '.').

        Returns:
            The value stored at the specified nested key.

        Raises:
            KeyError: If any part of the key path is missing.
        """
        try:
            return dpath.util.get(self.config, path, separator=sep)
        except KeyError as e:
            raise KeyError(f"Failed to get nested key '{path}': {e}")

    def set(self, path, value, sep='.', override=True):
        """
        Update a configuration value using a unified nested key.

        Parameters:
            path (str): A dot-separated string (or list/tuple) representing the nested key path,
                        e.g. 'database.credentials.user'.
            value: The new value to assign.
            sep (str): Separator used when path is a string (default is '.').
            override (bool): If True (default), the method will check that the key path exists
                             and raise a KeyError if any part is missing.
                             If False, missing keys will be created automatically.

        Raises:
            KeyError: If override is False and any portion of the key path is missing.
        """
        if not override:
            # Verify that the nested key exists; this will raise a KeyError if it doesn't.
            try:
                dpath.util.get(self.config, path, separator=sep)
            except KeyError as e:
                raise KeyError(f"Cannot set value for nested key '{path}' because it does not exist: {e}")
        
        # dpath.util.set will create missing keys if override is True.
        dpath.util.set(self.config, path, value, separator=sep)

    # Allow bracket-notation for getting values.
    def __getitem__(self, key):
        """
        Enable retrieval via bracket notation.
        
        Example:
            config['database.credentials.user']  -> returns the nested value.
        """
        return self.get(key)

    # Allow bracket-notation for setting values.
    def __setitem__(self, key, value):
        """
        Enable setting via bracket notation.
        
        Example:
            config['database.credentials.user'] = "new_user"
            
        By default, this operation will fail if any portion of the path is missing.
        To override and allow creating missing keys, you could call set with override=True.
        """
        self.set(key, value)

    def save(self, filename='experiment_config.json'):
        """Save the configuration to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, filename='experiment_config.json'):
        """Load a configuration from a JSON file."""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def __str__(self):
        """Return a pretty-printed JSON representation of the configuration."""
        return json.dumps(self.config, indent=4)
