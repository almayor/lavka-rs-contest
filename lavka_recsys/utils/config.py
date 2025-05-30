from __future__ import annotations

import copy
import os
import json
import dpath
import yaml

from typing import Any

class Config:
    """Immutable configuration management for experiments using dpath for nested access.

    This version loads default configuration from a YAML or JSON file.
    Any modification returns a new Config instance; the original is remains unchanged.
    """

    def __init__(self, config_dict=None, default_config_path=None):
        """
        Initialize with provided configuration, merging with defaults loaded from a file.

        Parameters:
            config_dict (dict, optional): Custom configuration dictionary.
            default_config_path (str, optional): Path to the default configuration file.
                If not provided, it defaults to a file named 'default_config.yaml' in the same
                directory as this module. It will fall back to 'default_config.json' if the YAML file
                is not found.
        """
        self._config = copy.deepcopy(config_dict) if config_dict is not None else {}
        self._set_defaults(default_config_path=default_config_path)

    def _set_defaults(self, default_config_path=None):
        """
        Merge default configuration into self._config using dpath.

        This method loads the default configuration from a YAML or JSON file and merges them into
        the existing configuration without overwriting keys that are already present.

        Parameters:
            default_config_path (str, optional): Path to the default config file.
                If not provided, defaults to 'default_config.yaml' in the module directory.
        """
        if default_config_path is None:
            # Try default_config.yaml first; if not found, fall back to default_config.json.
            yaml_path = os.path.join(os.getcwd(), 'default_config.yaml')
            json_path = os.path.join(os.getcwd(), 'default_config.json')
            if os.path.exists(yaml_path):
                default_config_path = yaml_path
            elif os.path.exists(json_path):
                default_config_path = json_path
            else:
                raise FileNotFoundError("No default configuration file found (neither YAML nor JSON).")
        
        try:
            with open(default_config_path, 'r') as f:
                if default_config_path.lower().endswith(('.yaml', '.yml')):
                    default_config = yaml.safe_load(f)
                else:
                    default_config = json.load(f)
        except Exception as e:
            raise FileNotFoundError(
                f"Unable to load default configuration from {default_config_path}: {e}"
            )
        
        # Merge defaults into the current configuration.
        self._config = dpath.merge(default_config, self._config, flags=dpath.MergeType.REPLACE)

    def get(self, path, default=None, sep='.') -> Any:
        """
        Retrieve a configuration value from a nested dictionary using a unified path.

        Parameters:
            path (str): A dot-separated string representing the nested key path,
                        e.g. 'database.credentials.user'.
            default: Default value to return if the key is not found.
            sep (str): Separator used (default is '.').

        Returns:
            The value stored at the specified key path.
        """
        try:
            return dpath.get(self._config, path, separator=sep)
        except KeyError:
            return default

    def set(self, path, value, sep='.', override=True) -> Config:
        """
        Return a new Config with the updated value at `path`.

        Parameters:
            path (str): A dot-separated string representing the nested key path.
            value: The new value to assign.
            sep (str): Separator used (default is '.').
            override (bool): If True, missing keys will be automatically created.
                             If False, a KeyError is raised for missing keys.
        """
        new_config = copy.deepcopy(self._config)
        if not override:
            try:
                dpath.get(self._config, path, separator=sep)
            except KeyError as e:
                raise KeyError(f"Cannot set value for nested key '{path}' because it does not exist: {e}")
        
        dpath.set(new_config, path, value, separator=sep)
        return Config(config_dict=new_config)

    def __getitem__(self, key, sep='.', ) -> Any:
        """Enable retrieval via bracket notation. Fail if key is not found."""
        try:
            return dpath.get(self._config, key, separator=sep)
        except KeyError:
            raise KeyError(f"Key '{key}' not found in configuration")

    def __setitem__(self, key, value):
        raise TypeError("Config is immutable; use .set(path, value) to obtain a modified copy.")

    def save(self, filename='experiment_config.yaml'):
        """Save the configuration to a YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, filename='experiment_config.yaml'):
        """Load a configuration from a YAML file."""
        with open(filename, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def to_dict(self):
        """Return a copy of the configuration as a dictionary."""
        return copy.deepcopy(self._config)
        
    def copy(self) -> Config:
        """Create a deep copy of this configuration object."""
        return Config(config_dict=self.to_dict())

    def __str__(self):
        """Return a pretty-printed YAML representation of the configuration."""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        """Return a string representation of the configuration."""
        return f"Config({self._config})"