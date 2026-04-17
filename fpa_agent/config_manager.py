import os
import json


class ConfigManager:
    """Load and manage configuration from config.json."""

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        return self._default_config()

    def _default_config(self):
        return {
            "model": {
                "pth_path": "",
                "exp_path": "",
                "classes_path": "",
                "device": "cpu"
            },
            "video": {"fps": 33},
            "ui": {
                "window_width": 1100,
                "window_height": 720,
                "display_confidence": True,
                "box_thickness": 2,
                "text_size": 0.5
            },
            "export": {"default_format": "YOLO"}
        }

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
