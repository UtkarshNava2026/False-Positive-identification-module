import os
import json


class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = os.path.abspath(config_path)
        self.config = self._load_config()

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        return self._default_config()

    def _default_config(self):
        return {
            "model": {
                "path": "",
                "pth_path": "",
                "exp_path": "",
                "classes_path": "",
                "device": "cpu",
            },
            "drift": {
                "reference_path": "embeddings.npy",
                "encoder": "yolox_standard",
                "input_size": [640, 640],
                "knn_sample_size": 2048,
            },
            "video": {"fps": 0, "frame_step": 1, "offline_fps": 0.0},
            "ui": {"window_width": 1280, "window_height": 800},
            "export": {"default_format": "YOLO"},
        }

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key, value):
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
