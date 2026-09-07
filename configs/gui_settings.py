import json
import os

from tools.file_io import read_text

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

PRESET_KEYS = (
    "model_root",
    "model_identity",
    "sg_hostapi",
    "sg_wasapi_exclusive",
    "sg_input_device",
    "sg_output_device",
    "sr_type",
    "threhold",
    "pitch",
    "formant",
    "index_rate",
    "rms_mix_rate",
    "block_time",
    "crossfade_length",
    "extra_time",
    "f0method",
    "debug",
    "I_noise_reduce",
    "O_noise_reduce",
)

DEFAULTS = {
    "model_root": "",
    "model_identity": "",
    "sg_hostapi": "",
    "sg_wasapi_exclusive": False,
    "sg_input_device": "",
    "sg_output_device": "",
    "sr_type": "sr_model",
    "threhold": -60,
    "pitch": 0,
    "formant": 0.0,
    "index_rate": 0,
    "rms_mix_rate": 0,
    "block_time": 0.25,
    "crossfade_length": 0.05,
    "extra_time": 2.5,
    "f0method": "rmvpe",
    "debug": False,
    "I_noise_reduce": False,
    "O_noise_reduce": False,
    "preset": "",
}


def with_derived(data):
    data = dict(data)
    if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
        data["f0method"] = "rmvpe"
    data["sr_model"] = data.get("sr_type") == "sr_model"
    data["sr_device"] = data.get("sr_type") == "sr_device"
    data["pm"] = data["f0method"] == "pm"
    data["rmvpe"] = data["f0method"] == "rmvpe"
    data["fcpe"] = data["f0method"] == "fcpe"
    return data


def build(**overrides):
    data = dict(DEFAULTS)
    data.update(overrides)
    return with_derived(data)


def read():
    try:
        data = json.loads(read_text(PATH))
    except:
        return {}
    return data if isinstance(data, dict) else {}


def save(data):
    with open(PATH, "w", encoding="utf8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def update(**changes):
    try:
        data = build(**read())
    except Exception:
        data = build()
    data.update(changes)
    data = with_derived(data)
    save(data)
    return data


def from_values(values):
    data = {key: values[key] for key in DEFAULTS if key in values}
    data["sr_type"] = ["sr_model", "sr_device"][[values["sr_model"], values["sr_device"]].index(True)]
    data["f0method"] = ["pm", "rmvpe", "fcpe"][[values["pm"], values["rmvpe"], values["fcpe"]].index(True)]
    data = build(**data)
    presets = read().get("presets")
    data["presets"] = presets if isinstance(presets, dict) else {}
    return data


def read_presets():
    presets = read().get("presets")
    return presets if isinstance(presets, dict) else {}


def snapshot(values):
    data = from_values(values)
    return {key: data[key] for key in PRESET_KEYS}


def loaded_preset(snap):
    return build(**{key: snap[key] for key in PRESET_KEYS if isinstance(snap, dict) and key in snap})
