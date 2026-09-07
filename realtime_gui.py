import os, sys, re, time, traceback, threading, queue, librosa
from tqdm import tqdm
import numpy as np

from configs import gui_settings

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "4"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

flag_vc = False
NONE_MODEL = "None"

_boot_t0 = time.perf_counter()
_step = {"t": _boot_t0, "label": None}

def _boot_step(pbar, label):
    now = time.perf_counter()
    if _step["label"] is not None:
        tqdm.write("%s (%.1fs)" % (_step["label"], now - _step["t"]))
    _step["t"] = now
    _step["label"] = label
    pbar.set_description(label)
    pbar.refresh()


if __name__ == "__main__":

    print("Booting realtime GUI...")
    with tqdm(total=7, unit="step", mininterval=0) as pbar:
        _boot_step(pbar, "Loading numpy")
        pbar.update(1)

        _boot_step(pbar, "Loading torch")
        import torch
        import torch.nn.functional as F
        import torchaudio.transforms as tat
        import torchaudio.functional as AF
        pbar.update(1)

        _boot_step(pbar, "Loading audio / GUI libs")
        import FreeSimpleGUI as sg
        import sounddevice as sd
        pbar.update(1)

        _boot_step(pbar, "Loading TorchGate")
        from tools.torchgate import TorchGate
        pbar.update(1)

        _boot_step(pbar, "Loading config (CUDA / DirectML / graph probe)")
        from configs.config import Config
        pbar.update(1)

        _boot_step(pbar, "Loading rtrvc (faiss / parselmouth / hubert)")
        from infer import rtrvc as rvc_for_realtime
        pbar.update(1)

        _boot_step(pbar, "Loading i18n / cuda_graph")
        from i18n.i18n import I18nAuto
        from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
        pbar.update(1)
        tqdm.write("%s (%.1fs)" % (_step["label"], time.perf_counter() - _step["t"]))

    print(f"Modules loaded in {time.perf_counter() - _boot_t0:.1f}s")

    i18n = I18nAuto()

class GUIConfig:
    def __init__(self) :
        self.pth_path = ""
        self.index_path = ""
        self.pitch = 0
        self.formant=0.0
        self.sr_type = "sr_model"
        self.block_time = 0.25  # s
        self.threhold = -60
        self.crossfade_time = 0.05
        self.extra_time = 2.5
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.0
        self.f0method = "rmvpe"
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""
        self.sg_output_device_2 = NONE_MODEL
        self.debug = False

class GUI:
    def __init__(self) :
        print("Initializing GUI...")
        self.gui_config = GUIConfig()
        self.config = Config()
        print(f"RVC_CUDA_GRAPH={os.environ.get('RVC_CUDA_GRAPH', '0')}")
        self.rvc = None
        self.change_voice = False
        self.low_latency_stream = False
        self.last_infer_ms = 0
        self.delay_time = 0
        self.needs_reload = False
        self.reloading = False
        self.closing = False
        self.reload_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.last_values = {}
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.stream = None
        self.stream2 = None
        self.monitor_q = None
        self.update_devices()
        self.launcher()

    def load(self):
        print(f"Loading settings from {gui_settings.PATH}")
        raw = gui_settings.read()
        if not raw:
            print("No valid settings file, using defaults")
        data = gui_settings.build(**raw)
        if data["sg_hostapi"] in self.hostapis:
            self.update_devices(hostapi_name=data["sg_hostapi"])
            if data["sg_input_device"] not in self.input_devices or data["sg_output_device"] not in self.output_devices:
                self.update_devices()
                data["sg_hostapi"] = self.hostapis[0]
                data["sg_input_device"] = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
                data["sg_output_device"] = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]
        else:
            data["sg_hostapi"] = self.hostapis[0]
            data["sg_input_device"] = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
            data["sg_output_device"] = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]
        if data.get("sg_output_device_2") not in self.output2_choices():
            data["sg_output_device_2"] = NONE_MODEL
        if not raw:
            gui_settings.save(data)
        return data

    def scan_model_root(self, root):
        models = {}
        try:
            names = sorted(os.listdir(root))
        except:
            return models
        for name in names:
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            try:
                files = sorted(os.listdir(folder))
            except:
                continue
            pths = [os.path.join(folder, f) for f in files if f.lower().endswith(".pth")]
            idxs = [os.path.join(folder, f) for f in files if f.lower().endswith(".index")]
            if pths and idxs:
                models[name] = (pths[0], idxs[0])
        return models

    def model_identity_list(self, root):
        return [NONE_MODEL] + [name for name in self.scan_model_root(root) if name != NONE_MODEL]

    def is_none_model(self, values=None):
        src = self.last_values if values is None else values
        return (src.get("model_identity") or "") == NONE_MODEL

    def output2_choices(self):
        return [NONE_MODEL] + list(self.output_devices or [])

    def persist_model_root(self, root, identity=None):
        if not root:
            try:
                root = self.window["model_root"].get()
            except:
                root = ""
        if not root:
            return
        changes = {"model_root": root}
        if identity is not None:
            changes["model_identity"] = identity
        try:
            gui_settings.update(**changes)
        except Exception as e:
            print(f"Failed to save model_root: {e}")

    def refresh_preset_combo(self, selected=""):
        names = sorted(gui_settings.read_presets())
        if selected not in names:
            selected = ""
        self.window["preset"].update(values=names, value=selected)

    def prompt_new_preset_name(self):
        layout = [
            [sg.Text(i18n("Preset name")), sg.Input(key="new_preset_name", size=(32, 1), focus=True)],
            [sg.Button(i18n("Cancel"), key="new_preset_cancel"), sg.Button(i18n("Save"), key="new_preset_save")],
        ]
        win = sg.Window(i18n("Create New Preset"), layout, modal=True, finalize=True)
        name = None
        while True:
            event, values = win.read()
            if event in (sg.WIN_CLOSED, "new_preset_cancel"):
                break
            if event == "new_preset_save":
                name = (values.get("new_preset_name") or "").strip()
                break
        win.close()
        return name

    def apply_preset(self, name, values):
        global flag_vc
        snap = gui_settings.read_presets().get(name)
        if not name or not isinstance(snap, dict):
            return
        data = gui_settings.loaded_preset(snap)
        cur_in = values.get("sg_input_device", "")
        cur_out = values.get("sg_output_device", "")
        cur_out2 = values.get("sg_output_device_2") or NONE_MODEL
        hostapi = data.get("sg_hostapi", "")
        if hostapi and hostapi != values.get("sg_hostapi"):
            if flag_vc:
                self.stop_stream()
            self.update_devices(hostapi_name=hostapi)
            if hostapi not in self.hostapis:
                hostapi = self.hostapis[0]
                data["sg_hostapi"] = hostapi
            self.window["sg_hostapi"].Update(values=self.hostapis)
            self.window["sg_input_device"].Update(values=self.input_devices)
            self.window["sg_output_device"].Update(values=self.output_devices)
            self.window["sg_output_device_2"].Update(values=self.output2_choices())
        if cur_in not in self.input_devices and self.input_devices:
            cur_in = self.input_devices[0]
        if cur_out not in self.output_devices and self.output_devices:
            cur_out = self.output_devices[0]
        if cur_out2 not in self.output2_choices():
            cur_out2 = NONE_MODEL
        self.window["sg_input_device"].Update(value=cur_in)
        self.window["sg_output_device"].Update(value=cur_out)
        self.window["sg_output_device_2"].Update(value=cur_out2)
        root = data.get("model_root", "")
        self.window["model_root"].update(root)
        identities = self.model_identity_list(root)
        identity = data.get("model_identity", "")
        if identity not in identities:
            identity = NONE_MODEL
            data["model_identity"] = identity
        self.window["model_identity"].Update(values=identities, value=identity)
        for key in ("sg_hostapi", "sg_wasapi_exclusive", "pitch", "formant", "index_rate", "rms_mix_rate", "block_time", "crossfade_length", "extra_time", "sr_model", "sr_device", "pm", "rmvpe", "fcpe", "I_noise_reduce", "O_noise_reduce", "debug"):
            self.window[key].update(data[key])
        self.window["preset"].update(name)
        merged = dict(values)
        for key in list(gui_settings.PRESET_KEYS) + ["sr_model", "sr_device", "pm", "rmvpe", "fcpe"]:
            if key in data:
                merged[key] = data[key]
        merged["run_audio"] = values.get("run_audio")
        merged["change_voice"] = values.get("change_voice")
        merged["preset"] = name
        merged["model_identity"] = identity
        merged["sg_input_device"] = cur_in
        merged["sg_output_device"] = cur_out
        merged["sg_output_device_2"] = cur_out2
        self.last_values = merged
        self.set_values(merged, require_model=False)
        if self.rvc is not None:
            self.rvc.change_key(merged["pitch"])
            self.rvc.change_formant(merged["formant"])
            self.rvc.change_index_rate(merged["index_rate"])
        gui_settings.update(**{key: data[key] for key in gui_settings.PRESET_KEYS}, preset=name)
        self.request_audio_reload()

    def launcher(self):
        data = self.load()
        model_root = data.get("model_root", "")
        identities = self.model_identity_list(model_root)
        model_identity = data.get("model_identity", "")
        if model_identity not in identities:
            model_identity = NONE_MODEL
        self.gui_config.debug = data.get("debug", False)
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        preset_names = sorted(presets)
        preset = data.get("preset", "")
        if preset not in preset_names:
            preset = ""
        sg.theme("LightBlue3")
        layout = [
            [
                sg.Frame(
                    title=i18n("Load model"),
                    layout=[
                        [
                            sg.Input(default_text=model_root, key="model_root", enable_events=True, tooltip=i18n("Folder whose direct subfolders each contain a .pth and a .index. Those subfolder names become the Model list.")),
                            sg.FolderBrowse(i18n("Select root folder"), key="select_model_root", target="model_root", enable_events=True, initial_folder=model_root if os.path.isdir(model_root) else os.path.join(os.getcwd(), "assets/weights"), tooltip=i18n("Folder whose direct subfolders each contain a .pth and a .index. Those subfolder names become the Model list.")),
                        ],
                        [
                            sg.Text(i18n("Model"), tooltip=i18n("Voice identity to load. None keeps your own voice and still applies Pitch and Gender factor. Other names use the .pth and .index inside that subfolder.")),
                            sg.Combo(identities, key="model_identity", default_value=model_identity, enable_events=True, size=(45, 1), tooltip=i18n("Voice identity to load. None keeps your own voice and still applies Pitch and Gender factor. Other names use the .pth and .index inside that subfolder.")),
                        ],
                    ],
                ),
                sg.Frame(
                    title=i18n("Presets"),
                    layout=[
                        [
                            sg.Combo(preset_names, key="preset", default_value=preset, enable_events=True, size=(28, 1), tooltip=i18n("Saved combinations of model, devices, and conversion settings. Run Audio and Change Voice are not stored.")),
                        ],
                        [
                            sg.Button(i18n("Create New Preset"), key="preset_create", tooltip=i18n("Save the current settings as a new named preset.")),
                            sg.Button(i18n("Update"), key="preset_update", tooltip=i18n("Overwrite the selected preset with the current settings.")),
                            sg.Button(i18n("Delete"), key="preset_delete", tooltip=i18n("Remove the selected preset.")),
                        ],
                    ],
                ),
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("Device type"), tooltip=i18n("Audio host API (MME, WASAPI, ASIO, …). Filters the input/output lists. WASAPI or ASIO is usually lowest latency.")),
                            sg.Combo(self.hostapis, key="sg_hostapi", default_value=data.get("sg_hostapi", ""), enable_events=True, size=(20, 1), tooltip=i18n("Audio host API (MME, WASAPI, ASIO, …). Filters the input/output lists. WASAPI or ASIO is usually lowest latency.")),
                            sg.Checkbox(i18n("Exclusive WASAPI device"), key="sg_wasapi_exclusive", default=data.get("sg_wasapi_exclusive", False), enable_events=True, tooltip=i18n("When Device type is WASAPI, take exclusive control of the device for lower latency. Other apps cannot use it at the same time.")),
                        ],
                        [
                            sg.Text(i18n("Input device"), tooltip=i18n("Capture device for your microphone / voice.")),
                            sg.Combo(self.input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), enable_events=True, size=(45, 1), tooltip=i18n("Capture device for your microphone / voice.")),
                        ],
                        [
                            sg.Text(i18n("Output device"), tooltip=i18n("Primary playback device. Use this for the virtual cable / passthrough others should hear.")),
                            sg.Combo(self.output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), enable_events=True, size=(45, 1), tooltip=i18n("Primary playback device. Use this for the virtual cable / passthrough others should hear.")),
                        ],
                        [
                            sg.Text(i18n("Output device 2"), tooltip=i18n("Optional second playback device, for example your headphones so you can hear yourself. None disables it.")),
                            sg.Combo(self.output2_choices(), key="sg_output_device_2", default_value=data.get("sg_output_device_2", NONE_MODEL), enable_events=True, size=(45, 1), tooltip=i18n("Optional second playback device, for example your headphones so you can hear yourself. None disables it.")),
                        ],
                        [
                            sg.Button(i18n("Reload device list"), key="reload_devices", tooltip=i18n("Rescan host APIs and devices after plugging something in.")),
                            sg.Radio(i18n("Use model sample rate"), "sr_type", key="sr_model", default=data.get("sr_model", True), enable_events=True, tooltip=i18n("Run the stream at the voice model's trained sample rate (for example 40 kHz or 48 kHz).")),
                            sg.Radio(i18n("Use device sample rate"), "sr_type", key="sr_device", default=data.get("sr_device", False), enable_events=True, tooltip=i18n("Run at the audio device's default sample rate instead. Can avoid resampling problems on some devices.")),
                            sg.Text(i18n("Sample rate:"), tooltip=i18n("Sample rate actually used after you start conversion.")),
                            sg.Text("", key="sr_stream"),
                        ],
                    ],
                    title=i18n("Audio device"),
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("Response threshold"), tooltip=i18n("Silence gate in dB. Chunks quieter than this are muted so the converter stays quiet. -60 turns the gate off.")),
                            sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -60), enable_events=True, tooltip=i18n("Silence gate in dB. Chunks quieter than this are muted so the converter stays quiet. -60 turns the gate off.")),
                        ],
                        [
                            sg.Text(i18n("Pitch settings"), tooltip=i18n("Transpose in semitones applied to detected pitch. +12 is one octave up, -12 one octave down.")),
                            sg.Slider(range=(-16, 16), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), enable_events=True, tooltip=i18n("Transpose in semitones applied to detected pitch. +12 is one octave up, -12 one octave down.")),
                        ],
                        [
                            sg.Text(i18n("Gender factor / voice thickness"), tooltip=i18n("Formant shift in semitones. Changes apparent vocal-tract size / brightness separately from pitch. Positive often sounds thinner or higher; negative thicker or lower.")),
                            sg.Slider(range=(-5, 5), key="formant", resolution=0.05, orientation="h", default_value=data.get("formant", 0.0), enable_events=True, tooltip=i18n("Formant shift in semitones. Changes apparent vocal-tract size / brightness separately from pitch. Positive often sounds thinner or higher; negative thicker or lower.")),
                        ],
                        [
                            sg.Text(i18n("Index Rate"), tooltip=i18n("How strongly retrieval replaces live features with the closest features from this model's .index. 0 is off. Higher values lock timbre closer to the training voice and can reduce leakage.")),
                            sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0), enable_events=True, tooltip=i18n("How strongly retrieval replaces live features with the closest features from this model's .index. 0 is off. Higher values lock timbre closer to the training voice and can reduce leakage.")),
                        ],
                        [
                            sg.Text(i18n("loudness factor"), tooltip=i18n("How much the output follows your microphone volume envelope. 0 matches your speaking loudness; 1 keeps the model's own output level.")),
                            sg.Slider(range=(0.0, 1.0), key="rms_mix_rate", resolution=0.01, orientation="h", default_value=data.get("rms_mix_rate", 0), enable_events=True, tooltip=i18n("How much the output follows your microphone volume envelope. 0 matches your speaking loudness; 1 keeps the model's own output level.")),
                        ],
                        [
                            sg.Text(i18n("pitch detection algorithm"), tooltip=i18n("How pitch (F0) is estimated. RMVPE is the usual choice. PM is faster on CPU but less accurate. FCPE is another neural estimator.")),
                            sg.Radio("pm", "f0method", key="pm", default=data.get("pm", False), enable_events=True, tooltip=i18n("Praat/Parselmouth autocorrelation. Light on CPU, more errors and muted notes.")),
                            sg.Radio("rmvpe", "f0method", key="rmvpe", default=data.get("rmvpe", True), enable_events=True, tooltip=i18n("RMVPE neural pitch tracker. Best quality for speech and singing; recommended default.")),
                            sg.Radio("fcpe", "f0method", key="fcpe", default=data.get("fcpe", False), enable_events=True, tooltip=i18n("FCPE neural pitch tracker. Another GPU-friendly F0 estimator; try it if RMVPE artifacts bother you.")),
                        ],
                    ],
                    title=i18n("General settings"),
                ),
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("Sample length"), tooltip=i18n("Chunk size in seconds processed each callback. Smaller is lower latency but harder on the GPU and more likely to glitch. Larger is smoother with more delay.")),
                            sg.Slider(range=(0.02, 1.5), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 0.25), tooltip=i18n("Chunk size in seconds processed each callback. Smaller is lower latency but harder on the GPU and more likely to glitch. Larger is smoother with more delay.")),
                        ],
                        # [
                        #     sg.Text("Device latency"),
                        #     sg.Slider(range=(0, 1), key="device_latency", resolution=0.001, orientation="h", default_value=data.get("device_latency", 0.1), enable_events=True),
                        # ],
                        [
                            sg.Text(i18n("Fade length"), tooltip=i18n("SOLA crossfade overlap between chunks, in seconds. Longer fades hide seams; they also add delay.")),
                            sg.Slider(range=(0.01, 0.15), key="crossfade_length", resolution=0.01, orientation="h", default_value=data.get("crossfade_length", 0.05), tooltip=i18n("SOLA crossfade overlap between chunks, in seconds. Longer fades hide seams; they also add delay.")),
                        ],
                        [
                            sg.Text(i18n("Extra inference time"), tooltip=i18n("Extra past audio (seconds) given to the model as context before the current chunk. More can improve quality; it uses more compute and adds delay.")),
                            sg.Slider(range=(0.05, 5.00), key="extra_time", resolution=0.01, orientation="h", default_value=data.get("extra_time", 2.5), tooltip=i18n("Extra past audio (seconds) given to the model as context before the current chunk. More can improve quality; it uses more compute and adds delay.")),
                        ],
                        [
                            sg.Checkbox(i18n("Input noise reduction"), key="I_noise_reduce", default=data.get("I_noise_reduce", False), enable_events=True, tooltip=i18n("Spectral gate on the microphone before conversion. Cuts hiss; adds a little delay.")),
                            sg.Checkbox(i18n("Output noise reduction"), key="O_noise_reduce", default=data.get("O_noise_reduce", False), enable_events=True, tooltip=i18n("Spectral gate on the converted voice. Can tame residual hiss after conversion.")),
                        ],
                    ],
                    title=i18n("Performance settings"),
                ),
            ],
            [
                sg.Checkbox(i18n("Run Audio"), key="run_audio", default=False, enable_events=True, tooltip=i18n("When checked, audio from the input device is sent to the output. Uncheck to stop all audio.")),
                sg.Checkbox(i18n("Change Voice"), key="change_voice", default=True, enable_events=True, tooltip=i18n("When checked with Run Audio, convert with the selected model, or apply pitch/formant only if Model is None. When unchecked, the input is forwarded to the output with low latency.")),
                sg.Checkbox(i18n("Debug"), key="debug", default=data.get("debug", False), enable_events=True, tooltip=i18n("When checked, print SOLA offset and per-chunk inference time to the console.")),
                sg.Text(i18n("Algorithmic delays(ms):"), tooltip=i18n("Estimated extra delay from the audio device, chunk size, fade, and input noise reduction. Not the same as inference time.")),
                sg.Text("0", key="delay_time"),
                sg.Text(i18n("Inference time (ms):"), tooltip=i18n("How long the last chunk took to convert. Keep this below Sample length or you will hear dropouts.")),
                sg.Text("0", key="infer_time"),
            ],
        ]
        print("Creating window...")
        self.window = sg.Window("RVC - GUI", layout=layout, finalize=True)
        for key in ("block_time", "crossfade_length", "extra_time"):
            self.window[key].Widget.bind("<ButtonRelease-1>", lambda e, k=key: self.window.write_event_value("-SLIDER_RELEASE-", k))
        print(f"Realtime GUI ready ({time.perf_counter() - _boot_t0:.1f}s)")
        self.event_handler()

    def event_handler(self):
        global flag_vc
        while True:
            event, values = self.window.read(timeout=100)
            if event in (sg.TIMEOUT_EVENT, "__TIMEOUT__"):
                if flag_vc and self.change_voice:
                    self.window["infer_time"].update(self.last_infer_ms)
                continue
            if event == sg.WINDOW_CLOSED:
                self.close_window()
                return
            if event == "-AUDIO_UI-":
                self.apply_audio_ui(values)
                continue
            if event == "-SLIDER_RELEASE-":
                key = values["-SLIDER_RELEASE-"] if isinstance(values, dict) and "-SLIDER_RELEASE-" in values else values
                if key in ("block_time", "crossfade_length", "extra_time"):
                    try:
                        self.last_values[key] = float(self.window[key].Widget.get())
                    except:
                        pass
                    self.request_audio_reload()
                continue
            if values:
                self.last_values = values
            self.handle_event(event, values)

    def close_window(self):
        self.closing = True
        self.needs_reload = False
        try:
            self.persist_model_root(self.window["model_root"].get(), self.window["model_identity"].get())
        except:
            pass
        self.stop_stream()
        exit()

    def model_paths_error(self, values):
        if self.is_none_model(values):
            return None
        model_files = self.scan_model_root(values.get("model_root") or "")
        pth_path, index_path = model_files.get(values.get("model_identity", ""), ("", ""))
        if len(pth_path.strip()) == 0:
            return i18n("Please choose the .pth file")
        if len(index_path.strip()) == 0:
            return i18n("Please choose the .index file")
        pattern = re.compile("[^\x00-\x7F]+")
        if pattern.findall(pth_path):
            return i18n("The .pth file path cannot contain Chinese characters")
        if pattern.findall(index_path):
            return i18n("The index file path cannot contain Chinese characters")
        return None

    def push_audio_ui(self, infer=None):
        payload = {"sr": getattr(self.gui_config, "samplerate", ""), "delay_ms": int(np.round(self.delay_time * 1000))}
        if infer is not None:
            payload["infer"] = infer
        try:
            self.window.write_event_value("-AUDIO_UI-", payload)
        except:
            pass

    def apply_audio_ui(self, values):
        payload = values
        if isinstance(values, dict) and "-AUDIO_UI-" in values and isinstance(values.get("-AUDIO_UI-"), dict):
            payload = values["-AUDIO_UI-"]
        if not isinstance(payload, dict):
            return
        if "sr" in payload:
            self.window["sr_stream"].update(payload["sr"])
        if "delay_ms" in payload:
            self.window["delay_time"].update(payload["delay_ms"])
        if "infer" in payload:
            self.window["infer_time"].update(payload["infer"])

    def request_audio_reload(self, validate=True):
        global flag_vc
        if self.closing:
            return
        running = bool(self.last_values.get("run_audio")) if self.last_values else flag_vc
        if not running and not flag_vc:
            return
        if validate and running and self.last_values.get("change_voice"):
            err = self.model_paths_error(self.last_values)
            if err:
                sg.popup(err)
                return
        self.needs_reload = True
        if not self.reload_lock.acquire(blocking=False):
            return
        try:
            threading.Thread(target=self.reload_worker, name="audio-reload", daemon=True).start()
        except:
            self.reload_lock.release()
            raise

    def reload_worker(self):
        self.reloading = True
        try:
            while not self.closing:
                self.needs_reload = False
                values = dict(self.last_values)
                try:
                    if not values.get("run_audio"):
                        self.stop_stream()
                    else:
                        self.apply_audio_state(values, bool(values.get("run_audio")), bool(values.get("change_voice")))
                except Exception:
                    print(traceback.format_exc())
                if not self.needs_reload or self.closing:
                    break
        finally:
            self.reloading = False
            self.reload_lock.release()
            if self.needs_reload and not self.closing:
                self.request_audio_reload(validate=False)

    def handle_event(self, event, values):
        if values:
            self.last_values = values
        if event == "reload_devices" or event == "sg_hostapi":
            if flag_vc:
                self.stop_stream()
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.update_devices(hostapi_name=values["sg_hostapi"])
            if self.gui_config.sg_hostapi not in self.hostapis:
                self.gui_config.sg_hostapi = self.hostapis[0]
            self.window["sg_hostapi"].Update(values=self.hostapis)
            self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
            if self.gui_config.sg_input_device not in self.input_devices and len(self.input_devices) > 0:
                self.gui_config.sg_input_device = self.input_devices[0]
            self.window["sg_input_device"].Update(values=self.input_devices)
            self.window["sg_input_device"].Update(value=self.gui_config.sg_input_device)
            if self.gui_config.sg_output_device not in self.output_devices:
                self.gui_config.sg_output_device = self.output_devices[0]
            self.window["sg_output_device"].Update(values=self.output_devices)
            self.window["sg_output_device"].Update(value=self.gui_config.sg_output_device)
            out2 = values.get("sg_output_device_2") or NONE_MODEL
            if out2 not in self.output2_choices():
                out2 = NONE_MODEL
            self.gui_config.sg_output_device_2 = out2
            self.window["sg_output_device_2"].Update(values=self.output2_choices())
            self.window["sg_output_device_2"].Update(value=out2)
            self.request_audio_reload()
            return
        if event == "threhold":
            self.gui_config.threhold = values["threhold"]
        elif event == "pitch":
            self.gui_config.pitch = values["pitch"]
            if self.rvc is not None and not self.reloading:
                self.rvc.change_key(values["pitch"])
        elif event == "formant":
            self.gui_config.formant = values["formant"]
            if self.rvc is not None and not self.reloading:
                self.rvc.change_formant(values["formant"])
        elif event == "index_rate":
            self.gui_config.index_rate = values["index_rate"]
            if self.rvc is not None and not self.reloading:
                self.rvc.change_index_rate(values["index_rate"])
        elif event == "rms_mix_rate":
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
        elif event in ["pm", "rmvpe", "fcpe"]:
            self.gui_config.f0method = event
        elif event == "I_noise_reduce":
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            if self.stream is not None:
                self.delay_time += (1 if values["I_noise_reduce"] else -1) * min(values["crossfade_length"], 0.04)
                self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))
        elif event == "O_noise_reduce":
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
        elif event == "run_audio" or event == "change_voice":
            if event == "run_audio" and values["run_audio"]:
                gui_settings.save(gui_settings.from_values(values))
            if values.get("run_audio") and values.get("change_voice"):
                err = self.model_paths_error(values)
                if err:
                    sg.popup(err)
                    self.window[event].update(False)
                    return
            if event == "run_audio" and not values["run_audio"]:
                self.stop_stream()
            self.request_audio_reload()
        elif event == "model_root" or event == "select_model_root":
            candidates = [values.get("select_model_root") if event == "select_model_root" else None, values.get("model_root")]
            try:
                candidates.append(self.window["model_root"].get())
            except:
                pass
            root = next((c for c in candidates if c and os.path.isdir(c)), "")
            if not root:
                root = next((c for c in candidates if c), "") or ""
            if root:
                self.window["model_root"].update(root)
            identities = self.model_identity_list(root)
            selected = values.get("model_identity", "")
            if selected not in identities:
                selected = NONE_MODEL
            self.window["model_identity"].Update(values=identities)
            self.window["model_identity"].Update(value=selected)
            self.persist_model_root(root, selected)
            self.request_audio_reload()
        elif event == "model_identity":
            self.persist_model_root(values.get("model_root") or "", values.get("model_identity") or "")
            self.request_audio_reload()
        elif event == "debug":
            self.gui_config.debug = values["debug"]
            gui_settings.update(debug=values["debug"])
        elif event == "preset":
            self.apply_preset(values.get("preset") or "", values)
        elif event == "preset_create":
            name = self.prompt_new_preset_name()
            if name is None:
                return
            if not name:
                sg.popup(i18n("Please enter a preset name"))
                return
            presets = gui_settings.read_presets()
            if name in presets:
                sg.popup(i18n("A preset with that name already exists"))
                return
            presets[name] = gui_settings.snapshot(values)
            gui_settings.update(presets=presets, preset=name)
            self.refresh_preset_combo(name)
            if self.last_values:
                self.last_values["preset"] = name
        elif event == "preset_update":
            name = values.get("preset") or ""
            if not name or name not in gui_settings.read_presets():
                sg.popup(i18n("Select a preset"))
                return
            if sg.popup_yes_no(i18n("Are you sure?"), title=i18n("Update")) != "Yes":
                return
            presets = gui_settings.read_presets()
            presets[name] = gui_settings.snapshot(values)
            gui_settings.update(presets=presets, preset=name)
        elif event == "preset_delete":
            name = values.get("preset") or ""
            if not name or name not in gui_settings.read_presets():
                sg.popup(i18n("Select a preset"))
                return
            if sg.popup_yes_no(i18n("Are you sure?"), title=i18n("Delete")) != "Yes":
                return
            presets = gui_settings.read_presets()
            presets.pop(name, None)
            gui_settings.update(presets=presets, preset="")
            self.refresh_preset_combo("")
            if self.last_values:
                self.last_values["preset"] = ""
        else:
            self.request_audio_reload()

    def apply_audio_state(self, values, run, convert):
        global flag_vc
        if not run:
            self.stop_stream()
            return True
        wanted_sr = "sr_model" if values["sr_model"] else "sr_device"
        out2 = values.get("sg_output_device_2") or NONE_MODEL
        out2_dirty = flag_vc and out2 != (self.gui_config.sg_output_device_2 or NONE_MODEL)
        stream_dirty = flag_vc and (self.low_latency_stream == convert or self.gui_config.block_time != values["block_time"] or self.gui_config.sr_type != wanted_sr or self.gui_config.sg_input_device != values["sg_input_device"] or self.gui_config.sg_output_device != values["sg_output_device"] or self.gui_config.sg_wasapi_exclusive != values["sg_wasapi_exclusive"] or self.gui_config.sg_hostapi != values["sg_hostapi"])
        if convert:
            none = self.is_none_model(values)
            model_files = self.scan_model_root(values["model_root"])
            pth_path, index_path = model_files.get(values.get("model_identity", ""), ("", ""))
            buf_ok = hasattr(self, "input_wav") and self.input_wav.shape[0] == self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame and self.gui_config.block_time == values["block_time"] and self.gui_config.crossfade_time == values["crossfade_length"] and self.gui_config.sr_type == wanted_sr
            ready = (buf_ok and self.rvc is None) if none else (buf_ok and self.rvc is not None and self.rvc.pth_path == pth_path and self.rvc.index_path == index_path and self.gui_config.extra_time == values["extra_time"])
            if not ready:
                if not self.set_values(values, require_model=not none):
                    return False
                if flag_vc:
                    self.stop_stream()
                print(i18n("CUDA available: %s") % torch.cuda.is_available())
                self.start_vc(reuse_stream=False)
                if self.rvc is not None and self.last_values:
                    self.rvc.change_key(self.last_values["pitch"])
                    self.rvc.change_formant(self.last_values["formant"])
                    self.rvc.change_index_rate(self.last_values["index_rate"])
            elif stream_dirty:
                if not self.set_values(values, require_model=False):
                    return False
                if flag_vc:
                    self.stop_stream()
            self.change_voice = True
            if self.closing or not self.last_values.get("run_audio"):
                self.stop_stream()
                return True
            if not flag_vc:
                if ready and not self.set_values(values, require_model=False):
                    return False
                self.start_stream()
        else:
            self.change_voice = False
            restart = (not flag_vc) or (not self.low_latency_stream) or stream_dirty
            if restart:
                if flag_vc:
                    self.stop_stream()
                if self.closing or not self.last_values.get("run_audio"):
                    return True
                if not self.set_values(values, require_model=False):
                    return False
                self.prepare_stream_params()
                self.start_stream(low_latency=True)
            self.push_audio_ui(infer=0)
        if self.stream is not None:
            if self.change_voice:
                self.delay_time = self.stream.latency[-1] + values["block_time"] + values["crossfade_length"] + 0.01
                if values["I_noise_reduce"]:
                    self.delay_time += min(values["crossfade_length"], 0.04)
            else:
                self.delay_time = float(self.stream.latency[0]) + float(self.stream.latency[-1])
            self.push_audio_ui()
        if self.stream is not None and out2_dirty and not stream_dirty:
            self.gui_config.sg_output_device_2 = out2
            extra_settings = sd.WasapiSettings(exclusive=True) if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive else None
            self.start_output2(extra_settings)
        return True

    def prepare_stream_params(self):
        self.gui_config.samplerate = self.get_device_samplerate()
        self.gui_config.channels = self.get_device_channels()
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = max(64, self.gui_config.samplerate // 200)

    def set_values(self, values, require_model=True):
        model_files = self.scan_model_root(values["model_root"])
        pth_path, index_path = model_files.get(values.get("model_identity", ""), ("", ""))
        if require_model and self.model_paths_error(values):
            return False
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        # self.device_latency = values["device_latency"]
        self.gui_config.sg_hostapi = values["sg_hostapi"]
        self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
        self.gui_config.sg_input_device = values["sg_input_device"]
        self.gui_config.sg_output_device = values["sg_output_device"]
        self.gui_config.sg_output_device_2 = values.get("sg_output_device_2") or NONE_MODEL
        self.gui_config.pth_path = pth_path
        self.gui_config.index_path = index_path
        self.gui_config.sr_type = ["sr_model", "sr_device"][[values["sr_model"], values["sr_device"]].index(True)]
        self.gui_config.threhold = values["threhold"]
        self.gui_config.pitch = values["pitch"]
        self.gui_config.formant = values["formant"]
        self.gui_config.block_time = values["block_time"]
        self.gui_config.crossfade_time = values["crossfade_length"]
        self.gui_config.extra_time = values["extra_time"]
        self.gui_config.I_noise_reduce = values["I_noise_reduce"]
        self.gui_config.O_noise_reduce = values["O_noise_reduce"]
        self.gui_config.rms_mix_rate = values["rms_mix_rate"]
        self.gui_config.index_rate = values["index_rate"]
        self.gui_config.f0method = ["pm", "rmvpe", "fcpe"][[values["pm"], values["rmvpe"], values["fcpe"]].index(True)]
        self.gui_config.debug = values["debug"]
        return True

    def start_vc(self, reuse_stream=False):
        none = self.is_none_model()
        print("Starting pitch/formant (no voice model)..." if none else "Starting voice conversion (loading model)...")
        _start_t0 = time.perf_counter()
        if none:
            self.rvc = None
        torch.cuda.empty_cache()
        if not none:
            self.rvc = rvc_for_realtime.RVC(self.gui_config.pitch, self.gui_config.formant, self.gui_config.pth_path, self.gui_config.index_path, self.gui_config.index_rate, self.config, self.rvc)
        if not reuse_stream:
            self.gui_config.samplerate = self.get_device_samplerate() if none or self.gui_config.sr_type != "sr_model" else self.rvc.tgt_sr
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
        else:
            self.zc = self.gui_config.samplerate // 100
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = 0 if none else int(np.round(self.gui_config.extra_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.input_wav = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame, device=self.config.device, dtype=torch.float32)
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * self.input_wav.shape[0] // self.zc, device=self.config.device, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        if none:
            self.resampler = None
            self.resampler2 = None
        else:
            self.resampler = tat.Resample(orig_freq=self.gui_config.samplerate, new_freq=16000, dtype=torch.float32).to(self.config.device)
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.gui_config.samplerate, dtype=torch.float32).to(self.config.device)
            else:
                self.resampler2 = None
        # Bundled torch.istft is not CUDA Graph-capturable, so TorchGate
        # stays eager while resampling and RVC inference still use graphs.
        self.tg = TorchGate(sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.config.device)
        if not none:
            self.prewarm_cuda_graph()
        print(f"{'Pitch/formant' if none else 'Voice conversion'} started in {time.perf_counter() - _start_t0:.1f}s")

    def prewarm_cuda_graph(self):
        if not cuda_graph_enabled(self.config.device):
            return
        try:
            print(i18n("Warming up CUDA Graph"))
            samples = self.input_wav_res.shape[0]
            phase = torch.arange(samples, device=self.config.device, dtype=torch.float32)
            probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
            self.input_wav_res.copy_(probe)

            if self.gui_config.I_noise_reduce:
                short = self.input_wav[-self.sola_buffer_frame - self.block_frame :].unsqueeze(0)
                self.tg(short, self.input_wav.unsqueeze(0))

            resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
            run_cuda_graph(self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input)

            inferred = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.gui_config.f0method)
            if self.resampler2 is not None:
                inferred = run_cuda_graph(self.resampler2, "realtime-output-resample", lambda audio: self.resampler2(audio), inferred)
            if self.gui_config.O_noise_reduce:
                self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
            torch.cuda.synchronize(self.config.device)
            print(i18n("CUDA Graph warm-up complete"))
        except Exception:
            print(traceback.format_exc())
        finally:
            self.input_wav.zero_()
            self.input_wav_denoise.zero_()
            self.input_wav_res.zero_()
            self.output_buffer.zero_()
            self.sola_buffer.zero_()
            self.nr_buffer.zero_()
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()

    def start_stream(self, low_latency=False):
        global flag_vc
        with self.stream_lock:
            if not flag_vc:
                flag_vc = True
                self.low_latency_stream = low_latency
                if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive:
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                self.stream = sd.Stream(callback=self.audio_callback, blocksize=self.block_frame, samplerate=self.gui_config.samplerate, channels=self.gui_config.channels, dtype="float32", extra_settings=extra_settings, latency="low")
                self.stream.start()
                self.start_output2(extra_settings)

    def start_output2(self, extra_settings):
        self.stop_output2()
        name = self.gui_config.sg_output_device_2
        if not name or name == NONE_MODEL or name == self.gui_config.sg_output_device:
            return
        try:
            idx = self.output_devices_indices[self.output_devices.index(name)]
        except (ValueError, IndexError):
            return
        try:
            ch = min(int(sd.query_devices(device=idx)["max_output_channels"]), max(1, self.gui_config.channels), 2)
            self.monitor_q = queue.Queue(maxsize=8)
            self.stream2 = sd.OutputStream(device=idx, callback=self.output2_callback, blocksize=self.block_frame, samplerate=self.gui_config.samplerate, channels=ch, dtype="float32", extra_settings=extra_settings, latency="low")
            self.stream2.start()
            print(i18n("Output device 2: %s:%s") % (str(idx), name))
        except Exception:
            print(traceback.format_exc())
            self.stop_output2()

    def stop_output2(self):
        self.monitor_q = None
        if self.stream2 is not None:
            try:
                self.stream2.abort()
                self.stream2.close()
            except:
                pass
            self.stream2 = None

    def stop_stream(self):
        global flag_vc
        with self.stream_lock:
            flag_vc = False
            self.stop_output2()
            if self.stream is not None:
                try:
                    self.stream.abort()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            self.change_voice = False

    def shift_pitch_formant(self, wav):
        wav = wav.clone()
        pitch = float(self.gui_config.pitch)
        formant = float(self.gui_config.formant)
        if abs(pitch) < 1e-6 and abs(formant) < 1e-6:
            return wav
        n_fft = int(4 * self.zc)
        hop = int(self.zc)
        key = (n_fft, hop, wav.device)
        if getattr(self, "_pf_key", None) != key:
            self._pf_key = key
            self._pf_window = torch.hann_window(n_fft, device=wav.device)
            self._pf_phase_advance = torch.linspace(0, np.pi * hop, n_fft // 2 + 1, device=wav.device).unsqueeze(-1)
        window = self._pf_window
        n = wav.shape[0]
        y = wav
        if abs(pitch) > 1e-4:
            stretch = 1.0 / (2.0 ** (pitch / 12.0))
            spec = torch.stft(y, n_fft, hop, window=window, return_complex=True)
            stretched = AF.phase_vocoder(spec.unsqueeze(0), stretch, self._pf_phase_advance)
            new_len = max(n_fft, int(round(n / stretch)))
            y = torch.istft(stretched.squeeze(0), n_fft, hop, window=window, length=new_len)
            y = F.interpolate(y.view(1, 1, -1), size=n, mode="linear", align_corners=False).view(-1)
        env_shift = formant - pitch
        if abs(env_shift) > 1e-4:
            ratio = 2.0 ** (env_shift / 12.0)
            spec = torch.stft(y, n_fft, hop, window=window, return_complex=True)
            mag = spec.abs().clamp_min(1e-8)
            n_freq = mag.shape[0]
            log_mag = torch.log(mag)
            smooth_n = max(5, int(round(250.0 * n_fft / float(self.gui_config.samplerate))) | 1)
            kernel = torch.hann_window(smooth_n, device=y.device)
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            log_env = F.conv1d(log_mag.transpose(0, 1).unsqueeze(1), kernel, padding=smooth_n // 2).squeeze(1).transpose(0, 1)
            idx = (torch.arange(n_freq, device=y.device, dtype=torch.float32) / ratio).clamp(0, n_freq - 1)
            i0 = idx.long()
            i1 = torch.clamp(i0 + 1, max=n_freq - 1)
            frac = (idx - i0.float()).unsqueeze(1)
            log_env_w = log_env[i0] * (1 - frac) + log_env[i1] * frac
            mag_w = torch.exp(log_env_w + (log_mag - log_env))
            y = torch.istft(torch.polar(mag_w, spec.angle()), n_fft, hop, window=window, length=n)
        return y

    def audio_callback(self, indata, outdata, frames, times, status):
        """
        Audio callback
        """
        global flag_vc
        if not flag_vc:
            outdata[:] = 0
            return
        if not self.change_voice:
            if indata.shape == outdata.shape:
                outdata[:] = indata
            else:
                mono = librosa.to_mono(indata.T)
                outdata[:] = np.repeat(mono.reshape(-1, 1), outdata.shape[1], axis=1)
            self.push_output2(outdata)
            return
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.gui_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        if self.gui_config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
        if self.rvc is None:
            src = self.input_wav_denoise if self.gui_config.I_noise_reduce else self.input_wav
            infer_wav = self.shift_pitch_formant(src[self.extra_frame :])
        else:
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
            if self.gui_config.I_noise_reduce:
                resample_input = self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input)[160:]
            else:
                resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = run_cuda_graph(self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input)[160:]
            infer_wav = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.gui_config.f0method)
            if self.resampler2 is not None:
                infer_wav = run_cuda_graph(self.resampler2, "realtime-output-resample", lambda audio: self.resampler2(audio), infer_wav)
        # output noise reduction
        if self.gui_config.O_noise_reduce:
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1:
            if self.gui_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(y=input_wav[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - self.gui_config.rms_mix_rate)
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        if self.gui_config.debug:
            print(i18n("SOLA offset: %d") % int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        outdata[:] = infer_wav[: self.block_frame].repeat(self.gui_config.channels, 1).t().cpu().numpy()
        self.push_output2(outdata)
        self.last_infer_ms = int((time.perf_counter() - start_time) * 1000)
        if self.gui_config.debug:
            print(i18n("Inference time: %.2f seconds") % (self.last_infer_ms / 1000.0))

    def push_output2(self, outdata):
        q = self.monitor_q
        if q is None:
            return
        try:
            q.put_nowait(np.copy(outdata))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(np.copy(outdata))
            except queue.Full:
                pass

    def output2_callback(self, outdata, frames, times, status):
        q = self.monitor_q
        if q is None or not flag_vc:
            outdata[:] = 0
            return
        try:
            data = q.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            return
        src = np.asarray(data, dtype=np.float32)
        if src.ndim == 1:
            src = src.reshape(-1, 1)
        if src.shape[0] != frames:
            outdata[:] = 0
            return
        if src.shape[1] < outdata.shape[1]:
            outdata[:] = np.repeat(src[:, :1], outdata.shape[1], axis=1)
        elif src.shape[1] > outdata.shape[1]:
            outdata[:] = src[:, : outdata.shape[1]]
        else:
            outdata[:] = src

    def update_devices(self, hostapi_name=None):
        """List audio devices"""
        global flag_vc
        print("Enumerating audio devices...")
        with self.stream_lock:
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
        self.output_devices = [d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]
        self.input_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
        self.output_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]
        print(f"Audio devices: {len(self.hostapis)} host APIs, {len(self.input_devices)} inputs, {len(self.output_devices)} outputs")

    def set_devices(self, input_device, output_device):
        """Set input and output devices"""
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        print(i18n("Input device: %s:%s") % (str(sd.default.device[0]), input_device))
        print(i18n("Output device: %s:%s") % (str(sd.default.device[1]), output_device))

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)

if __name__ == "__main__":
    gui = GUI()
