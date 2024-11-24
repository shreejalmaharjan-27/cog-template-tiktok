from typing import Optional, Any
import os
import time
import subprocess
import torch
import numpy as np
import ffmpeg
import hashlib
import json
from pathlib import Path

from cog import BasePredictor, Input, Path, BaseModel, emit_metric
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import format_timestamp

MODEL_CACHE = "weights"
BASE_URL = f"https://weights.replicate.delivery/default/whisper-v3/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

def run_bun_install():
    try:
        lockfile = Path("bun.lockb")
        if not lockfile.exists():
            # Use which to find bun executable
            bun_path = "/root/.bun/bin/bun"
            
            if not bun_path:
                raise FileNotFoundError("bun executable not found")
                
            # Run bun install with proper error handling
            subprocess.run([bun_path, "install"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running bun install: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
class Predictor(BasePredictor):
    def setup(self):
        """Load the large-v3 model"""
        self.model_cache = MODEL_CACHE
        self.models = {}
        self.current_model = "large-v3"
        self.load_model("large-v3")
        run_bun_install()

    def load_model(self, model_name):
        if model_name not in self.models:
            if not os.path.exists(self.model_cache):
                os.makedirs(self.model_cache)

            model_file = f"{model_name}.pt"
            url = BASE_URL + model_file
            dest_path = os.path.join(self.model_cache, model_file)

            if not os.path.exists(dest_path):
                download_weights(url, dest_path)

            with open(dest_path, "rb") as fp:
                checkpoint = torch.load(fp, map_location="cpu")
                dims = ModelDimensions(**checkpoint["dims"])
                model = Whisper(dims)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to("cuda")

            self.models[model_name] = model
        self.current_model = model_name
        return self.models[model_name]

    def predict(
        self,
        video: Path = Input(description="Video Path"),
        caption_size: int =Input (
            default=30,
            description="The maximum number of words to generate in each window",
        ),
        model: str = Input(
            choices=[
                "large-v3",
            ],
            default="large-v3",
            description="Whisper model size (currently only large-v3 is supported).",
        ),
        language: str = Input(
            choices=["auto"]
            + sorted(LANGUAGES.keys())
            + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
            default="auto",
            description="Language spoken in the audio, specify 'auto' for automatic language detection",
        ),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        patience: float = Input(
            default=None,
            description="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        condition_on_previous_text: bool = Input(
            default=True,
            description="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        ),
        temperature_increment_on_fallback: float = Input(
            default=0.2,
            description="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        ),
        compression_ratio_threshold: float = Input(
            default=2.4,
            description="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        ),
        logprob_threshold: float = Input(
            default=-1.0,
            description="if the average log probability is lower than this value, treat the decoding as failed",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Transcribe with {model} model.")
        duration = get_audio_duration(video)
        print(f"Audio duration: {duration} sec")

        # file identifier
        hash = hashlib.md5(str(video).encode('utf-8')).hexdigest()
        print(f"Hash: {hash}")

        if model != self.current_model:
            self.model = self.load_model(model)
        else:
            self.model = self.models[self.current_model]

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        normalized_language = language.lower() if language.lower() != "auto" else None
        if normalized_language and normalized_language not in LANGUAGES:
            normalized_language = TO_LANGUAGE_CODE.get(normalized_language, normalized_language)

        args = {
            "language": normalized_language,
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "fp16": True,
            "word_timestamps": True
        }


        print("Running inference...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.inference_mode():
            result = self.model.transcribe(str(video), temperature=temperature, **args)


        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Inference completed in {elapsed_time:.2f} ms")

        detected_language_code = result["language"]
        detected_language_name = LANGUAGES.get(detected_language_code, detected_language_code)

        emit_metric("audio_duration", duration)
        print(f"Detected language: {detected_language_name}")

        formatted_results = format_whisper_results(result)

        # save the results to a json file on the /src/public directory
        output_file = f"/src/public/{hash}.json"
        with open(output_file, "w") as f:
            json.dump(formatted_results, f, indent=4)

        print(f"Results saved to {output_file}")



        # copy the video file to the /src/public directory
        video_file = f"/src/public/{hash}.mp4"
        os.system(f"cp {str(video)} {video_file}")

        # run bun rendering command to render the video with the subtitles
        props = {
            "video": hash + ".mp4",
            "caption_size": caption_size,
        }

        render_command = f"/root/.bun/bin/bunx remotion render --props='{json.dumps(props)}' CaptionedVideo out/{hash}_captioned.mp4"
        print(f"Running render command: {render_command}")
        subprocess.run(['bash', '-c', render_command], check=True)

        # cleanup
        os.remove(video_file)
        os.remove(output_file)

        return Path(f"/src/out/{hash}_captioned.mp4")
        
    

def format_whisper_results(whisper_result):
    formatted_results = []
    
    for segment in whisper_result['segments']:
        for word in segment['words']:
            formatted_word = {
                "text": word['word'], 
                "startMs": int(word['start'] * 1000),
                "endMs": int(word['end'] * 1000),
                "timestampMs": int((word['end'] * 1000)), 
                "confidence": round(word['probability'], 6)
            }
            
            formatted_results.append(formatted_word)
    
    return formatted_results

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream and 'duration' in audio_stream:
            duration = float(audio_stream['duration'])
            return np.round(duration)
        else:
            print("No audio stream found, cannot calculate duration")
            return -1
    except ffmpeg.Error as e:
        print(f"Error reading audio file: {e.stderr}")
        return -1
    