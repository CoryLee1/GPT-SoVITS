import base64
import subprocess
from io import BytesIO

import numpy as np
import soundfile as sf
import runpod
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

cut_method_names = get_cut_method_names()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化推理模型
config_path = f"{os.getcwd()}/GPT_SoVITS/configs/tts_infer.yaml"
tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    print({
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "streaming_mode": streaming_mode,
        "media_type": media_type,
        "prompt_lang": prompt_lang,
        "text_split_method": text
    })

    if ref_audio_path in [None, ""]:
        raise Exception("ref_audio_path is required")

    if text in [None, ""]:
        raise Exception("text is required")
    if (text_lang in [None, ""]):
        raise Exception("text_lang is required")
    elif text_lang.lower() not in tts_config.languages:
        raise Exception(
            f"text_lang: {text_lang} is not supported in version {tts_config.version}"
        )
    if (prompt_lang in [None, ""]):
        raise Exception("prompt_lang is required")
    elif prompt_lang.lower() not in tts_config.languages:
        raise Exception(f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}")
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        raise Exception(f"media_type: {media_type} is not supported")
    elif media_type == "ogg" and not streaming_mode:
        raise Exception("ogg format is not supported in non-streaming mode")

    if text_split_method not in cut_method_names:
        raise Exception(f"text_split_method:{text_split_method} is not supported")

    return None


# handler for runpod
def handler(job):
    req = job["input"]

    # 参数检查（不通过直接返回错误）
    try:
        global tts_pipeline
        del tts_pipeline  # 显式释放资源
        tts_pipeline = TTS(tts_config)  # 重新加载

        check_params(req)
        req["return_fragment"] = False
        req["streaming_mode"] = False
        media_type = req.get("media_type", "wav")
        sr, audio_data = next(tts_pipeline.run(req))
        audio_bytes = pack_audio(
            BytesIO(), audio_data, sr, media_type
        ).getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "media_type": media_type,
            "sample_rate": sr,
            "audio_data": audio_base64,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
