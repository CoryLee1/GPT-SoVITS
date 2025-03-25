import runpod
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

cut_method_names = get_cut_method_names()
# 初始化推理模型
config_path = f"{os.getcwd()}/GPT_SoVITS/configs/tts_infer.yaml"
tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)


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
        check_params(req)
        req["return_fragment"] = False
        req["streaming_mode"] = False
        sr, audio_data = next(tts_pipeline.run(req))
        audio_bytes = audio_data.tobytes()
        return {
            "sample_rate": sr,
            "audio_bytes": audio_bytes.hex()  # RunPod 不支持 byte 数据，转 hex
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
