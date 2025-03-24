import runpod
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from api_v2 import tts_handle, check_params

# 初始化推理模型
config_path = "GPT_SoVITS/configs/tts_infer.yaml"
tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)

# handler for runpod
def handler(job):
    req = job["input"]

    # 参数检查（不通过直接返回错误）
    err = check_params(req)
    if err is not None:
        return {"error": err.body}

    try:
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