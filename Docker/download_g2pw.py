import os
import zipfile

import requests


def download_and_decompress(model_dir: str = 'G2PWModel/'):
    if not os.path.exists(model_dir):
        parent_directory = os.path.dirname(model_dir)
        zip_dir = os.path.join(parent_directory, "G2PWModel_1.1.zip")
        extract_dir = os.path.join(parent_directory, "G2PWModel_1.1")
        extract_dir_new = os.path.join(parent_directory, "G2PWModel")
        print("Downloading g2pw model...")
        modelscope_url = "https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/G2PWModel_1.1.zip"
        with requests.get(modelscope_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_dir, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Extracting g2pw model...")
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(parent_directory)

        os.rename(extract_dir, extract_dir_new)

    return model_dir


if __name__ == "__main__":
    download_and_decompress("GPT_SoVITS/text/G2PWModel/")