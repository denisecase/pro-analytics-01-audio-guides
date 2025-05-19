# Build Transcripts

> Tested with Python 3.10 and torch CPU option in WSL. 

- DISK SPACE: WhisperX with diarization will create large temporary files. Ensure you have at least 10GB free.
- RAM: Diarization and alignment are memory intensive. Monitor with: `free -h`

## IMPORTANT: If Windows, Work in WSL

Open PowerShell and run `wsl`.

## Clone Project Repo and Open in Code

1. Change directory into ~/Repos.
2. Git clone into ~/Repos folder.
3. Change directory into new pro-analytics-01-audio-guides repo folder.
4. Start VS Code in the repo folder.

```shell
cd ~/Repos
git clone https://github.com/denisecase/pro-analytics-01-audio-guides
cd pro-analytics-01-audio-guides
code .
```

## Set Up WSL Environment

From VS Code menu, select "Terminal", "New Terminal" and run the following commands.

```shell
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install python3.10 python3.10-venv -y
sudo apt-get install ffmpeg -y
ffmpeg -version
```

## Manage Python Virtual Environment: Repeatable Setup

Repeatable as it may take several tries to get consistent versions installed. 

1. Deactivate the virtual environment if it's currently active.
2. Remove `.venv` directory if it exists. 
3. Create a new virtual environment using Python 3.10.
4. Activate the new virtual environment.
5. Install and upgrade key packages.
6. Install packages from requirements.txt.
7. Run pip check.
8. Save installed versions to req-installed.txt.


```shell
deactivate
rm -rf .venv

python3.10 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt --timeout 300 --progress-bar on --no-cache-dir

python3 -m pip check
python3 -m pip list > req-installed.txt
```

## Run The Python Script

It's recommended to restart VS Code after significant environment changes to ensure everything loads correctly. 
Close VS Code and reopen it from the main repo directory by running `code .`

If `.venv` is active, just run the script. 

```shell
python3 build.py
```

If not or to be safe, re-activate `.venv` and run the script.

```shell
deactivate
source .venv/bin/activate
python3 build.py
```

## Package List and Approximate Sizes

| Package          |  Approx. Size | Description                                                  |
|------------------|---------------|--------------------------------------------------------------|
| `torch`          |  ~3.5 GB      | PyTorch with CUDA support for GPU acceleration               |
| `torchvision`    |  ~300-800 MB  | Datasets, model architectures, and image processing for PyTorch |
| `torchaudio`     |  ~100-250 MB  | Audio loading, transformation, and I/O for PyTorch           |
| `whisperx`       | ~200 MB       | Enhanced version of OpenAI's Whisper for transcription       |
| `pyannote.audio` |  ~1.2 GB      | Speaker Diarization and Audio Processing                     |
| `transformers`   |  ~700 MB      | State-of-the-art Natural Language Processing (NLP) library   |

**Total Estimated Disk Space:** 
- **CPU Setup:** ~4.56 GB  
- **GPU Setup:** ~6.66 GB  

If `hf_xet` is not installed: Model downloads may be slower, but the setup will still work.  
These are approximate sizes and may vary.
