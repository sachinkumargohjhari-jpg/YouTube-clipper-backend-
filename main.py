"""AI YouTube Clipper backend (FastAPI).

Pipeline per job:
  yt-dlp (download / record live) -> faster-whisper (word timestamps)
  -> highlight picking + viral score -> OpenCV face tracking -> 9:16 crop
  -> ASS captions burned in with FFmpeg -> MP4 ready to download.
"""
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import cv2
import yt_dlp
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ---------------------------------------------------------------- config
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/clipper"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
LIVE_SECONDS = int(os.getenv("LIVE_RECORD_SECONDS", "600"))  # live: record last N sec
MAX_SOURCE_MIN = int(os.getenv("MAX_SOURCE_MINUTES", "120"))
KEEP_HOURS = int(os.getenv("KEEP_HOURS", "6"))
ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
COOKIES = os.getenv("YT_COOKIES")  # path to cookies.txt (needed on most cloud IPs)
ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

app = FastAPI(title="AI Clipper API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
GPU_LOCK = threading.Semaphore(int(os.getenv("MAX_PARALLEL_JOBS", "1")))
_model: Optional[WhisperModel] = None

