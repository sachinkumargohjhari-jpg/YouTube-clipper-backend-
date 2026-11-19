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


def whisper() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    return _model


# ---------------------------------------------------------------- schemas
class LinkIn(BaseModel):
    url: str


class ProcessIn(LinkIn):
    resolution: Literal[720, 1080, 1440, 2160] = 1080
    clip_length: int = 60  # seconds; 0 = auto (45s)
    max_clips: int = 6
    language: Optional[str] = None  # e.g. "hi", "en"; None = auto-detect
    caption_style: Literal["bold", "karaoke", "minimal"] = "bold"
    captions: bool = True
    track_face: bool = True


def check_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise HTTPException(400, "Sirf YouTube links allowed hain")
    return url


def ydl_opts(**extra) -> dict:
    o = {"quiet": True, "noplaylist": True, "no_warnings": True, **extra}
    if COOKIES:
        o["cookiefile"] = COOKIES
    return o


# ---------------------------------------------------------------- download
def probe(url: str) -> dict:
    with yt_dlp.YoutubeDL(ydl_opts(skip_download=True)) as y:
        i = y.extract_info(url, download=False)
    heights = sorted({f["height"] for f in i.get("formats", []) if f.get("height")})
    return {
        "title": i.get("title"),
        "duration": i.get("duration"),
        "is_live": bool(i.get("is_live")),
        "thumbnail": i.get("thumbnail"),
        "channel": i.get("channel"),
        "available_heights": heights,
    }


def download_source(url: str, res: int, live: bool, d: Path) -> Path:
    if live:
        out = d / "source.mp4"
        with yt_dlp.YoutubeDL(ydl_opts(skip_download=True, format=f"best[height<={res}]/best")) as y:
            stream = y.extract_info(url, download=False)["url"]
        subprocess.run(
            ["ffmpeg", "-y", "-i", stream, "-t", str(LIVE_SECONDS), "-c", "copy", str(out)],
            check=True, capture_output=True,
        )
        return out
    opts = ydl_opts(
        format=f"bestvideo[height<={res}]+bestaudio/best[height<={res}]",
        merge_output_format="mkv",
        outtmpl=str(d / "source.%(ext)s"),
    )
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    return next(d.glob("source.*"))


# ---------------------------------------------------------------- AI: transcript + highlights
def transcribe(path: Path, lang: Optional[str]):
    segs, info = whisper().transcribe(str(path), language=lang, word_timestamps=True, vad_filter=True)
    sents, words = [], []
    for s in segs:
        sents.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    return sents, words, info.language


HOOK = re.compile(r"\b(secret|mistake|never|always|why|how|biggest|nobody|truth|stop|free|\d+)\b", re.I)


def viral_score(text: str, dur: float) -> int:
    wps = len(text.split()) / max(dur, 1)
    s = 40 + min(wps, 3.5) * 8 + 5 * text.count("?") + 2 * text.count("!") + 3 * len(HOOK.findall(text))
    return min(99, int(s))


def pick_clips(sents: list, length: int, n: int) -> list:
    target = length or 45
    cands = []
    for i, s in enumerate(sents):
        j, txt = i, ""
        while j < len(sents) and sents[j]["end"] - s["start"] < target:
            txt += " " + sents[j]["text"]
            j += 1
        if j >= len(sents) or sents[j - 1]["end"] - s["start"] < 10:
            continue
        end = sents[j - 1]["end"]
        cands.append({"start": s["start"], "end": end, "score": viral_score(txt, end - s["start"]),
                      "title": " ".join(s["text"].split()[:8]).strip(" ,.") or "Clip"})
    cands.sort(key=lambda c: -c["score"])
    chosen = []
    for c in cands:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------- AI: smart framing 16:9 -> 9:16
def render_vertical(src: Path, start: float, end: float, dst: Path, track: bool):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw = (int(H * 9 / 16) // 2) * 2
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    x = target = (W - cw) / 2
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{H}", "-r", str(fps), "-i", "-",
         "-ss", str(start), "-t", str(end - start), "-i", str(src),
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(dst)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    step = max(int(fps // 3), 1)
    for i in range(int((end - start) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if track and i % step == 0:  # detect ~3x per second, then smooth
            small = cv2.resize(f, (640, int(640 * H / W)))
            hits = face.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40, 40))
            if len(hits):
                fx, _, fw, _ = max(hits, key=lambda r: r[2] * r[3])
                target = (fx + fw / 2) * W / 640 - cw / 2
        x += (target - x) * 0.08
        xi = int(min(max(x, 0), W - cw))
        ff.stdin.write(f[:, xi:xi + cw].tobytes())
    ff.stdin.close()
    ff.wait()
    cap.release()
    return cw, H


# ---------------------------------------------------------------- captions (ASS, burned in)
def ts(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_ass(words: list, start: float, end: float, style: str, path: Path, w: int, h: int):
    fs = {"bold": 0.055, "karaoke": 0.05, "minimal": 0.035}[style]
    prim, sec = ("&H0000D7FF", "&H00FFFFFF") if style == "karaoke" else ("&H00FFFFFF", "&H0000D7FF")
    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: D,Noto Sans,{int(h * fs)},{prim},{sec},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(2, int(h * 0.004))},2,2,40,40,{int(h * 0.18)},1\n\n[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    ws = [x for x in words if x["start"] >= start and x["end"] <= end + 0.3]
    n = 6 if style == "minimal" else 3
    lines = []
    for i in range(0, len(ws), n):
        ch = ws[i:i + n]
        a, b = ch[0]["start"] - start, ch[-1]["end"] - start
        if style == "karaoke":
            txt = " ".join(f"{{\\k{max(int((x['end'] - x['start']) * 100), 1)}}}{x['word']}" for x in ch)
        else:
            txt = " ".join(x["word"] for x in ch)
            txt = txt.upper() if style == "bold" else txt
        lines.append(f"Dialogue: 0,{ts(a)},{ts(b)},D,,0,0,0,,{txt}")
    path.write_text(head + "\n".join(lines), encoding="utf-8")


def burn_captions(raw: Path, ass: Path, out: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw.name, "-vf", f"ass={ass.name}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-c:a", "copy", out.name],
        check=True, capture_output=True, cwd=raw.parent,
    )


# ---------------------------------------------------------------- job runner
def upd(jid: str, **kw):
    JOBS[jid].update(kw)


def cleanup_old():
    for d in DATA_DIR.iterdir():
        if d.is_dir() and time.time() - d.stat().st_mtime > KEEP_HOURS * 3600:
            shutil.rmtree(d, ignore_errors=True)
            JOBS.pop(d.name, None)


def run_job(jid: str, req: ProcessIn, meta: dict):
    d = DATA_DIR / jid
    try:
        with GPU_LOCK:
            upd(jid, status="running", stage="Stream fetch ho rahi hai", progress=5)
            src = download_source(req.url, req.resolution, meta["is_live"], d)

            upd(jid, stage="Transcription (Whisper)", progress=30)
            sents, words, lang = transcribe(src, req.language)
            (d / "transcript.txt").write_text("\n".join(s["text"] for s in sents), encoding="utf-8")
            upd(jid, language=lang, transcript=sents)

            upd(jid, stage="Viral moments dhoondh rahe hain", progress=50)
            picks = pick_clips(sents, req.clip_length, req.max_clips)
            if not picks:
                raise RuntimeError("Koi clip nahi mil payi (video mein bolne wali awaaz kam hai)")

            clips = []
            for k, p in enumerate(picks, 1):
                upd(jid, stage=f"Clip {k}/{len(picks)} render ho rahi hai", progress=50 + int(45 * (k - 1) / len(picks)))
                raw, final = d / f"raw{k}.mp4", d / f"clip{k}.mp4"
                w, h = render_vertical(src, p["start"], p["end"], raw, req.track_face)
                if req.captions:
                    ass = d / f"clip{k}.ass"
                    make_ass(words, p["start"], p["end"], req.caption_style, ass, w, h)
                    burn_captions(raw, ass, final)
                    raw.unlink(missing_ok=True)
                else:
                    raw.rename(final)
                clips.append({"index": k, "title": p["title"], "viral_score": p["score"],
                              "start": round(p["start"], 1), "end": round(p["end"], 1),
                              "download_url": f"/api/clips/{jid}/{k}/download"})
            src.unlink(missing_ok=True)
            upd(jid, status="done", stage="Taiyar", progress=100, clips=clips)
    except Exception as e:  # noqa: BLE001
        upd(jid, status="error", stage="Error", error=str(e)[:300])


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/fetch-stream")
def fetch_stream(body: LinkIn):
    """Video / live stream ki info (title, duration, live?, available qualities)."""
    check_url(body.url)
    try:
        return probe(body.url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Link read nahi ho paya: {str(e)[:200]}")


@app.post("/api/process-clip", status_code=202)
def process_clip(body: ProcessIn, bg: BackgroundTasks):
    """Job start karta hai. Progress ke liye /api/jobs/{job_id} poll karein."""
    check_url(body.url)
    try:
        meta = probe(body.url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Link read nahi ho paya: {str(e)[:200]}")
    if not meta["is_live"] and (meta["duration"] or 0) > MAX_SOURCE_MIN * 60:
        raise HTTPException(413, f"Video {MAX_SOURCE_MIN} minute se chhota hona chahiye")
    cleanup_old()
    jid = uuid.uuid4().hex
    (DATA_DIR / jid).mkdir(parents=True)
    JOBS[jid] = {"job_id": jid, "status": "queued", "stage": "Queue mein", "progress": 0, "meta": meta, "clips": []}
    bg.add_task(run_job, jid, body, meta)
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job nahi mila")
    return {k: v for k, v in job.items() if k != "transcript"}


@app.get("/api/jobs/{job_id}/transcript")
def job_transcript(job_id: str):
    job = JOBS.get(job_id)
    if not job or "transcript" not in job:
        raise HTTPException(404, "Transcript abhi taiyar nahi")
    return {"language": job.get("language"), "segments": job["transcript"]}


@app.get("/api/clips/{job_id}/{index}/download")
def download_clip(job_id: str, index: int):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Clip taiyar nahi")
    f = DATA_DIR / job_id / f"clip{index}.mp4"
    if not f.exists():
        raise HTTPException(404, "Clip nahi mili")
    return FileResponse(f, media_type="video/mp4", filename=f"short-{index}.mp4")
