"""Minimal web upload service: POST an audio file, get back a zipped USDX song
folder. Runs the same pipeline as generate.py in a background thread per job.

    uvicorn app.main:app --reload      # from the usdx-autochart dir
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from pipeline import (audio_io, align, assemble, lyrics, lyrics_api, metadata,
                      pitch, separate)
from pipeline import usdx_validate, usdx_writer

app = FastAPI(title="usdx-autochart")

JOBS: dict[str, dict] = {}
WORK = tempfile.mkdtemp(prefix="usdx_jobs_")


def safe_component(value: str, fallback: str, maxlen: int = 80) -> str:
    """Sanitize a user string for use as a single path component: strip any
    directory parts, allow only a conservative charset, never empty."""
    value = os.path.basename((value or "").strip())
    value = re.sub(r"[^A-Za-z0-9 ._()'\-áéíóúñüÁÉÍÓÚÑÜ]", "_", value).strip(" .")
    return (value[:maxlen] or fallback)


def safe_audio_name(filename: str | None) -> str:
    raw = os.path.basename(filename or "audio.mp3")
    ext = os.path.splitext(raw)[1].lower()
    if ext not in (".mp3", ".ogg", ".wav", ".m4a", ".flac", ".opus"):
        ext = ".mp3"
    return "input" + ext

PAGE = """
<!doctype html><meta charset=utf-8><title>USDX Autochart</title>
<h2>UltraStar auto-chart</h2>
<form id=f>
  <input type=file name=audio required accept="audio/*"><br><br>
  Title <input name=title> Artist <input name=artist>
  Lang <input name=lang value=es size=3>
  <label><input type=checkbox name=separate> separate vocals</label><br><br>
  <button>Generate</button>
</form>
<pre id=out></pre>
<script>
const f=document.getElementById('f'), out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();out.textContent='uploading...';
 const r=await fetch('/upload',{method:'POST',body:new FormData(f)});
 const {job}=await r.json(); poll(job);};
async function poll(job){const r=await fetch('/status/'+job);const s=await r.json();
 out.textContent=JSON.stringify(s,null,2);
 if(s.status==='done'){out.textContent+='\\n\\nDownload: /download/'+job;}
 else if(s.status==='error'){}
 else setTimeout(()=>poll(job),1500);}
</script>
"""


def run_job(job_id: str, audio_path: str, title: str, artist: str,
            lang: str, do_separate: bool) -> None:
    job = JOBS[job_id]
    try:
        job.update(status="analyzing", stage="duration")
        duration = audio_io.duration_seconds(audio_path)

        analysis = audio_path
        if do_separate:
            job["stage"] = "separating"
            voc = separate.separate_vocals(audio_path)
            analysis = voc or audio_path

        job["stage"] = "pitch"
        audio, sr = audio_io.load_mono(analysis, sr=16000)
        ptrack = pitch.extract_pitch(audio, sr)

        job["stage"] = "lyrics"
        words = lyrics.transcribe_words(analysis, language=lang)
        # deterministic recognition: replace text with canonical lyrics if found
        canon = lyrics_api.fetch_lyrics(artist, title, duration)
        if canon:
            aligned = align.build_words(canon["lines"], words)
            if aligned:
                words = aligned

        job["stage"] = "assemble"
        chart = assemble.assemble(
            words, ptrack, duration, title=title, artist=artist,
            audio=os.path.basename(audio_path), language=lang.upper(),
            duet="auto",
        )
        problems = usdx_validate.validate(chart)

        song_dir = os.path.join(WORK, job_id, f"{artist} - {title}")
        os.makedirs(song_dir, exist_ok=True)
        shutil.copy2(audio_path, os.path.join(song_dir, os.path.basename(audio_path)))
        cover = metadata.ensure_cover(audio_path, song_dir, title, artist)
        if cover:
            chart.cover = cover
        usdx_writer.write(chart, os.path.join(song_dir, f"{artist} - {title}.txt"))

        zip_path = os.path.join(WORK, job_id, "song.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _d, files in os.walk(song_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, os.path.dirname(song_dir)))

        tracks = chart.tracks if chart.tracks else [chart.lines]
        job.update(status="done", stage="done", zip=zip_path,
                   notes=sum(len(l.notes) for tr in tracks for l in tr),
                   duet=bool(chart.tracks),
                   words=len(words), problems=problems)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=repr(e))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.post("/upload")
async def upload(audio: UploadFile = File(...), title: str = Form(""),
                 artist: str = Form(""), lang: str = Form("es"),
                 separate: bool = Form(False)) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WORK, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # server-generated name; never trust the client filename for the path
    audio_path = os.path.join(job_dir, safe_audio_name(audio.filename))
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    base = os.path.splitext(os.path.basename(audio.filename or ""))[0]
    # honour "Artist - Title" upload names, else fall back to ID3 tags
    r_title, r_artist = metadata.resolve_title_artist(
        base if " - " in base else audio_path, title or None, artist or None)
    title = safe_component(title or r_title, "Untitled")
    artist = safe_component(artist or r_artist, "Unknown")

    JOBS[job_id] = {"status": "queued", "stage": "queued"}
    threading.Thread(target=run_job,
                     args=(job_id, audio_path, title, artist, lang, separate),
                     daemon=True).start()
    return JSONResponse({"job": job_id})


@app.get("/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JSONResponse(JOBS[job_id])


@app.get("/download/{job_id}")
def download(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(job["zip"], filename="usdx-song.zip",
                        media_type="application/zip")
