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

from pipeline import (audio_io, align, assemble, diarize, lyrics, lyrics_api,
                      metadata, pitch, separate)
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
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:2em auto;padding:0 1em}
 fieldset{border:1px solid #ccc;margin-bottom:1em}
 label{margin-right:1em} .ok{color:#080} .bad{color:#c00}
 #out{background:#f6f6f6;padding:1em;white-space:pre-wrap}
</style>
<h2>UltraStar auto-chart</h2>
<form id=f>
  <fieldset><legend>Song</legend>
    <input type=file name=audio required accept="audio/*"><br><br>
    Title <input name=title> Artist <input name=artist><br><br>
    Language <select name=lang>
      <option value=es selected>Spanish</option><option value=en>English</option>
      <option value=fr>French</option><option value=de>German</option>
      <option value=it>Italian</option><option value=pt>Portuguese</option>
      <option value=ja>Japanese</option><option value=auto>auto-detect</option>
    </select>
    Mode <select name=mode>
      <option value=auto selected>auto (duet if multi-artist)</option>
      <option value=no>solo</option>
      <option value=yes>duet (P1/P2)</option>
    </select>
    <br><br>Lyrics (optional — paste when LRCLIB misses the song; one line per sung line)<br>
    <textarea name=lyrics_text rows=4 style="width:100%"
      placeholder="Overrides the LRCLIB lookup. Line breaks become karaoke line breaks."></textarea>
  </fieldset>
  <fieldset><legend>Analysis</legend>
    <label><input type=checkbox name=separate checked> separate vocals (Demucs)</label>
    <label><input type=checkbox name=diarize checked> diarize singers</label>
    <label><input type=checkbox name=adlibs checked> drop ad-libs</label><br><br>
    Whisper model <select name=whisper>
      <option>tiny</option><option>base</option><option selected>small</option>
      <option>medium</option><option>large-v3</option>
    </select>
    <details style="display:inline-block;margin-left:1em"><summary>advanced</summary>
      Unison lines <select name=unison>
        <option value=both selected>both tracks</option><option value=lead>lead only</option>
      </select>
      <label><input type=checkbox name=multif0 checked> per-singer pitch</label>
      <label><input type=checkbox name=lyrics_api checked> canonical lyrics (LRCLIB)</label>
    </details>
  </fieldset>
  <button>Generate</button>
</form>
<div id=out></div>
<script>
const f=document.getElementById('f'), out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();out.textContent='uploading...';
 const r=await fetch('/upload',{method:'POST',body:new FormData(f)});
 const {job}=await r.json(); poll(job);};
async function poll(job){const r=await fetch('/status/'+job);const s=await r.json();
 if(s.status==='done'){out.innerHTML=summary(s,job);}
 else if(s.status==='error'){out.innerHTML='<span class=bad>error: '+esc(s.error)+'</span>';}
 else{out.textContent='working... stage: '+s.stage;setTimeout(()=>poll(job),1500);}}
function esc(t){const d=document.createElement('div');d.textContent=t??'';return d.innerHTML;}
function summary(s,job){
 let h='<b>'+(s.duet?'DUET (P1/P2)':'SOLO')+'</b> — '+s.notes+' notes / '
   +s.lines+' lines, '+Math.round(s.duration||0)+'s<br>';
 h+='lyrics: '+({pasted:'pasted (aligned)',lrclib:'canonical (LRCLIB)'}[s.lyric_source]
   ||'Whisper transcription')+'<br>';
 h+=s.problems&&s.problems.length
   ?'<span class=bad>problems:<br> - '+s.problems.map(esc).join('<br> - ')+'</span><br>'
   :'<span class=ok>validation OK</span><br>';
 h+='<br><a href="/download/'+job+'"><button>Download song folder (zip)</button></a>';
 return h;}
</script>
"""


def run_job(job_id: str, audio_path: str, title: str, artist: str,
            lang: str, do_separate: bool, do_diarize: str = "auto",
            duet: str = "auto", whisper: str = "small", unison: str = "both",
            adlibs: str = "drop", multif0: str = "auto",
            use_lyrics_api: bool = True, lyrics_text: str = "") -> None:
    job = JOBS[job_id]
    try:
        job.update(status="analyzing", stage="duration")
        duration = audio_io.duration_seconds(audio_path)

        analysis = audio_path
        vocals_wav = None
        if do_separate:
            job["stage"] = "separating"
            voc = separate.separate_vocals(audio_path)
            if voc:
                analysis = voc
                vocals_wav = voc

        job["stage"] = "pitch"
        audio, sr = audio_io.load_mono(analysis, sr=16000)
        ptrack = pitch.extract_pitch(audio, sr)

        job["stage"] = "lyrics"
        wlang = None if lang == "auto" else lang
        words = lyrics.transcribe_words(analysis, model_size=whisper, language=wlang)
        # deterministic recognition: replace text with canonical lyrics —
        # pasted lyrics take precedence over the LRCLIB lookup
        warnings = []
        lyric_source = "whisper"
        pasted = [l.strip() for l in lyrics_text.splitlines() if l.strip()]
        if pasted:
            aligned = align.build_words(pasted, words)
            if aligned:
                words = aligned
                lyric_source = "pasted"
            else:
                warnings.append("pasted lyrics did not align with the audio; "
                                "ignored")
        if lyric_source == "whisper" and use_lyrics_api:
            canon = lyrics_api.fetch_lyrics(artist, title, duration)
            if canon:
                aligned = align.build_words(canon["lines"], words)
                if aligned:
                    words = aligned
                    lyric_source = "lrclib"
        warn = lyrics.low_words_warning(len(words), duration)
        if warn:
            warnings.append(warn)

        # speaker diarization (+ per-singer pitch) for the duet split
        diar = []
        want_diar = do_diarize == "yes" or (
            do_diarize == "auto" and (duet == "yes"
                                      or assemble.is_multi_artist(artist)))
        if want_diar and duet != "no" and vocals_wav:
            job["stage"] = "diarize"
            diar = diarize.diarize(vocals_wav, num_speakers=2)
        pps = {}
        if diar and multif0 != "no" and vocals_wav:
            job["stage"] = "multif0"
            pps = pitch.extract_pitch_per_speaker(vocals_wav, diar)

        job["stage"] = "assemble"
        chart = assemble.assemble(
            words, ptrack, duration, title=title, artist=artist,
            audio=os.path.basename(audio_path),
            language=(wlang.upper() if wlang else None),
            duet=duet, diarization=diar, pitch_per_speaker=pps,
            unison=unison, drop_adlibs=(adlibs == "drop"),
        )
        problems = usdx_validate.validate(chart) + warnings

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
                   lines=sum(len(tr) for tr in tracks),
                   duet=bool(chart.tracks), duration=duration,
                   lyric_source=lyric_source,
                   words=len(words), problems=problems)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=repr(e))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.post("/upload")
async def upload(audio: UploadFile = File(...), title: str = Form(""),
                 artist: str = Form(""), lang: str = Form("es"),
                 mode: str = Form("auto"), whisper: str = Form("small"),
                 unison: str = Form("both"),
                 separate: bool = Form(False), diarize: bool = Form(False),
                 adlibs: bool = Form(False), multif0: bool = Form(False),
                 lyrics_api: bool = Form(False),
                 lyrics_text: str = Form("")) -> JSONResponse:
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

    # whitelist client-supplied option strings (trust boundary)
    if whisper not in ("tiny", "base", "small", "medium", "large-v3"):
        whisper = "small"
    mode = mode if mode in ("auto", "yes", "no") else "auto"
    unison = unison if unison in ("both", "lead") else "both"
    lang = lang if re.fullmatch(r"[a-z]{2}|auto", lang or "") else "es"
    lyrics_text = (lyrics_text or "")[:20000]  # cap pasted lyrics (trust boundary)

    JOBS[job_id] = {"status": "queued", "stage": "queued"}
    threading.Thread(
        target=run_job,
        args=(job_id, audio_path, title, artist, lang, separate),
        kwargs={"do_diarize": "auto" if diarize else "no",
                "duet": mode, "whisper": whisper, "unison": unison,
                "adlibs": "drop" if adlibs else "keep",
                "multif0": "auto" if multif0 else "no",
                "use_lyrics_api": lyrics_api, "lyrics_text": lyrics_text},
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
