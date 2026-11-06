# eyesopenscan

Finds the moments in an overnight video (IR night-vision camera) when the
sleeper's **eyes are open** and/or there is **strong body movement**. It
outputs timestamps, durations and short review clips, to help count nocturnal
seizures and track how often and how long they happen.

> This is a review aid, not a medical device. Normal waking also opens the
> eyes. Every event is a *candidate* that a person should check in the clip.
> For real-time safety alarms, use a certified seizure alarm.

## Setup

### macOS

```sh
brew install ffmpeg uv
uv sync
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
uv run pytest    # quick check that everything works
```

### Windows

> Not yet tested on Windows. Everything it uses (Python, MediaPipe, OpenCV,
> ffmpeg) is cross-platform, but please report any problems.

In PowerShell:

```powershell
# 1. Tools (open a NEW terminal afterwards so ffmpeg is on PATH)
winget install astral-sh.uv
winget install Gyan.FFmpeg
winget install Git.Git

# 2. Code, dependencies (uv downloads Python 3.12 by itself) and the face model
git clone <repo-url> eyesopenscan      # or copy the folder, without .venv and out
cd eyesopenscan
uv sync
curl.exe -L -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
uv run pytest    # quick check that everything works
```

Use `curl.exe`, not `curl`. In Windows PowerShell, `curl` is a different
command.

All commands in this README work the same on Windows. Only the shell
syntax differs:
- Put RTSP URLs in **double quotes**, especially if the password contains
  `&`, `$` or similar characters.
- To continue a command on the next line, use a backtick `` ` ``, not `\`.
  You can also write it all on one line.
- Paths use `\`, e.g. `uv run eyesopenscan analyze out\live-2026-11-06_2230`.

```powershell
uv run eyesopenscan live "rtsp://USER:PASS@192.168.1.50:554/stream1" `
    --face-roi 640,180,700,520 --until 07:30
```

Differences from macOS:
- **Sleep is not prevented automatically.** On macOS the tool uses
  `caffeinate`; on Windows you must disable sleep yourself for the night.
  Either go to Settings → System → Power → Sleep: *Never* (when plugged in),
  or run `powercfg /change standby-timeout-ac 0` once. If the PC sleeps, the
  session stops. The data up to the last checkpoint can still be recovered
  with `analyze`.
- **Ctrl+C can take up to about 10 s** to react if the camera happens to be
  offline at that moment. With a working stream it reacts immediately.

## Hardware requirements

| | Recommendation | Why |
|---|---|---|
| **Computer** | 4+ CPU cores, any reasonably modern Intel/AMD or Apple Silicon, 8 GB RAM | Decoding the 1080p stream, plus face analysis 5× per second on the CPU |
| **GPU** | Not needed | Everything runs on the CPU |
| **Disk** | A few GB free | Live mode keeps about 5 min of stream plus flagged moments (about 15–30 MB per minute of video, depending on camera bitrate), plus the clips |
| **Network** | Wired PC-to-router connection if possible; camera with a good Wi-Fi signal | Fewer stream dropouts. Reconnection is automatic, but nothing is analyzed while the stream is down |
| **Power** | Plugged in, sleep disabled | The computer must stay awake all night |
| **Camera** | IR night vision, RTSP support, ≥ 1080p main stream (2K is better) | Eye detection needs enough pixels on the face (see *Camera tips*) |

Measured on an M1 Pro MacBook:
- Live mode kept up with a 1080p stream, with eye analysis less than 0.3 s behind.
- Recorded files process at roughly 10× real time.

To check whether a slower machine keeps up, watch `eye lag` in the
status line printed every minute. If it stays under about 1 s, the machine
is fast enough.

## Typical workflow

```sh
# 1. Once per camera setup: check that the detector can see the eyes
uv run eyesopenscan debug night.mp4 --at 0:10:00 --duration 60 -o debug.mp4

# 2. Once per person/camera angle: calibrate from segments you know
#    (eyes closed asleep; eyes open, e.g. lying awake before falling asleep)
uv run eyesopenscan calibrate night.mp4 --closed 0:30:00-0:31:00 --open 0:01:00-0:01:30

# 3. Every night
uv run eyesopenscan scan night.mp4 --calibration calibration.json

# 4. Re-tune without re-processing the video (takes seconds)
uv run eyesopenscan analyze out/night --calibration calibration.json --motion-k 4 --eye-min-dur 3
```

Optional regions (draw a box with `uv run eyesopenscan pick-roi night.mp4 --at 0:10:00`):
- `--face-roi x,y,w,h`: limits the face search to the pillow area. This is
  faster and gives fewer false detections (for example a face on a poster).
- `--body-roi x,y,w,h`: limits motion detection to the bed, which ignores a
  partner, pets, curtains and so on.

## Live mode (camera stream, overnight)

Start it in the evening. In the morning press **Ctrl+C**, or let `--until`
stop it by itself. Either way you get the same report and clips as `scan`:

```sh
uv run eyesopenscan live "rtsp://USER:PASS@192.168.1.50:554/stream1" \
    --calibration calibration.json --until 07:30
```

- **Camera URL:** enable RTSP or "camera account" in the camera's app. Tapo:
  `rtsp://user:pass@IP:554/stream1`. Reolink: `rtsp://user:pass@IP:554/h264Preview_01_main`.
  Eufy: enable RTSP in the app (it shows the URL). Use the camera's *main*
  (high-resolution) stream, because the eyes need the pixels. Passwords are
  never written to logs or `meta.json`.
- **Video is clips only, not the whole night.** The stream is copied without
  re-encoding into 30 s segments in a rolling buffer. Segments older than 5 min
  (`--retention-min`) are deleted unless something suspicious happened in them,
  meaning eyes possibly open or movement. When the session stops, the normal
  analysis runs, a clip is cut for each event, and the buffer is deleted.
  - Disk use while running is about 5 minutes of stream plus the flagged moments.
  - The first 5 minutes are always kept while the baselines settle.
- **Stopping:**
  - Ctrl+C, SIGTERM or `--until` finishes the last segment, then builds the
    report and clips.
  - Pressing Ctrl+C a second time during that step skips the clips but keeps
    everything. Finish later with `uv run eyesopenscan analyze out/live-…`.
  - `--until` accepts `07:30` (the next time it is 07:30), `+8h`, or a full date and time.
- **Stream dropouts** (Wi-Fi, a camera reboot): it reconnects automatically. The
  offline periods are listed in `summary.txt` and hatched in `timeline.png`.
- **Crash safety:** signals are saved every 5 min (`--checkpoint-min`). After a
  crash or power cut, `uv run eyesopenscan analyze out/live-…` still gives a
  report from the saved data, with clips from the buffered segments that remain.
- **Keep the computer awake.** On macOS the tool prevents idle sleep
  (`caffeinate`), but a MacBook with its **lid closed on battery** still
  sleeps. Leave it open or plugged in. On Windows you must disable sleep
  yourself (see *Setup → Windows*).
- `pick-roi` also works on a stream: `uv run eyesopenscan pick-roi "rtsp://…"`.
  The ROIs are in the stream's pixel coordinates.

## Output (`out/<video>/`)

| file | content |
|---|---|
| `events.csv` | one row per event: start/end, duration, `type` (`both` / `eyes` / `motion`), confidence, rhythmicity, ... |
| `summary.txt` | count, events per hour, total/mean/longest duration, time between events, % of the night the face was visible |
| `timeline.png` | eye openness, motion and events across the night |
| `clips/` | one MP4 per event, padded by 10 s, for quick review |
| `signals.npz` | raw per-frame signals (used by `analyze`) |

Reading the event types:
- **both**: eyes open *and* strong movement at the same time. This is the strongest seizure candidate.
- **eyes**: eyes open while lying still. This can be a seizure without jerking, or simply waking up.
- **motion**: strong movement with the face not visible or the eyes closed. A `rhythmicity`
  value of ≥ 0.4 means rhythmic 2–8 Hz movement (jerking), unlike turning over.

## How it works

1. **Eyes** (5 fps): CLAHE contrast enhancement on the IR frame, then the
   MediaPipe Face Landmarker gives eyelid landmarks, then the Eye Aspect Ratio
   (EAR).
   - The detector searches square tiles and 4 rotations, because a face lying
     sideways on a pillow in a wide shot is otherwise missed. After that it
     tracks a crop around the face.
   - A missing face is recorded as "no face", never as "eyes closed".
2. **Motion** (15 fps): the share of pixels that changed between frames in the
   body region. The threshold adapts to the night (rolling median + k·MAD).
   Frames where the whole picture changes, such as an IR on/off switch, are
   ignored.
3. **Events**:
   - A rolling-median filter and hysteresis smooth the signals.
   - Events shorter than the minimum duration are dropped. The minimum is 2 s
     for eyes, which removes blinks, and 3 s for motion.
   - Eye and motion events less than 3 s apart are merged into one event.

Without calibration, the eye thresholds are set automatically from the
person's typical (closed-eye) EAR for that night. That works when the eyes are
closed most of the time the face is visible. Calibration is more reliable.

## Camera tips (matters more than the software)

- IR night-vision camera, ≥ 1080p (2K is better), ≥ 15 fps, recording to a
  file.
- Mount it **1–1.5 m from the pillow, slightly above the face**, with the head
  and upper body in view. Faces seen side-on or from far away are the main
  cause of missed eyes.
- Check `debug` output in real darkness. If the eyes look like dark blobs, add
  an 850 nm IR illuminator or move the camera closer.

## Tests

```sh
uv run pytest
```
