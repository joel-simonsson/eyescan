# eyesopenscan

Finds the moments in an overnight video (IR night-vision camera) when the
sleeper's **eyes are open** and/or there is **strong body movement**. It
outputs timestamps, durations and short review clips, to help count nocturnal
seizures and track how often and how long they happen.

> This is a review aid, not a medical device. Normal waking also opens the
> eyes. Every event is a *candidate* that a person should check in the clip.
> For real-time safety alarms, use a certified seizure alarm.

## Setup

```sh
brew install ffmpeg uv
uv sync
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

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
