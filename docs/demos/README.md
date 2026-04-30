# Demo videos — shot lists

Four 30-60s clips that showcase capabilities our integration ships
that aren't in any other Reachy Mini voice integration. Recorded once
hardware is back; this file is the script.

Recording rig: phone on a tripod about 60 cm from Reachy at desk level,
landscape, 1080p60. Mic close enough to clearly capture the operator's
voice — Reachy's response audio comes from its own onboard speaker so
it'll always be clear in the room.

Captions are burned in post (titles + a 2-line "what's happening" so
the clip plays without sound).

---

## 1. Watch mode — "Watch the door for someone walking in"

**Capability**: persistent vision monitor with VLM-checked condition;
fires a voice alert when condition is met. Unique to jetson-assistant
in the Reachy Mini ecosystem.

**Setup**: A second person in frame, off-camera initially. Reachy
positioned with a clear view of a door or hallway. Profile:
`curious_reachy`.

**Shot list (45 s):**
1. (0-5 s) Operator says: "Watch the door — tell me when someone
   walks in." Reachy nods (`yeah_nod` from dance lib) and replies
   "I'll watch."
2. (5-25 s) Cut to fast-forward of the empty doorway. Caption:
   *"Reachy is checking the camera every 5 seconds, asking the VLM:
   is someone there?"*
3. (25-35 s) Person walks in. Within 5-7 s, Reachy says
   "Hey, someone just came in!" Caption: *"VLM detection → voice
   alert."*
4. (35-45 s) Operator: "Stop watching." Reachy:
   `play_emotion("understanding1")` + "Stopped."

**Title card**: *"Persistent vision monitor — Reachy keeps an eye on
things even when you're not looking."*

---

## 2. Browser teleop — "Your phone is Reachy's eyes"

**Capability**: WebSocket PCM mic + speaker plus live MJPEG video
served from the assistant. Operator's phone connects via browser, no
app install. Unique among local-only Reachy integrations.

**Setup**: Phone in operator's hand, Reachy on a desk. Stream port
9090 enabled (`stream_vision: 9090` in config).

**Shot list (50 s):**
1. (0-5 s) Operator types `https://10.0.0.28:9090/` into phone
   browser. Browser shows: live camera feed + chat transcript
   panel. Caption: *"No app install. Just open the URL."*
2. (5-15 s) Operator taps "Connect Audio" on the browser. Speaks
   into phone: "What do you see?" Reachy responds via its own
   speaker AND the phone's speaker (browser playback). Caption:
   *"Same audio, two endpoints — full duplex over WebSocket."*
3. (15-30 s) Operator walks across the room with the phone. Phone
   shows Reachy's view in real time, transcript scrolls as Reachy
   continues to respond. Caption: *"You can be anywhere on the LAN."*
4. (30-50 s) Operator from another room: "Look left — and tell me
   what's there." Phone shows Reachy turning, then VLM describing
   what's in frame. Caption: *"Embodied teleop without rolling your
   own iOS app."*

**Title card**: *"Your phone, Reachy's eyes."*

---

## 3. Multi-language hot-switch — "Speak Japanese"

**Capability**: STT, LLM, TTS all switch language mid-conversation
on a single voice command. Unique to jetson-assistant.

**Setup**: `curious_reachy` profile (or `language_learner` for an
inverse-direction demo). English by default.

**Shot list (40 s):**
1. (0-5 s) Operator (English): "What's your favorite food?"
   Reachy (English Kokoro voice): "I don't eat — but watching humans
   eat looks fun."
2. (5-15 s) Operator: "Speak Japanese." Reachy:
   `play_emotion("inquiring1")` + a one-line acknowledgment in
   Japanese. Caption: *"STT, LLM, AND TTS just hot-switched."*
3. (15-30 s) Operator (Japanese, basic): "好きな色は何ですか?"
   ("What's your favorite color?") Reachy responds in Japanese
   with an appropriate emotion. Caption: *"Same brain, different
   voice + comprehension."*
4. (30-40 s) Operator: "Switch back to English." Reachy: a one-
   liner in English to confirm.

**Title card**: *"One robot, nine languages — switch live."*

---

## 4. One Thor → two Reachys

**Capability**: jetson-assistant's `serve` mode keeps models loaded
once on a beefy host; multiple lightweight clients (robots, laptops,
browsers) connect simultaneously. Unique architecturally — every
other Reachy integration assumes single-machine.

**Setup**: 2 Reachy Minis on the same desk, ~50 cm apart. Both Pi5s
running `run-conversation-app.sh` with `LMSTUDIO_ENDPOINT` pointing
to the same Thor. Both with `curious_reachy` profile but different
Kokoro voices (`af_heart` and `am_michael`) so they sound distinct.

**Shot list (60 s):**
1. (0-10 s) Caption: *"Two robots. One inference server."* Camera
   pans both Reachys idle-breathing.
2. (10-25 s) Operator: "Hey Reachy on the left — wave at the
   camera." LEFT one waves. Caption: *"Per-robot system prompts can
   give them different names if you want."*
3. (25-45 s) Operator: "Both of you — do a dance!" Both robots
   dance simultaneously, with slightly different choreographies
   (random pick from the dance library). Caption: *"Same Thor
   serving both. ~50 ms extra latency from the second client."*
4. (45-60 s) Show `nvidia-smi` on Thor: only one vLLM process,
   serving both robots. Caption: *"Scales linearly — each new
   robot costs you a Pi5, not a GPU."*

**Title card**: *"Brain on Thor. Body anywhere."*

---

## Notes on capture

- Use `--debug` flag so the launcher logs are clean enough to
  screenshot if needed.
- Record a dry run of each clip first; the LLM + tool-call timing is
  the part most likely to bite you.
- Trim aggressively in post — every clip should hit its title card
  in <10 s.
- Caption font: same across all four. Mono caps, bottom-third.
