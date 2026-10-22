# voice-agent

Talk with Claude Code out loud, like talking to a colleague. You discuss a task, Claude hands it to a background agent once you agree, and you keep chatting while the agent works. When the agent finishes, Claude tells you what happened.

All the speech runs on your own machine:

- **Hearing:** [Moonshine](https://github.com/moonshine-ai/moonshine) Streaming Medium by Moonshine AI (MIT, 245M parameters) on the CPU, with WebRTC voice detection. It is about as accurate as Whisper Large v3 at a sixth of the size, and favours the names you give it ("Claude", your project's words). NVIDIA [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) through [onnx-asr](https://github.com/istupakov/onnx-asr) is still there as a setting, and can use an NVIDIA GPU. Either one transcribes during the pause at the end of what you say, so the text is usually ready the moment the pause counts as the end.
- **Knowing when you're done:** [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) by Pipecat (BSD 2-Clause, 9 MB int8 ONNX) listens to how a sentence ends. If you sound finished, Claude answers about 0.15 s into the pause; if you trail off mid-thought ("can you check whether the…", "so I was thinking, um…"), it waits up to 1.2 s for you to go on. See [End-of-turn detection](#end-of-turn-detection).
- **Speaking:** [Phonon](https://gradium.ai/phonon) 31m by Gradium: open weights (Apache 2.0), about 65 MB, streams audio while it generates.
- **Audio:** PipeWire/PulseAudio (`parecord`/`paplay`), falling back to ALSA (`arecord`/`aplay`).

It is built and tested for Linux, Omarchy (Arch + Hyprland + PipeWire) in particular.

## How it works

```
 you ─► mic ─► voice detection ─► Moonshine ─► "hey Claude, add dark mode"
                                                  │
                                                  ▼
                                   Claude, the partner (talks, plans, asks)
                                     │                      │
                     you agree ──► delegate_task      replies stream ─► Phonon ─► speaker
                                     │
                                     ▼
                         background agent works on its own
                                     │
                       report ─► Claude tells you in a sentence or two
```

- **The partner.** While voice is on, the main Claude gets a system-prompt section that makes it conversational: short spoken replies, no markdown, one question at a time. It talks the task through with you, and once you say yes it calls `delegate_task` with a self-contained brief.
- **Background agents.** `delegate_task` starts a background subagent and returns at once, so the conversation continues. Several can run at once. Ask "how's it going?" and Claude calls `list_tasks`, which shows each agent's latest activity. When an agent finishes, its report goes back to the partner, which summarizes it out loud.
- **Streaming speech.** Replies are read sentence by sentence as they stream in, so Claude starts talking before it has finished writing. Code blocks are not read out ("I've put the code on screen").
- **Barge-in.** Talk over Claude and it stops. Say "stop" to silence it.
- **Permissions by voice.** When Claude or an agent needs permission, you hear "The dark mode agent wants to run the unit tests. Is that okay?" Shell commands are never read out: you hear the short description Claude gave the command, or, without one, just the program ("wants to run a git command"). Answer yes or no. If you don't answer, the normal permission prompt appears.

## Listening modes

| Mode | How you talk to it |
| --- | --- |
| `wake` (default) | Say **"hey Claude"**, then your request, in one breath or after the chime. Right after Claude speaks you can reply without the wake word (8 s by default). |
| `natural` | The mic is always open, like a phone call. Use headphones or echo cancellation (below). |
| `ptt` | Only while you hold the push-to-talk key. |

Push-to-talk works in every mode. Switch modes with `/talk mode natural`, the **Mode** button in the Voice pane, or by saying "switch to natural mode", "switch to push to talk" or "switch to wake word mode".

## Install

### 1. System packages (Omarchy / Arch)

```sh
sudo pacman -S --needed uv curl pipewire-pulse
```

`uv` runs the voice helper and installs its Python dependencies (moonshine-voice, onnx-asr with onnxruntime, ptts, webrtcvad, numpy) into its own cache on first start. Nothing is installed system-wide.

### 2. The mod

At the prompt of a Claude Code terminal session:

```
/plugin install voice-agent --marketplace sabeel111/voice-agent-mod
```

From a shell:

```sh
claude plugin install voice-agent --marketplace sabeel111/voice-agent-mod
```

In the desktop app, where `/plugin` does not run: click + next to the prompt box, then Plugins, Add marketplace with `sabeel111/voice-agent-mod`, then Add plugin.

### 3. Turn it on

```
/talk on
```

The first start downloads Moonshine Streaming Medium from Moonshine AI's server (quantized; smaller than Parakeet's 650 MB), Phonon 31m (about 65 MB) and Smart Turn v3 (9 MB) from Hugging Face, which takes a minute. Parakeet, if you pick it, is about 650 MB int8 for the CPU; on an NVIDIA GPU the CUDA runtime, about 2 GB, and the full model, about 2.4 GB. Later starts take a few seconds. A **Voice** pane opens with the live state, your tasks and the conversation.

### 4. Push-to-talk key (Hyprland)

Run `/talk keys` to print bindings for your setup. Add them to `~/.config/hypr/bindings.conf`; Hyprland reloads on save:

```ini
# Hold Super+` to talk
bind = SUPER, grave, exec, curl -s --unix-socket "$XDG_RUNTIME_DIR/claude-voice.sock" -X POST http://voice/ptt/down
bindr = SUPER, grave, exec, curl -s --unix-socket "$XDG_RUNTIME_DIR/claude-voice.sock" -X POST http://voice/ptt/up

# Or press once to start and again to send
# bind = SUPER SHIFT, grave, exec, curl -s --unix-socket "$XDG_RUNTIME_DIR/claude-voice.sock" -X POST http://voice/ptt/toggle

# Make Claude stop talking
bind = SUPER CTRL, grave, exec, curl -s --unix-socket "$XDG_RUNTIME_DIR/claude-voice.sock" -X POST http://voice/stop
```

The key reaches whichever Claude Code session has voice on. Only one session holds the microphone at a time: turning voice on in another session takes it over.

## Commands

| | |
| --- | --- |
| `/talk` | Toggle voice on or off |
| `/talk on`, `/talk off` | |
| `/talk mode wake\|natural\|ptt` | Change how it listens |
| `/talk mute`, `/talk unmute` | Stop hands-free listening (push-to-talk still works) |
| `/talk stop` | Stop talking now |
| `/talk status` | State, tasks and socket path |
| `/talk keys` | Hyprland push-to-talk bindings |
| `/talk pane` | Reopen the Voice pane |

Things you can say that the mod handles itself, without sending them to Claude: "stop" / "be quiet", "stop listening" / "start listening", "switch to … mode", "turn off voice".

## Settings

Change these in `/config` (or `/plugin` → voice-agent → configure):

| Setting | Default | |
| --- | --- | --- |
| Listening mode | `wake` | `wake`, `natural` or `ptt` |
| Wake word | `hey claude` | Matched loosely: "hey cloud" and "hey Clyde" also count |
| Speech recognition model | `moonshine-medium` | `moonshine-small` (123M) and `moonshine-tiny` (34M) are lighter and less accurate. `parakeet-v2` (English) and `parakeet-v3` (25 European languages) are NVIDIA's larger models |
| Speech recognition device | `auto` | Parakeet only: an NVIDIA GPU when there is one, else the CPU. `cpu` skips the GPU download. Moonshine always runs on the CPU |
| Key terms | (none) | Names and jargon to recognize more reliably, comma-separated, e.g. `Omarchy, Hyprland, Kubernetes`. Moonshine only; the wake word's name is always included |
| Smart end-of-turn detection | on | Answer as soon as you sound finished; wait through pauses mid-sentence. Off: a fixed pause ends what you say (see below) |
| Voice detection strictness | `3` | `0` to `3`. Higher ignores more background noise but may miss very quiet speech |
| Minimum speech length (ms) | 500 | Shorter sounds are ignored |
| Minimum loudness over background | 2 | How many times louder than the background noise speech must be. Higher ignores more background voices; `0` turns it off |
| Pause that ends your turn (ms) | 800 | How long a silence means you've finished talking. Longer lets you pause mid-sentence; shorter replies sooner. Used when turn detection is off |
| Language | `en` | Other languages need a multilingual Phonon 40m model. Moonshine Medium is English only: Spanish, German and Japanese use Moonshine Small, Arabic, Vietnamese, Chinese and Tagalog Moonshine Tiny, the rest `parakeet-v3` |
| Phonon voice model | Phonon 31m | e.g. `hf://gradium/phonon-40m-en` (non-commercial license, gated: accept its terms on Hugging Face, then `hf auth login`) |
| Voice | model default | List voices with `uvx ptts --list-voices` |
| Interrupt by talking | on | |
| Follow-up window | 8 s | Wake mode: how long you can reply without the wake word |
| Chimes | on | |
| Answer permission prompts by voice | on | |
| Model for the conversation | session's | A faster model (e.g. `sonnet`) makes the partner snappier; agents keep their own |
| Thinking effort for the conversation | default | `low` answers faster |
| Model / type for background agents | defaults | |
| Start voice with each session | off | |

## End-of-turn detection

About 0.15 s into each pause, two things run side by side on the CPU:

- **Smart Turn v3**, an audio model (a Whisper Tiny encoder with a classifier, 9 MB int8), gives the probability that you have finished, from the last 8 s of sound: intonation, rhythm and the words themselves.
- **Moonshine** (or Parakeet) transcribes what you said so far.

Then:

| Smart Turn says | Transcript | Ends the utterance |
| --- | --- | --- |
| finished (p ≥ 0.5) | does not stop on "the", "and", "um", a comma… | as soon as the transcript is in, usually 0.2 to 0.3 s into the pause |
| finished | stops mid-sentence | after 1.2 s of silence |
| not finished | | after 1.2 s of silence |
| model unavailable | | after the fixed pause (0.8 s by default) |

Speaking again during the pause starts over, so a pause for thought never sends half a sentence. The transcript check is for English only; other languages rely on Smart Turn alone.

Measured on a Ryzen 5 5600H laptop (CPU only, int8 Parakeet, a busy machine), from the end of speech to the transcript event, with synthesized speech through a fake real-time mic:

| | Finished sentence (median) | Sentence with a 0.6 s pause in the middle |
| --- | --- | --- |
| Fixed 0.5 s pause (before) | 620 ms | cut in two, every time |
| Smart Turn | 520 ms | kept whole, every time; sent about 0.6 s after the end |

Most of what remains is the recognizer itself (0.2 to 0.3 s for a short request on this CPU) plus the voice detector's own hangover. Smart Turn takes 80 to 150 ms on this CPU (about 10 ms on recent ones with VNNI), but runs alongside the transcription, so it adds nothing. Turn it off with the **Smart end-of-turn detection** setting; the model is downloaded on first use to `~/.cache/voice-agent/models/smart-turn`.

## Natural mode without headphones: echo cancellation

With speakers, the mic hears Claude's own voice. The helper ignores what it just said and needs a loud, sustained voice to barge in, but PipeWire's echo canceller works much better. A starting point, `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`:

```
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      library.name = aec/libspa-aec-webrtc
      monitor.mode = true
      source.props = { node.name = "echo-cancel-source" node.description = "Echo-cancelled mic" }
    }
  }
]
```

Then `systemctl --user restart pipewire` and pick "Echo-cancelled mic" as the default input (`wiremix`, or `pactl set-default-source echo-cancel-source`).

## Troubleshooting

- **"could not start the voice helper"**: install `uv` (`sudo pacman -S uv`). Or point the **Voice helper command** setting at your own Python environment, e.g. `~/.venvs/voice/bin/python /path/to/helper/voice_helper.py`, after `pip install numpy moonshine-voice 'onnx-asr[cpu,hub]' ptts webrtcvad-wheels` (for the GPU, `onnxruntime-gpu[cuda,cudnn]` in place of `onnxruntime`).
- **"no recorder found"**: install `pipewire-pulse` (it provides `parecord`/`paplay`) or `alsa-utils`.
- **It doesn't react to "hey Claude"**, or mishears names: add them under **Key terms**, use `moonshine-medium` if you picked a smaller model, speak a little slower, or set your own wake word. `/talk status` shows the last notice.
- **Model downloads fail**: Moonshine comes from download.moonshine.ai, the rest from huggingface.co, so check that both are reachable. The 40m Phonon models also need `hf auth login`.
- **Run the helper by hand** to see everything it logs:
  ```sh
  uv run --script helper/voice_helper.py --socket /tmp/v.sock --config '{"mode":"natural"}'
  ```

## Development

```sh
claude plugin validate .          # manifest and hooks module
claude plugin test .              # the mod's tests against the engine
uv run --with numpy --with webrtcvad-wheels --with pytest pytest helper -q   # the voice helper's tests
```

Layout:

- `hooks/register.tsx`: the mod. Starts the helper, streams speech, runs the partner prompt, the `delegate_task` and `list_tasks` tools, voice permissions, `/talk` and the Voice pane.
- `hooks/speech.ts`: sentence splitting, markdown-to-speech, spoken commands, permission wording.
- `hooks/prompts.ts`: what the model reads.
- `helper/voice_helper.py`: the local voice process: mic, voice detection, wake word, push-to-talk, Moonshine and Parakeet, Smart Turn end-of-turn detection, Phonon, barge-in and spoken yes/no questions, controlled over a Unix socket.

## License

Apache 2.0. Moonshine streaming models are MIT; Parakeet weights are CC BY 4.0. Phonon 31m weights are Apache 2.0; the Phonon 40m models are CC BY-NC 4.0. Smart Turn v3 weights are BSD 2-Clause (Daily).
