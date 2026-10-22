# voice-agent

Talk with Claude Code out loud, like talking to a colleague. You discuss a task, Claude hands it to a background agent once you agree, and you keep chatting while the agent works. When the agent finishes, Claude tells you what happened.

All the speech runs on your own machine:

- **Hearing:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper, int8 on the CPU) with WebRTC voice detection.
- **Speaking:** [Phonon](https://gradium.ai/phonon) 31m by Gradium: open weights (Apache 2.0), about 65 MB, streams audio while it generates.
- **Audio:** PipeWire/PulseAudio (`parecord`/`paplay`), falling back to ALSA (`arecord`/`aplay`).

It is built and tested for Linux, Omarchy (Arch + Hyprland + PipeWire) in particular.

## How it works

```
 you ─► mic ─► voice detection ─► Whisper ─► "hey Claude, add dark mode"
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
- **Permissions by voice.** When Claude or an agent needs permission, you hear "The dark mode agent wants to run the command npm test. Is that okay?" Answer yes or no. If you don't answer, the normal permission prompt appears.

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

`uv` runs the voice helper and installs its Python dependencies (faster-whisper, ptts, webrtcvad, numpy) into its own cache on first start. Nothing is installed system-wide.

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

The first start downloads the Whisper model (about 145 MB for `base.en`) and Phonon 31m (about 65 MB) from Hugging Face, which takes a minute. Later starts take a few seconds. A **Voice** pane opens with the live state, your tasks and the conversation.

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
| Speech recognition model | `base.en` | `tiny.en` is fastest, `small.en` most accurate |
| Language | `en` | Other languages need a multilingual Phonon 40m model and a Whisper model without `.en` |
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

- **"could not start the voice helper"**: install `uv` (`sudo pacman -S uv`). Or point the **Voice helper command** setting at your own Python environment, e.g. `~/.venvs/voice/bin/python /path/to/helper/voice_helper.py`, after `pip install numpy faster-whisper ptts webrtcvad-wheels`.
- **"no recorder found"**: install `pipewire-pulse` (it provides `parecord`/`paplay`) or `alsa-utils`.
- **It doesn't react to "hey Claude"**: try `small.en`, speak a little slower, or set your own wake word. `/talk status` shows the last notice.
- **Model downloads fail**: they come from huggingface.co, so check that it's reachable. The 40m Phonon models also need `hf auth login`.
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
- `helper/voice_helper.py`: the local voice process: mic, voice detection, wake word, push-to-talk, Whisper, Phonon, barge-in and spoken yes/no questions, controlled over a Unix socket.

## License

Apache 2.0. Phonon 31m weights are Apache 2.0; the Phonon 40m models are CC BY-NC 4.0.
