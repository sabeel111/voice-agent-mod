// Text shaping for the voice: what Claude writes becomes what the voice says,
// and what the person says becomes either a local command or a prompt.

import type { VoiceMode } from '../types'

const FENCE = '```'

/**
 * Cuts streamed reply text into sentences the voice can start on right away.
 * Code blocks are not read out: the first one in a reply becomes a short
 * spoken note, the rest are skipped.
 */
export class SentenceSplitter {
  private buf = ''
  private inFence = false
  private saidCodeNote = false

  push(text: string): string[] {
    this.buf += text
    const out: string[] = []
    for (;;) {
      if (this.inFence) {
        const end = this.buf.indexOf(FENCE)
        if (end < 0) {
          // keep a possible partial closing fence
          this.buf = this.buf.slice(-2)
          return out
        }
        this.buf = this.buf.slice(end + FENCE.length).replace(/^[^\n]*\n?/, '')
        this.inFence = false
        continue
      }
      const fence = this.buf.indexOf(FENCE)
      const head = fence < 0 ? this.buf : this.buf.slice(0, fence)
      const cut = lastBoundary(head)
      if (fence >= 0) {
        out.push(...speakable(head))
        if (!this.saidCodeNote) {
          out.push("I've put the code on screen.")
          this.saidCodeNote = true
        }
        this.buf = this.buf.slice(fence + FENCE.length)
        this.inFence = true
        continue
      }
      if (cut <= 0) {
        // Hold back a lone backtick or two: it may be the start of a fence.
        return out
      }
      out.push(...speakable(this.buf.slice(0, cut)))
      this.buf = this.buf.slice(cut)
      return out
    }
  }

  flush(): string[] {
    const rest = this.inFence ? '' : this.buf
    this.buf = ''
    this.inFence = false
    return speakable(rest)
  }
}

/** Index just past the last sentence or line end in `text`, or 0. */
function lastBoundary(text: string): number {
  let cut = 0
  const re = /[.!?]["')\]]*(?=\s)|\n/g
  for (let m = re.exec(text); m !== null; m = re.exec(text)) {
    cut = m.index + m[0].length
  }
  if (cut === 0 && text.length > 160) {
    // A long run-on: break at the last comma so the voice can start.
    const comma = text.lastIndexOf(', ')
    if (comma > 40) cut = comma + 1
  }
  // Never end inside a run of backticks that may become a fence.
  while (cut > 0 && text[cut - 1] === '`') cut -= 1
  return cut
}

/** Splits text into sentences and cleans each for speech, dropping empties. */
export function speakable(text: string): string[] {
  const pieces = text
    .split(/(?<=[.!?]["')\]]*)\s+|\n+/)
    .map(toSpeech)
    .filter(p => /[A-Za-z0-9]/.test(p))
  return pieces
}

/** Markdown and other screen-only text, turned into something sayable. */
export function toSpeech(line: string): string {
  let s = line.trim()
  if (/^\|.*\|$/.test(s) || /^[-|: ]+$/.test(s)) return '' // table rows and rules
  s = s
    .replace(/^#{1,6}\s+/, '') // headings
    .replace(/^>\s?/, '') // quotes
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?/, '') // list markers, checkboxes
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1') // images
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') // links
    .replace(/https?:\/\/\S+/g, 'a link')
    .replace(/`([^`]*)`/g, '$1') // inline code
    .replace(/(\*\*|__)(.+?)\1/g, '$2') // bold
    .replace(/(^|[\s(])[*_]([^*_\s][^*_]*?)[*_](?=[\s).,!?:;]|$)/g, '$1$2') // italics
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/[`*_#]+/g, '')
    .replace(/\s*->\s*/g, ' to ')
    .replace(/\s*=>\s*/g, ' then ')
    .replace(/\s+/g, ' ')
    .trim()
  return s
}

// ---------------------------------------------------------------------------
// Spoken commands handled by the mod itself, never sent to Claude.

export type VoiceCommand =
  | { kind: 'stop' }
  | { kind: 'mute' }
  | { kind: 'unmute' }
  | { kind: 'mode'; mode: VoiceMode }
  | { kind: 'off' }

const words = (text: string) =>
  text
    .toLowerCase()
    .replace(/[’']/g, "'")
    .replace(/[^a-z0-9' ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

export function parseVoiceCommand(text: string): VoiceCommand | undefined {
  const t = words(text).replace(/^(?:okay|ok|um|uh|please|claude)\s+/, '').replace(/\s+please$/, '')
  if (/^(?:stop|stop talking|stop it|be quiet|quiet|shush|shut up|enough|that's enough|thats enough|ok stop|okay stop|never mind|nevermind)$/.test(t)) {
    return { kind: 'stop' }
  }
  if (/^(?:stop listening|mute|mute yourself|mute the mic|go to sleep|pause listening)$/.test(t)) {
    return { kind: 'mute' }
  }
  if (/^(?:start listening|unmute|wake up|resume listening)$/.test(t)) {
    return { kind: 'unmute' }
  }
  if (/^(?:turn off voice|voice off|turn voice off|stop voice mode|exit voice mode)$/.test(t)) {
    return { kind: 'off' }
  }
  const mode = /^(?:switch to|go to|use|change to|set) (?:the )?(.+?)(?: mode)?$/.exec(t)?.[1]
  if (mode !== undefined) {
    if (/^(?:natural|conversation|conversational|always on|open|hands free|free)$/.test(mode)) return { kind: 'mode', mode: 'natural' }
    if (/^(?:wake word|wake|wakeword|hey claude)$/.test(mode)) return { kind: 'mode', mode: 'wake' }
    if (/^(?:push to talk|push talk|ptt|button|key)$/.test(mode)) return { kind: 'mode', mode: 'ptt' }
  }
  return undefined
}

export function modeLabel(mode: VoiceMode): string {
  return mode === 'natural' ? 'natural talk' : mode === 'wake' ? 'wake word' : 'push to talk'
}

// ---------------------------------------------------------------------------
// Permission questions, said out loud.

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n).trimEnd()}…` : s)
const base = (p: string) => p.split('/').filter(Boolean).at(-1) ?? p

/** The program a shell command runs, e.g. "git" for "cd repo && FOO=1 sudo git push". */
export function commandProgram(command: string): string | undefined {
  for (const segment of command.split(/&&|\|\||;|\n/)) {
    const words = segment.trim().replace(/^[({]\s*/, '').split(/\s+/).filter(Boolean)
    while (words.length > 0 && (/^[A-Za-z_][A-Za-z0-9_]*=/.test(words[0]!) || /^(?:sudo|env|time|nohup|exec|command)$/.test(words[0]!))) {
      words.shift()
    }
    const first = words[0]
    if (first === undefined || /^(?:cd|pushd|popd|export|set|source|\.)$/.test(first)) continue
    const name = base(first)
    return /^[A-Za-z][A-Za-z0-9._+-]{0,29}$/.test(name) ? name : undefined
  }
  return undefined
}

/** A shell command, said without reading it out: the agent's own description, or just the program. */
function describeCommand(description: string, command: string, shell: string): string {
  const d = description.replace(/\s+/g, ' ').trim().replace(/[.\s]+$/, '')
  // "Run the tests" reads as "wants to run the tests"; acronyms like "NPM" keep their case.
  if (d) return clip(/^[A-Z][a-z]/.test(d) ? d.charAt(0).toLowerCase() + d.slice(1) : d, 100)
  const program = commandProgram(command)
  if (program === undefined) return `run a ${shell} command`
  return `run ${/^[aeio]/i.test(program) ? 'an' : 'a'} ${program} command`
}

/** "find the failing test", "run a git command", "edit login.ts", ... Never reads a command out. */
export function describeToolCall(tool: string, input: unknown): string {
  const i = (input ?? {}) as Record<string, unknown>
  const str = (k: string) => (typeof i[k] === 'string' ? (i[k] as string) : '')
  switch (tool) {
    case 'Bash':
    case 'PowerShell':
      return describeCommand(str('description'), str('command'), tool === 'PowerShell' ? 'PowerShell' : 'shell')
    case 'Edit':
    case 'MultiEdit':
      return `edit ${base(str('file_path'))}`
    case 'Write':
      return `write the file ${base(str('file_path'))}`
    case 'NotebookEdit':
      return `edit the notebook ${base(str('notebook_path'))}`
    case 'Read':
      return `read ${base(str('file_path'))}`
    case 'WebFetch':
      try {
        return `fetch a page from ${new URL(str('url')).hostname}`
      } catch {
        return 'fetch a web page'
      }
    case 'WebSearch':
      return `search the web for ${clip(str('query'), 80)}`
    default: {
      const mcp = /^mcp__(.+?)__(.+)$/.exec(tool)
      if (mcp) return `use ${(mcp[2] ?? '').replace(/_/g, ' ')} from ${(mcp[1] ?? '').replace(/_/g, ' ')}`
      return `use the ${tool} tool`
    }
  }
}

/** Tools whose question only the person at the screen can answer. */
export const SCREEN_ONLY_TOOLS = new Set(['AskUserQuestion', 'ExitPlanMode', 'EnterPlanMode'])

// ---------------------------------------------------------------------------
// Agents

export function slugify(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32)
    .replace(/-+$/g, '')
  return slug || 'task'
}

export function agentBrief(brief: string): string {
  return `${brief.trim()}

---
You are a background agent. The user is talking with a voice assistant that handed you this task, and they can't see your work as it happens or answer questions mid-task. Work autonomously: make sensible decisions, follow the repository's conventions, and verify your work (build, tests, lint) where the project supports it.

When you finish, reply with a short report for the assistant to relay out loud:
- What you did, in two or three plain sentences.
- Files you changed.
- How you verified it, and anything that failed.
- Any decision the user still needs to make.`
}

export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}
