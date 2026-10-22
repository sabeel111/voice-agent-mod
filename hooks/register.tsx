import { atom, read, update } from 'claude-code'
import type { HttpInit, PluginOptions, Register } from 'claude-code'

import type { LogLine, Task, VoiceMode, VoiceState, VoiceStatus } from '../types'
import { DELEGATE_DESCRIPTION, LIST_TASKS_DESCRIPTION, partnerPrompt, reportPrompt } from './prompts'
import {
  SCREEN_ONLY_TOOLS,
  SentenceSplitter,
  agentBrief,
  describeToolCall,
  formatElapsed,
  modeLabel,
  parseVoiceCommand,
  slugify,
} from './speech'

const PLUGIN = 'voice-agent'
const PANE = 'voice-agent'
const MODES: readonly VoiceMode[] = ['wake', 'natural', 'ptt']

const OFF: VoiceStatus = {
  isWanted: false,
  state: 'off',
  mode: 'wake',
  isMuted: false,
  isAwake: false,
  isAsking: false,
  isPtt: false,
  detail: '',
}

const status = atom({ plugin: 'voice-agent', key: 'status' } as const, OFF)
const log = atom({ plugin: 'voice-agent', key: 'log' } as const, [])
const tasks = atom({ plugin: 'voice-agent', key: 'tasks' } as const, [])

/** The running voice helper process, as this module knows it. */
type Helper = { socket: string; isStopping: boolean; isReady: boolean }

type HelperEvent =
  | { type: 'ready'; socket: string; mode: VoiceMode; recorder: string; player: string }
  | { type: 'state'; state: VoiceState | 'stopped'; mode: VoiceMode; muted: boolean; awake: boolean; asking: boolean; ptt: boolean }
  | { type: 'transcript'; text: string; source: string }
  | { type: 'barge_in' }
  | { type: 'wake' }
  | { type: 'log'; level: string; msg: string }
  | { type: 'error'; msg: string; fatal?: boolean }
  | { type: 'ping' }

const STATE_LABEL: Record<VoiceState, string> = {
  off: 'off',
  starting: 'starting',
  loading: 'loading models',
  listening: 'listening',
  hearing: 'hearing you',
  transcribing: 'thinking',
  speaking: 'speaking',
  muted: 'muted',
  idle: 'ready',
  error: 'error',
}

export const register: Register = (on, options) => {
  const opt = readOptions(options)

  // Module state: starts over when the mod reloads ($.state keeps the rest).
  let helper: Helper | undefined
  let socket = ''
  let speakChain: Promise<unknown> = Promise.resolve()
  let isTurnSilenced = false
  let nextLogId = 1

  const isVoiceOn = () => helper !== undefined && helper.isReady && !helper.isStopping

  /** Arguments for $.http.fetch: a POST to the voice helper's socket. */
  const rq = (path: string, body: unknown = {}): [string, HttpInit] => [
    `http://voice${path}`,
    {
      method: 'POST',
      socketPath: helper?.socket ?? socket,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
  ]

  const logLine = (who: LogLine['who'], text: string) => (lines: LogLine[]) => {
    const line: LogLine = { id: nextLogId++, who, text: text.trim().slice(0, 600) }
    return line.text ? [...lines, line].slice(-60) : lines
  }

  const patch = (p: Partial<VoiceStatus>) => (s: VoiceStatus) => ({ ...s, ...p })

  // ---------------------------------------------------------------- session

  on('session.start', async ($, e, next) => {
    const started = await next(e)
    socket = socketPathFor(await $.env.get('XDG_RUNTIME_DIR'), await $.env.get('USER'))
    await $.command.register({
      name: 'talk',
      description: 'Talk with Claude by voice: on, off, mode wake|natural|ptt, mute, unmute, stop, status, keys',
      argumentHint: '[on|off|mode <wake|natural|ptt>|mute|unmute|stop|status|keys|pane]',
      immediate: true,
    })
    await $.tool.register({
      name: 'delegate_task',
      description: DELEGATE_DESCRIPTION,
      isDeferred: false,
      inputSchema: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'A short name for the task, 2 to 6 words, e.g. "Fix login redirect".' },
          brief: {
            type: 'string',
            description: 'Complete instructions for the agent: goal, context, relevant files, constraints, and what done looks like.',
          },
        },
        required: ['title', 'brief'],
      },
    })
    await $.tool.register({
      name: 'list_tasks',
      description: LIST_TASKS_DESCRIPTION,
      isDeferred: false,
      inputSchema: { type: 'object', properties: {} },
    })
    const s = await read($, status)
    if (s.isWanted || opt.autoStart) {
      // Back after a reload of the mod, or on by default.
      void $.command.run({ command: 'talk', args: 'on' }).catch(() => undefined)
    } else if (s.state !== 'off') {
      await update($, status, () => ({ ...OFF, mode: s.mode }))
    }
    return started
  })

  on('session.end', async ($, e, next) => {
    const h = helper
    if (h) {
      h.isStopping = true
      helper = undefined
      await $.http.fetch(...rq('/shutdown')).catch(() => undefined)
    }
    return next(e)
  })

  // ---------------------------------------------------------------- /talk

  on('command.run', { command: 'talk' }, async ($, e) => {
    const [sub = '', arg = ''] = e.args.trim().toLowerCase().split(/\s+/)
    const showStatus = (s: VoiceStatus) => $.ui.status(s.state === 'off' ? undefined : statusLine(s))

    const turnOff = async (detail = '') => {
      const h = helper
      const s = await read($, status)
      showStatus(await update($, status, () => ({ ...OFF, mode: s.mode, detail })))
      if (!h) return 'Voice is off.'
      h.isStopping = true
      helper = undefined
      await $.http.fetch(...rq('/shutdown')).catch(() => undefined)
      return 'Voice is off.'
    }

    const turnOn = async () => {
      if (helper && !helper.isStopping) return 'Voice is already on.'
      void $.ui.open({ id: PANE, title: 'Voice' })
      const mode = (await read($, status)).mode ?? opt.mode
      const config = {
        mode,
        wake_word: opt.wakeWord,
        stt_model: opt.sttModel,
        stt_device: opt.sttDevice,
        stt_language: opt.language,
        tts_model: opt.ttsModel,
        tts_voice: opt.ttsVoice,
        tts_lang: opt.language,
        barge_in: opt.bargeIn,
        follow_up_s: opt.followUpSeconds,
        chimes: opt.chimes,
      }
      const prefix = opt.helperCommand
        ? opt.helperCommand.split(/\s+/).filter(Boolean)
        : ['uv', 'run', '--quiet', '--script', `${$.plugin.root}/helper/voice_helper.py`]
      const argv = [...prefix, '--socket', socket, '--config', JSON.stringify(config)]
      const mine: Helper = { socket, isStopping: false, isReady: false }
      helper = mine
      showStatus(await update($, status, patch({ isWanted: true, state: 'starting', mode, detail: '' })))

      // What the person said: a command for the mod, or a prompt for Claude.
      const onHeard = async (text: string) => {
        await update($, log, logLine('you', text))
        const command = parseVoiceCommand(text)
        if (!command) {
          // Something new said cuts off whatever is still being read out.
          await $.http.fetch(...rq('/stop')).catch(() => undefined)
          await $.prompt.submit({ text, asUser: true })
          return
        }
        if (command.kind === 'stop') {
          isTurnSilenced = true
          await $.http.fetch(...rq('/stop')).catch(() => undefined)
        } else if (command.kind === 'mute' || command.kind === 'unmute') {
          await $.http.fetch(...rq('/mute', { muted: command.kind === 'mute' })).catch(() => undefined)
        } else if (command.kind === 'mode') {
          await update($, status, patch({ mode: command.mode }))
          await $.http.fetch(...rq('/mode', { mode: command.mode })).catch(() => undefined)
          await $.http.fetch(...rq('/speak', { text: `Okay, ${modeLabel(command.mode)} mode.` })).catch(() => undefined)
        } else if (command.kind === 'off') {
          await turnOff()
        }
      }

      const onEvent = async (event: HelperEvent) => {
        switch (event.type) {
          case 'ready':
            mine.isReady = true
            await update($, log, logLine('system', `Voice on: ${modeLabel(event.mode)}.`))
            $.ui.toast(readyHint(event.mode, opt.wakeWord))
            return
          case 'state':
            if (event.state === 'stopped' || helper !== mine) return
            showStatus(
              await update(
                $,
                status,
                patch({
                  state: event.state,
                  mode: event.mode,
                  isMuted: event.muted,
                  isAwake: event.awake,
                  isAsking: event.asking,
                  isPtt: event.ptt,
                }),
              ),
            )
            return
          case 'transcript':
            await onHeard(event.text)
            return
          case 'barge_in':
            isTurnSilenced = true
            return
          case 'log':
            if (!mine.isReady) $.ui.log(`${PLUGIN}: ${event.msg}`)
            else $.ui.log(`${PLUGIN}: ${event.msg}`, { to: 'debug' })
            return
          case 'error':
            await update($, status, patch({ detail: event.msg }))
            $.ui.log(`${PLUGIN}: ${event.msg}`, { to: event.fatal ? 'transcript' : 'debug' })
            return
          case 'wake':
          case 'ping':
            return
        }
      }

      // The helper lives as long as this loop; the mod unloading ends both.
      void (async () => {
        let stderr = ''
        let lastError = ''
        let pending = ''
        try {
          const child = $.process.spawn({ argv, env: { PYTHONUNBUFFERED: '1' } })
          for await (const piece of child) {
            if (piece.stream === 'stderr') {
              stderr = (stderr + piece.text).slice(-4000)
              continue
            }
            pending += piece.text
            for (let nl = pending.indexOf('\n'); nl >= 0; nl = pending.indexOf('\n')) {
              const event = parseEvent(pending.slice(0, nl))
              pending = pending.slice(nl + 1)
              if (!event) continue
              if (event.type === 'error') lastError = event.msg
              try {
                await onEvent(event)
              } catch (err) {
                $.ui.log(`${PLUGIN}: ${event.type} event failed: ${String(err)}`, { to: 'debug' })
              }
            }
          }
        } catch (err) {
          lastError = `could not start the voice helper: ${String(err)}${opt.helperCommand ? '' : ' (is uv installed? sudo pacman -S uv)'}`
        }
        if (helper !== mine) return
        helper = undefined
        if (mine.isStopping) return
        const why = lastError || stderr.trim().split('\n').slice(-3).join(' ') || 'the voice helper stopped'
        const s = await read($, status)
        showStatus(await update($, status, () => ({ ...OFF, mode: s.mode, state: 'error', detail: why })))
        $.ui.toast(`Voice stopped: ${why.slice(0, 160)}`)
        $.ui.log(`${PLUGIN}: voice stopped: ${why}`)
      })()

      return `Voice is starting (${modeLabel(mode)}). The first start downloads the speech models, which can take a minute.`
    }

    switch (sub) {
      case '':
        return { text: helper ? await turnOff() : await turnOn() }
      case 'on':
      case 'start':
        return { text: await turnOn() }
      case 'off':
        return { text: await turnOff() }
      case 'mode': {
        const mode = MODES.find(m => m === arg) ?? (arg === 'push-to-talk' ? 'ptt' : undefined)
        if (!mode) return { text: 'Usage: /talk mode wake|natural|ptt' }
        await update($, status, patch({ mode }))
        if (helper) await $.http.fetch(...rq('/mode', { mode })).catch(() => undefined)
        return { text: `Voice mode: ${modeLabel(mode)}.` }
      }
      case 'mute':
      case 'unmute':
        if (!helper) return { text: 'Voice is off. Turn it on with /talk on.' }
        await $.http.fetch(...rq('/mute', { muted: sub === 'mute' })).catch(() => undefined)
        return { text: sub === 'mute' ? 'Microphone muted (push-to-talk still works).' : 'Listening again.' }
      case 'stop':
      case 'quiet':
        isTurnSilenced = true
        if (helper) await $.http.fetch(...rq('/stop')).catch(() => undefined)
        return { text: 'Stopped talking.' }
      case 'status': {
        const s = await read($, status)
        const list = await read($, tasks)
        return { text: statusReport(s, list, socket) }
      }
      case 'keys':
        return { text: keysHelp(socket) }
      case 'pane':
        await $.ui.open({ id: PANE, title: 'Voice' })
        return { text: 'Voice pane opened.' }
      default:
        return { text: 'Usage: /talk [on|off|mode <wake|natural|ptt>|mute|unmute|stop|status|keys|pane]' }
    }
  })

  // ---------------------------------------------------------------- the partner

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    if (!isVoiceOn()) return composed
    const s = await read($, status)
    return {
      ...composed,
      sections: [...composed.sections, { id: `${PLUGIN}:partner`, text: partnerPrompt(s.mode, opt.wakeWord), scope: 'session' as const }],
    }
  })

  on('turn.start', ($, e, next) => {
    if (e.agentId === undefined) isTurnSilenced = false
    return next(e)
  })

  // Speak the main conversation's replies sentence by sentence as they stream.
  on('turn.step', async function* ($, e, next) {
    if (e.agentId !== undefined || !isVoiceOn()) return yield* next(e)
    const step = next({
      ...e,
      ...(opt.partnerModel ? { model: opt.partnerModel } : {}),
      ...(opt.partnerEffort ? { effort: opt.partnerEffort } : {}),
    })
    const splitter = new SentenceSplitter()
    const say = (text: string) => {
      if (isTurnSilenced || !isVoiceOn()) return
      speakChain = speakChain.then(() => $.http.fetch(...rq('/speak', { text }))).catch(() => undefined)
    }
    for await (const chunk of step) {
      yield chunk
      if (chunk.kind === 'text') splitter.push(chunk.text).forEach(say)
    }
    splitter.flush().forEach(say)
    return await step.result
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    if (e.agentId === undefined) {
      if (isVoiceOn() && e.answer) await update($, log, logLine('claude', e.answer))
      return done
    }
    const list = await read($, tasks)
    const task = list.find(t => t.agentId === e.agentId)
    if (!task) return done
    const ok = e.reason === 'answer'
    const finished: Task = {
      ...task,
      status: ok ? 'done' : 'failed',
      endedAt: await $.clock.now(),
      summary: e.answer.trim().slice(0, 4000),
    }
    await update($, tasks, all => all.map(t => (t.id === task.id ? finished : t)))
    await update($, log, logLine('agent', `#${task.id} ${task.title}: ${ok ? 'done' : 'stopped'}`))
    $.ui.toast(`Task #${task.id} ${ok ? 'done' : 'stopped'}: ${task.title}`)
    void $.prompt.submit({ text: reportPrompt(finished, e.answer, e.reason) }).catch(() => undefined)
    return done
  })

  // ---------------------------------------------------------------- tools

  on('tool.call', { tool: 'mcp__voice-agent__delegate_task' }, async ($, e) => {
    const input = e as unknown as { title?: unknown; brief?: unknown }
    const title = typeof input.title === 'string' && input.title.trim() ? input.title.trim().slice(0, 80) : 'Task'
    const brief = typeof input.brief === 'string' ? input.brief.trim() : ''
    if (brief.length < 20) {
      return { result: 'The brief is too short. Give the agent complete, self-contained instructions.' }
    }
    const list = await read($, tasks)
    const id = (list.at(-1)?.id ?? 0) + 1
    const name = `${slugify(title)}-${id}`
    const spawned = await $.agent.spawn({
      prompt: agentBrief(brief),
      description: title,
      name,
      ...(opt.agentType ? { subagentType: opt.agentType } : {}),
      ...(opt.agentModel ? { model: opt.agentModel } : {}),
    })
    if (spawned.deny !== undefined) return { result: `The agent could not start: ${spawned.deny}` }
    // The id comes with the answer; the session's list knows it by name too.
    const agentId = spawned.agentId ?? (await $.agent.list()).find(a => a.name === name)?.id
    if (agentId === undefined) return { result: 'The agent could not start: no agent was started.' }
    const task: Task = { id, agentId, name, title, status: 'running', startedAt: await $.clock.now() }
    await update($, tasks, all => [...all, task].slice(-30))
    await update($, log, logLine('agent', `#${id} ${title}: started`))
    return {
      result: `Started background task #${id} "${title}" (agent name: ${name}). It works on its own; a message from the voice-agent plugin will bring you its report when it finishes. Tell the user it's underway and carry on.`,
    }
  })

  on('tool.call', { tool: 'mcp__voice-agent__list_tasks' }, async $ => {
    const list = await read($, tasks)
    if (list.length === 0) return { result: 'No background tasks yet.' }
    const now = await $.clock.now()
    const rows: string[] = []
    for (const t of list) {
      const age = formatElapsed((t.endedAt ?? now) - t.startedAt)
      const head = `#${t.id} "${t.title}" (agent ${t.name}): ${t.status}, ${t.status === 'running' ? `running for ${age}` : `took ${age}`}`
      if (t.status !== 'running') {
        rows.push(`${head}\n  Report: ${(t.summary ?? '').slice(0, 1200) || '(none)'}`)
        continue
      }
      const found = await $.session.messages({ agentId: t.agentId })
      rows.push(`${head}\n  Latest: ${Array.isArray(found) ? latestActivity(found) : 'unknown'}`)
    }
    return { result: rows.join('\n') }
  })

  // ---------------------------------------------------------------- permissions by voice

  on('tool.check', async ($, e, next) => {
    const verdict = await next(e)
    if (verdict.decision !== 'ask' || !opt.voicePermissions || !isVoiceOn() || e.tool_use_id === undefined) return verdict
    if (SCREEN_ONLY_TOOLS.has(e.tool)) return verdict
    const task = e.agentId === undefined ? undefined : (await read($, tasks)).find(t => t.agentId === e.agentId)
    const who = task ? `The ${task.title} agent wants` : e.agentId === undefined ? 'I want' : 'An agent wants'
    const question = `${who} to ${describeToolCall(e.tool, e.input)}. Is that okay?`
    const res = await $.http.fetch(...rq('/ask', { question, timeout: opt.askTimeoutSeconds })).catch(() => undefined)
    if (!res?.ok) return verdict
    const answer = parseAnswer(res.text)
    await update($, log, logLine('system', `${question} ${answer === 'yes' ? 'Allowed.' : answer === 'no' ? 'Denied.' : 'No answer.'}`))
    if (answer === 'yes') return { decision: 'allow' as const, reason: 'The user allowed it by voice.' }
    if (answer === 'no') return { decision: 'deny' as const, reason: 'The user said no by voice.' }
    return verdict
  })

  // ---------------------------------------------------------------- pane

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Button, Text } = $.ui.resolve(e)
    const s = await read($, status)
    const list = await read($, tasks)
    const lines = await read($, log)
    const now = await $.clock.now()
    const room = Math.max(3, (e.viewport?.rows ?? 30) - 8 - Math.min(list.length, 6))
    const nextMode = MODES[(MODES.indexOf(s.mode) + 1) % MODES.length] ?? 'wake'
    const isOn = s.state !== 'off' && s.state !== 'error'

    return (
      <Box flexDirection="column">
        <Text bold>{statusLine(s)}</Text>
        {s.detail !== '' && <Text dimColor>{s.detail}</Text>}
        <Box flexDirection="row" gap={1}>
          <Button key="power" label={isOn ? 'Voice off' : 'Voice on'} onPress={() => $.command.run({ command: 'talk', args: isOn ? 'off' : 'on' })} />
          <Button key="mode" label={`Mode: ${modeLabel(nextMode)}`} onPress={() => $.command.run({ command: 'talk', args: `mode ${nextMode}` })} />
          {isOn && <Button key="mute" label={s.isMuted ? 'Unmute' : 'Mute'} onPress={() => $.http.fetch(...rq('/mute', { muted: !s.isMuted }))} />}
          {s.state === 'speaking' && <Button key="hush" label="Stop talking" onPress={() => $.http.fetch(...rq('/stop'))} />}
        </Box>
        {list.length > 0 && <Text bold>Tasks</Text>}
        {list.slice(-6).map(t => (
          <Text dimColor={t.status !== 'running'}>
            {t.status === 'running' ? '● ' : t.status === 'done' ? '✓ ' : '✗ '}#{t.id} {t.title} · {formatElapsed((t.endedAt ?? now) - t.startedAt)}
          </Text>
        ))}
        <Text bold>Conversation</Text>
        {lines.length === 0 && <Text dimColor>{isOn ? readyHint(s.mode, opt.wakeWord) : 'Turn voice on to start talking.'}</Text>}
        {lines.slice(-room).map(l => (
          <Text dimColor={l.who === 'system' || l.who === 'agent'} wrap="truncate-end">
            {l.who === 'you' ? 'You: ' : l.who === 'claude' ? 'Claude: ' : l.who === 'agent' ? 'Agent ' : ''}
            {l.text.replace(/\s+/g, ' ')}
          </Text>
        ))}
      </Box>
    )
  })
}

// ---------------------------------------------------------------------------

type Options = {
  autoStart: boolean
  mode: VoiceMode
  wakeWord: string
  sttModel: string
  sttDevice: string
  language: string
  ttsModel: string
  ttsVoice: string
  bargeIn: boolean
  followUpSeconds: number
  chimes: boolean
  voicePermissions: boolean
  askTimeoutSeconds: number
  partnerModel: string
  partnerEffort: 'low' | 'medium' | 'high' | undefined
  agentModel: string
  agentType: string
  helperCommand: string
}

function readOptions(o: PluginOptions): Options {
  const str = (k: string, d = '') => (typeof o[k] === 'string' ? (o[k] as string).trim() : d)
  const bool = (k: string, d: boolean) => (typeof o[k] === 'boolean' ? (o[k] as boolean) : d)
  const num = (k: string, d: number) => (typeof o[k] === 'number' && Number.isFinite(o[k]) ? (o[k] as number) : d)
  const mode = str('mode', 'wake')
  const effort = str('partnerEffort', 'default')
  return {
    autoStart: bool('autoStart', false),
    mode: mode === 'natural' || mode === 'ptt' ? mode : 'wake',
    wakeWord: str('wakeWord') || 'hey claude',
    sttModel: str('sttModel') || 'parakeet-v2',
    sttDevice: str('sttDevice') || 'auto',
    language: str('language') || 'en',
    ttsModel: str('ttsModel'),
    ttsVoice: str('ttsVoice'),
    bargeIn: bool('bargeIn', true),
    followUpSeconds: num('followUpSeconds', 8),
    chimes: bool('chimes', true),
    voicePermissions: bool('voicePermissions', true),
    askTimeoutSeconds: num('askTimeoutSeconds', 30),
    partnerModel: str('partnerModel'),
    partnerEffort: effort === 'low' || effort === 'medium' || effort === 'high' ? effort : undefined,
    agentModel: str('agentModel'),
    agentType: str('agentType'),
    helperCommand: str('helperCommand'),
  }
}

function socketPathFor(runtimeDir: string | undefined, user: string | undefined): string {
  return runtimeDir ? `${runtimeDir}/claude-voice.sock` : `/tmp/claude-voice-${user ?? 'user'}.sock`
}

function parseEvent(line: string): HelperEvent | undefined {
  const text = line.trim()
  if (!text.startsWith('{')) return undefined
  try {
    const event = JSON.parse(text) as HelperEvent
    return typeof event?.type === 'string' ? event : undefined
  } catch {
    return undefined
  }
}

function parseAnswer(text: string): string {
  try {
    const answer = (JSON.parse(text) as { answer?: unknown }).answer
    return typeof answer === 'string' ? answer : 'timeout'
  } catch {
    return 'timeout'
  }
}

function statusLine(s: VoiceStatus): string {
  if (s.state === 'off') return 'Voice off'
  if (s.state === 'error') return 'Voice stopped'
  const extra = s.isAsking ? ' · waiting for yes or no' : s.isAwake ? ' · go ahead' : ''
  return `Voice: ${STATE_LABEL[s.state]} · ${modeLabel(s.mode)}${extra}`
}

function statusReport(s: VoiceStatus, list: Task[], socket: string): string {
  const running = list.filter(t => t.status === 'running').length
  return [
    `Voice: ${STATE_LABEL[s.state]}${s.state === 'off' ? '' : ` (${modeLabel(s.mode)}${s.isMuted ? ', muted' : ''})`}`,
    s.detail ? `Last notice: ${s.detail}` : '',
    `Tasks: ${running} running, ${list.length - running} finished`,
    `Control socket: ${socket}`,
  ]
    .filter(Boolean)
    .join('\n')
}

function readyHint(mode: VoiceMode, wakeWord: string): string {
  if (mode === 'wake') return `Voice on. Say "${wakeWord}" to talk.`
  if (mode === 'ptt') return 'Voice on. Hold your push-to-talk key to talk (/talk keys).'
  return 'Voice on. Just talk.'
}

export function keysHelp(socket: string): string {
  const path = /\/claude-voice\.sock$/.test(socket) && !socket.startsWith('/tmp/') ? '"$XDG_RUNTIME_DIR/claude-voice.sock"' : socket
  const curl = (what: string) => `curl -s --unix-socket ${path} -X POST http://voice/${what}`
  return [
    'Add to ~/.config/hypr/bindings.conf (Omarchy); Hyprland reloads it on save.',
    '',
    '# Hold Super+` to talk',
    `bind = SUPER, grave, exec, ${curl('ptt/down')}`,
    `bindr = SUPER, grave, exec, ${curl('ptt/up')}`,
    '',
    '# Or press once to start talking and again to send',
    `# bind = SUPER SHIFT, grave, exec, ${curl('ptt/toggle')}`,
    '',
    '# Make Claude stop talking',
    `bind = SUPER CTRL, grave, exec, ${curl('stop')}`,
  ].join('\n')
}

type AgentMessage = { role: 'user' | 'assistant'; text: string; toolUses: { tool: string; input: Record<string, unknown> }[] }

export function latestActivity(messages: readonly AgentMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m === undefined || m.role !== 'assistant') continue
    const tool = m.toolUses.at(-1)
    if (tool) {
      const target = ['file_path', 'command', 'pattern', 'url', 'query'].map(k => tool.input[k]).find(v => typeof v === 'string') as
        | string
        | undefined
      return `${tool.tool}${target ? ` ${target.replace(/\s+/g, ' ').slice(0, 100)}` : ''}`
    }
    if (m.text.trim()) return m.text.trim().replace(/\s+/g, ' ').slice(0, 200)
  }
  return 'starting up'
}
