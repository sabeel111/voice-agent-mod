import { expect, mock, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { On } from 'claude-code'

// Everything beneath the mod is played by the test: the voice helper process
// (its stdout events), its control socket (HTTP requests), agents and prompts.

type World = {
  fetches: { path: string; body: Record<string, unknown> }[]
  prompts: { text: string; asUser: boolean }[]
  spawns: { prompt: string; description: string; name?: string }[]
  argv: readonly string[][]
  send: (event: Record<string, unknown>) => void
  end: () => void
  askAnswer: string
}

function world(on: On): World {
  const queue: string[] = []
  let wake: (() => void) | undefined
  let isEnded = false
  const w: World = {
    fetches: [],
    prompts: [],
    spawns: [],
    argv: [],
    askAnswer: 'yes',
    send: event => {
      queue.push(`${JSON.stringify(event)}\n`)
      wake?.()
    },
    end: () => {
      isEnded = true
      wake?.()
    },
  }
  mock.env(on, { XDG_RUNTIME_DIR: '/run/user/1000', USER: 'me' })
  mock.clock(on, { now: 1_000_000 })
  on('command.register', () => ({ value: { command: 'talk' } }))
  on('tool.register', (_, e) => ({ value: { tool: `mcp__voice-agent__${e.name}` } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.log', () => ({ value: undefined }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('process.spawn', async function* (_, e) {
    ;(w.argv as string[][]).push([...e.argv])
    for (;;) {
      while (queue.length) yield { stream: 'stdout' as const, text: queue.shift() ?? '' }
      if (isEnded) return { value: { code: 0, signal: null } }
      await new Promise<void>(resolve => (wake = resolve))
    }
  })
  on('http.fetch', (_, e) => {
    const path = e.url.replace('http://voice', '')
    const body = JSON.parse(e.init?.body ?? '{}') as Record<string, unknown>
    w.fetches.push({ path, body })
    expect(e.init?.socketPath).toBe('/run/user/1000/claude-voice.sock')
    if (path === '/shutdown') w.end()
    const text = path === '/ask' ? JSON.stringify({ answer: w.askAnswer, heard: w.askAnswer }) : '{"ok":true}'
    return { value: { status: 200, ok: true, headers: {}, text } }
  })
  on('prompt.submit', (_, e) => {
    w.prompts.push({ text: e.text, asUser: e.origin.kind === 'plugin' && e.origin.asUser === true })
    return { text: e.text }
  })
  on('agent.spawn', (_, e) => {
    w.spawns.push({ prompt: e.prompt, description: e.description, name: e.name })
    return { model: 'test-model' }
  })
  on('agent.list', () => ({
    value: w.spawns.map((s, i) => ({ id: `agent-${i + 1}`, name: s.name, description: s.description, type: 'general-purpose', status: 'running' as const, spawnedBy: 'voice-agent' })),
  }))
  on('session.messages', () => ({
    value: [{ role: 'assistant' as const, text: '', toolUses: [{ tool_use_id: 'x', tool: 'Edit', input: { file_path: '/repo/src/settings.tsx' } }] }],
  }))
  on('turn.complete', (_, e) => ({ text: e.answer }))
  on('turn.start', (_, e) => ({ turnId: e.turnId }))
  return w
}

const sleep = (ms: number) =>
  new Promise<void>(resolve => (globalThis as unknown as { setTimeout: (f: () => void, ms: number) => void }).setTimeout(resolve, ms))

/** Runs `/talk <args>` the way the person typing it would. */
const talk = ($: Engine, args: string) => $.command.run({ command: 'talk', args } as Parameters<Engine['command']['run']>[0])

const until = async (check: () => boolean, what: string) => {
  for (let i = 0; i < 200; i++) {
    if (check()) return
    await sleep(5)
  }
  throw new Error(`timed out waiting for ${what}`)
}

async function voiceOn($: Engine, w: World) {
  await $.session.start({ cwd: '/repo', surface: 'terminal', isInteractive: true })
  const ran = await talk($, 'on')
  expect(ran.text).toContain('Voice is starting')
  w.send({ type: 'ready', socket: '/run/user/1000/claude-voice.sock', mode: 'wake', recorder: 'parecord', player: 'paplay' })
  w.send({ type: 'state', state: 'listening', mode: 'wake', muted: false, awake: false, asking: false, ptt: false })
}

test('/talk on starts the helper with uv and the socket', async ($, on) => {
  const w = world(on)
  await voiceOn($, w)
  await until(() => w.argv.length === 1, 'the helper to start')
  const argv = w.argv[0] ?? []
  expect(argv.slice(0, 4)).toEqual(['uv', 'run', '--quiet', '--script'])
  expect(argv[4]).toMatch(/helper\/voice_helper\.py$/)
  expect(argv).toContain('/run/user/1000/claude-voice.sock')
  const config = JSON.parse(argv.at(-1) ?? '{}') as Record<string, unknown>
  expect(config.wake_word).toBe('hey claude')
  expect(config.mode).toBe('wake')
  expect(config.stt_model).toBe('moonshine-medium')
  expect(config.stt_keyterms).toEqual([])
  const off = await talk($, 'off')
  expect(off.text).toBe('Voice is off.')
  expect(w.fetches.at(-1)?.path).toBe('/shutdown')
})

test('the speech settings reach the helper', { options: { sttModel: 'moonshine-small', sttKeyterms: ' Omarchy, Hyprland  config;; Kubernetes ' } }, async ($, on) => {
  const w = world(on)
  await voiceOn($, w)
  await until(() => w.argv.length === 1, 'the helper to start')
  const config = JSON.parse(w.argv[0]?.at(-1) ?? '{}') as Record<string, unknown>
  expect(config.stt_model).toBe('moonshine-small')
  expect(config.stt_keyterms).toEqual(['Omarchy', 'Hyprland config', 'Kubernetes'])
  await talk($, 'off')
})

test('what the person says becomes a prompt; spoken commands stay local', async ($, on) => {
  const w = world(on)
  await voiceOn($, w)
  w.send({ type: 'transcript', text: 'Can you add dark mode to the settings page?', source: 'wake' })
  await until(() => w.prompts.length === 1, 'the prompt')
  expect(w.prompts[0]).toEqual({ text: 'Can you add dark mode to the settings page?', asUser: true })

  w.send({ type: 'transcript', text: 'Switch to natural mode.', source: 'follow_up' })
  await until(() => w.fetches.some(f => f.path === '/mode'), 'the mode change')
  expect(w.fetches.find(f => f.path === '/mode')?.body).toEqual({ mode: 'natural' })
  w.send({ type: 'transcript', text: 'stop', source: 'voice' })
  await until(() => w.fetches.filter(f => f.path === '/stop').length >= 2, 'stop')
  expect(w.prompts.length).toBe(1)
  await talk($, 'off')
})

test('replies are spoken sentence by sentence while voice is on', async ($, on) => {
  const w = world(on)
  on('turn.step', async function* (_, e) {
    for (const text of ['Sure. Should the ', 'agent also add tests?', '\n```ts\ncode()\n```\nOkay.']) {
      yield { kind: 'text' as const, index: 0, text, ref: 0 }
    }
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] } as never
  })
  await voiceOn($, w)
  await until(() => w.fetches.length === 0 && w.argv.length === 1, 'start')
  // the helper said ready, so voice is on
  await sleep(20)
  const step = $.turn.step({ turnId: 't1', index: 0, model: 'm', messageCount: 1 })
  for await (const _ of step) {
    // drain
  }
  await until(() => w.fetches.filter(f => f.path === '/speak').length === 4, 'speech')
  expect(w.fetches.filter(f => f.path === '/speak').map(f => f.body.text)).toEqual([
    'Sure.',
    'Should the agent also add tests?',
    "I've put the code on screen.",
    'Okay.',
  ])
  await talk($, 'off')
})

test('delegate_task starts a background agent and its report comes back', async ($, on) => {
  const w = world(on)
  await $.session.start({ cwd: '/repo', surface: 'terminal', isInteractive: true })
  const started = await $.tool.call({
    tool: 'mcp__voice-agent__delegate_task',
    title: 'Add dark mode',
    brief: 'Add a dark mode toggle to the settings page, persisted in localStorage.',
  } as never)
  expect(String(started.result)).toContain('Started background task #1 "Add dark mode"')
  expect(w.spawns[0]?.name).toBe('add-dark-mode-1')
  expect(w.spawns[0]?.prompt).toContain('You are a background agent')

  const listed = await $.tool.call({ tool: 'mcp__voice-agent__list_tasks' } as never)
  expect(String(listed.result)).toContain('#1 "Add dark mode" (agent add-dark-mode-1): running')
  expect(String(listed.result)).toContain('Latest: Edit /repo/src/settings.tsx')

  await $.turn.complete({ answer: 'Added the toggle in settings.tsx; tests pass.', durationMs: 5, isAborted: false, turnId: 'a1', agentId: 'agent-1', reason: 'answer' })
  await until(() => w.prompts.length === 1, 'the report')
  expect(w.prompts[0]?.asUser).toBe(false)
  expect(w.prompts[0]?.text).toContain('Background task #1 "Add dark mode" finished')
  expect(w.prompts[0]?.text).toContain('Added the toggle in settings.tsx')

  const after = await $.tool.call({ tool: 'mcp__voice-agent__list_tasks' } as never)
  expect(String(after.result)).toContain('done')
})

test('a permission prompt is asked out loud and answered by voice', async ($, on) => {
  const w = world(on)
  on('tool.check', () => ({ decision: 'ask' as const }))
  await voiceOn($, w)
  await sleep(20)
  const yes = await $.tool.check({ tool: 'Bash', input: { command: 'npm test', description: 'Run the tests' }, tool_use_id: 'tu1' })
  expect(yes.decision).toBe('allow')
  expect(w.fetches.find(f => f.path === '/ask')?.body.question).toBe('I want to run the tests. Is that okay?')
  w.askAnswer = 'no'
  expect((await $.tool.check({ tool: 'Bash', input: { command: 'rm -rf build' }, tool_use_id: 'tu2' })).decision).toBe('deny')
  w.askAnswer = 'timeout'
  expect((await $.tool.check({ tool: 'Bash', input: { command: 'ls' }, tool_use_id: 'tu3' })).decision).toBe('ask')
  // screen-only questions are left to the screen
  const asked = w.fetches.filter(f => f.path === '/ask').length
  expect((await $.tool.check({ tool: 'AskUserQuestion', input: {}, tool_use_id: 'tu4' })).decision).toBe('ask')
  expect(w.fetches.filter(f => f.path === '/ask').length).toBe(asked)
  await talk($, 'off')
})

test('with voice off, permissions are untouched', async ($, on) => {
  const w = world(on)
  on('tool.check', () => ({ decision: 'ask' as const }))
  await $.session.start({ cwd: '/repo', surface: 'terminal', isInteractive: true })
  expect((await $.tool.check({ tool: 'Bash', input: { command: 'ls' }, tool_use_id: 'tu1' })).decision).toBe('ask')
  expect(w.fetches).toEqual([])
})

test('/talk keys prints Hyprland bindings', async ($, on) => {
  world(on)
  await $.session.start({ cwd: '/repo', surface: 'terminal', isInteractive: true })
  const { text } = await talk($, 'keys')
  expect(text).toContain('bind = SUPER, grave, exec, curl -s --unix-socket "$XDG_RUNTIME_DIR/claude-voice.sock" -X POST http://voice/ptt/down')
  expect(text).toContain('bindr = SUPER, grave, exec')
})

// Each act on a drawing waits for the mod's background work to settle, and the
// fake helper never exits while voice is on: give this test room.
test('the pane shows state, tasks and the conversation on every surface', { timeoutMs: 20_000 }, async ($, on) => {
  const w = world(on)
  await voiceOn($, w)
  w.send({ type: 'transcript', text: 'Add dark mode please', source: 'wake' })
  await until(() => w.prompts.length === 1, 'the prompt')
  await $.tool.call({ tool: 'mcp__voice-agent__delegate_task', title: 'Add dark mode', brief: 'Add a dark mode toggle to the settings page.' } as never)
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({
      plugin: 'voice-agent',
      surface,
      component: 'Pane',
      requestId: 'voice-agent',
      props: { title: 'Voice', isFocused: false, bodyColumns: 50, placement: 'dock', scroll: { offset: 0, bodyRows: 30 }, view: {} },
    })
    const texts = (await ui.findAll({ type: 'Text' })).map(t => t.text)
    expect(texts).toContain('Voice: listening · wake word')
    expect(texts).toContain('● #1 Add dark mode · 0s')
    expect(texts).toContain('You: Add dark mode please')
    expect(await ui.find({ key: 'mute' })).toBeDefined()
    await ui.unmount()
  }
  await talk($, 'off')
})
