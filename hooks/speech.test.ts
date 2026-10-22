import { describe, expect, test } from 'claude-code/testing'

import { SentenceSplitter, describeToolCall, parseVoiceCommand, slugify, speakable, toSpeech } from './speech'

const stream = (pieces: string[]) => {
  const s = new SentenceSplitter()
  const out = pieces.flatMap(p => s.push(p))
  return [...out, ...s.flush()]
}

describe('SentenceSplitter', () => {
  test('releases each sentence as soon as it ends', () => {
    const s = new SentenceSplitter()
    expect(s.push('Sure, I can do')).toEqual([])
    expect(s.push(' that. Want tests')).toEqual(['Sure, I can do that.'])
    expect(s.push(' too? ')).toEqual(['Want tests too?'])
    expect(s.flush()).toEqual([])
  })

  test('skips code blocks and says so once', () => {
    expect(stream(['Here it is:\n``', '`ts\nconst a = 1.\nconst b = 2;\n`', '``\nDone. ```\nmore()\n```'])).toEqual([
      'Here it is:',
      "I've put the code on screen.",
      'Done.',
    ])
  })

  test('cleans markdown out of lists and links', () => {
    expect(stream(['## Plan\n- **Fix** the [login](https://x.y) bug\n- Run `npm test`\n'])).toEqual([
      'Plan',
      'Fix the login bug',
      'Run npm test',
    ])
  })

  test('breaks a long run-on at a comma', () => {
    const long = 'I looked through the authentication module and the session store, '.repeat(3)
    const s = new SentenceSplitter()
    expect(s.push(long).length).toBe(1)
  })
})

test('toSpeech drops tables and URLs', () => {
  expect(toSpeech('| a | b |')).toBe('')
  expect(toSpeech('See https://example.com/x for more')).toBe('See a link for more')
  expect(speakable('One. Two!\n\nThree?')).toEqual(['One.', 'Two!', 'Three?'])
})

describe('parseVoiceCommand', () => {
  test('recognises local commands', () => {
    expect(parseVoiceCommand('Stop.')).toEqual({ kind: 'stop' })
    expect(parseVoiceCommand('okay, be quiet')).toEqual({ kind: 'stop' })
    expect(parseVoiceCommand('Stop listening')).toEqual({ kind: 'mute' })
    expect(parseVoiceCommand('switch to natural mode')).toEqual({ kind: 'mode', mode: 'natural' })
    expect(parseVoiceCommand('Switch to push to talk.')).toEqual({ kind: 'mode', mode: 'ptt' })
    expect(parseVoiceCommand('use wake word mode please')).toEqual({ kind: 'mode', mode: 'wake' })
    expect(parseVoiceCommand('turn off voice')).toEqual({ kind: 'off' })
  })

  test('leaves real requests for Claude', () => {
    expect(parseVoiceCommand('stop the dev server and restart it')).toBeUndefined()
    expect(parseVoiceCommand('switch to the main branch')).toBeUndefined()
    expect(parseVoiceCommand('what should we build next')).toBeUndefined()
  })
})

test('describeToolCall says what a permission is for', () => {
  expect(describeToolCall('Bash', { command: 'npm test', description: 'Run the unit tests.' })).toBe('run the unit tests')
  expect(describeToolCall('Edit', { file_path: '/repo/src/login.ts' })).toBe('edit login.ts')
  expect(describeToolCall('WebFetch', { url: 'https://docs.python.org/3/' })).toBe('fetch a page from docs.python.org')
  expect(describeToolCall('mcp__github__create_issue', {})).toBe('use create issue from github')
})

describe('describeToolCall never reads a shell command out', () => {
  test('uses the description, lowercased and without its period', () => {
    expect(describeToolCall('Bash', { command: 'grep -rn "Is that okay" hooks', description: 'Find how the voice permission question is worded' })).toBe(
      'find how the voice permission question is worded',
    )
    expect(describeToolCall('PowerShell', { command: 'Get-ChildItem', description: 'List files.' })).toBe('list files')
    expect(describeToolCall('Bash', { command: 'npm ci', description: 'NPM install' })).toBe('NPM install')
  })

  test('clips a long description', () => {
    const said = describeToolCall('Bash', { command: 'ls', description: `List ${'very '.repeat(40)}many files` })
    expect(said.length).toBeLessThanOrEqual(101)
    expect(said.endsWith('…')).toBe(true)
  })

  test('without a description, names only the program', () => {
    expect(describeToolCall('Bash', { command: 'git push --force origin main' })).toBe('run a git command')
    expect(describeToolCall('Bash', { command: 'cd /repo && FOO=1 sudo npm test -- --watch' })).toBe('run a npm command')
    expect(describeToolCall('Bash', { command: '/usr/bin/rm -rf build' })).toBe('run a rm command')
    expect(describeToolCall('Bash', { command: 'uv run pytest' })).toBe('run a uv command')
    expect(describeToolCall('Bash', { command: 'echo hi' })).toBe('run an echo command')
    expect(describeToolCall('PowerShell', { command: 'Get-ChildItem C:\\' })).toBe('run a Get-ChildItem command')
  })

  test('falls back to a shell command when the program is unclear', () => {
    expect(describeToolCall('Bash', {})).toBe('run a shell command')
    expect(describeToolCall('Bash', { command: '$(curl -s https://example.com/x.sh)' })).toBe('run a shell command')
    expect(describeToolCall('PowerShell', { command: '& $script' })).toBe('run a PowerShell command')
  })
})

test('slugify makes agent names', () => {
  expect(slugify('Fix login redirect!')).toBe('fix-login-redirect')
  expect(slugify('???')).toBe('task')
})
