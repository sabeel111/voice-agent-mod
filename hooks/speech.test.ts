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
  expect(describeToolCall('Bash', { command: 'npm   test' })).toBe('run the command: npm test')
  expect(describeToolCall('Edit', { file_path: '/repo/src/login.ts' })).toBe('edit login.ts')
  expect(describeToolCall('WebFetch', { url: 'https://docs.python.org/3/' })).toBe('fetch a page from docs.python.org')
  expect(describeToolCall('mcp__github__create_issue', {})).toBe('use create issue from github')
})

test('slugify makes agent names', () => {
  expect(slugify('Fix login redirect!')).toBe('fix-login-redirect')
  expect(slugify('???')).toBe('task')
})
