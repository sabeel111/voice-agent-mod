export type VoiceMode = 'natural' | 'wake' | 'ptt'

export type VoiceState =
  | 'off'
  | 'starting'
  | 'loading'
  | 'listening'
  | 'hearing'
  | 'transcribing'
  | 'speaking'
  | 'muted'
  | 'idle'
  | 'error'

export type VoiceStatus = {
  /** The person turned voice on (survives a reload of the mod). */
  isWanted: boolean
  state: VoiceState
  mode: VoiceMode
  isMuted: boolean
  /** Wake-word mode is listening without the wake word for a moment. */
  isAwake: boolean
  /** A spoken yes/no question is waiting for an answer. */
  isAsking: boolean
  isPtt: boolean
  /** The last error or notice worth showing. */
  detail: string
}

export type LogLine = {
  id: number
  who: 'you' | 'claude' | 'agent' | 'system'
  text: string
}

export type TaskStatus = 'running' | 'done' | 'failed'

export type Task = {
  id: number
  agentId: string
  name: string
  title: string
  status: TaskStatus
  startedAt: number
  endedAt?: number
  summary?: string
}

declare module 'claude-code' {
  interface PluginState {
    'voice-agent': {
      status: VoiceStatus
      log: LogLine[]
      tasks: Task[]
    }
  }
}
