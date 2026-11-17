import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const KEY_CODE = /^(Key[A-Z]|Digit[0-9]|Numpad[0-9]|Space|Backspace|Minus|Equal|BracketLeft|BracketRight|Backslash|Semicolon|Quote|Comma|Period|Slash|Backquote)$/

export function useKeystrokeCapture() {
  const inputRef = useRef(null)
  const heldKeys = useRef(new Map())
  const lastEntry = useRef(null)
  const sequence = useRef(0)
  const [active, setActive] = useState(false)
  const [interrupted, setInterrupted] = useState(false)
  const [entries, setEntries] = useState([])

  const start = useCallback(() => {
    heldKeys.current.clear()
    lastEntry.current = null
    sequence.current = 0
    setEntries([])
    setInterrupted(false)
    setActive(true)
  }, [])

  const onKeyDown = useCallback((event) => {
    if (!active || event.repeat || event.metaKey || event.ctrlKey || event.altKey || !KEY_CODE.test(event.code)) return
    if (heldKeys.current.has(event.code)) return
    const entry = {
      id: sequence.current++,
      code: event.code,
      downMs: performance.now(),
      upMs: null,
      dwellMs: null,
      flightMs: null,
      previous: lastEntry.current,
    }
    if (entry.previous?.upMs !== null && entry.previous?.upMs !== undefined) {
      entry.flightMs = entry.downMs - entry.previous.upMs
    }
    if (entry.previous && entry.previous.upMs === null) entry.previous.next = entry
    lastEntry.current = entry
    heldKeys.current.set(event.code, entry)
  }, [active])

  const onKeyUp = useCallback((event) => {
    const entry = heldKeys.current.get(event.code)
    if (!entry) return
    entry.upMs = performance.now()
    entry.dwellMs = entry.upMs - entry.downMs
    heldKeys.current.delete(event.code)
    if (entry.next) entry.next.flightMs = entry.next.downMs - entry.upMs
    setEntries((current) => [...current, { ...entry, previous: undefined, next: undefined }].sort((a, b) => a.id - b.id))
  }, [])

  const stop = useCallback(() => {
    if (heldKeys.current.size > 0) {
      heldKeys.current.clear()
      setInterrupted(true)
    }
    setActive(false)
  }, [])

  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') stop()
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => document.removeEventListener('visibilitychange', handleVisibility)
  }, [stop])

  const reset = useCallback(() => {
    heldKeys.current.clear()
    lastEntry.current = null
    sequence.current = 0
    setEntries([])
    setInterrupted(false)
    setActive(false)
    if (inputRef.current) inputRef.current.value = ''
  }, [])

  const stats = useMemo(() => {
    const dwell = entries.map((entry) => entry.dwellMs).filter(Number.isFinite)
    const flight = entries.map((entry) => entry.flightMs).filter(Number.isFinite)
    const average = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
    const elapsed = entries.length > 1 ? Math.max(...entries.map((entry) => entry.upMs)) - Math.min(...entries.map((entry) => entry.downMs)) : null
    return {
      averageDwell: average(dwell),
      averageFlight: average(flight),
      elapsed,
    }
  }, [entries])

  return { inputRef, active, interrupted, entries, stats, start, stop, reset, onKeyDown, onKeyUp }
}
