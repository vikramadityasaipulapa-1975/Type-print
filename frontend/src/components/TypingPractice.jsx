import { useMemo, useState } from 'react'
import { Activity, RotateCcw, ShieldCheck } from 'lucide-react'
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture.js'
import { api } from '../services/api.js'

const PASSAGE = 'The quiet rhythm of focused work creates a pattern uniquely your own.'
const MAX_BARS = 18

function labelFor(code) {
  if (code.startsWith('Key')) return code.slice(3)
  if (code.startsWith('Digit')) return code.slice(5)
  if (code === 'Space') return 'SPC'
  if (code === 'Backspace') return 'BKSP'
  return code.slice(0, 4).toUpperCase()
}

function formatMs(value) {
  return value === null || !Number.isFinite(value) ? '—' : `${Math.round(value)} ms`
}

function resultLabel(mode, response) {
  if (mode === 'enroll') {
    if (response.profile_ready) return `Biometric profile ready with ${response.enrollment_samples} enrollment samples.`
    if (response.status === 'model_unavailable') return `${response.enrollment_samples} samples received. The LSTM model is not installed yet.`
    if (response.status === 'calibration_unavailable') return `${response.enrollment_samples} samples received. Verification calibration is not installed yet.`
    return `Sample encrypted and saved. ${response.samples_remaining} more enrollment sample${response.samples_remaining === 1 ? '' : 's'} needed.`
  }
  if (response.status === 'authenticated') return `Verified. Calibrated similarity: ${Math.round(response.similarity_score)}%.`
  if (response.status === 'rejected') return `Typing pattern did not match this profile. Calibrated similarity: ${Math.round(response.similarity_score)}%.`
  if (response.status === 'low_confidence') return `Low confidence: ${response.reason?.replaceAll('_', ' ') || 'try again'}.`
  return response.status.replaceAll('_', ' ')
}

export default function TypingPractice({ token, mode = 'enroll', onOutcome }) {
  const capture = useKeystrokeCapture()
  const [consented, setConsented] = useState(false)
  const [result, setResult] = useState('')
  const [complete, setComplete] = useState(false)
  const [typedLength, setTypedLength] = useState(0)
  const [busy, setBusy] = useState(false)
  const cpm = capture.stats.elapsed > 0 && typedLength > 0 ? Math.round((typedLength / capture.stats.elapsed) * 60_000) : null
  const maxDwell = useMemo(() => Math.max(1, ...capture.entries.slice(-MAX_BARS).map((entry) => entry.dwellMs || 0)), [capture.entries])

  function begin() {
    setResult('')
    setComplete(false)
    setTypedLength(0)
    capture.start()
    requestAnimationFrame(() => capture.inputRef.current?.focus())
  }

  async function finish() {
    const typed = capture.inputRef.current?.value ?? ''
    if (capture.interrupted) {
      setResult('Capture was interrupted by a missing key release. Reset and try again.')
      return
    }
    if (typed !== PASSAGE) {
      setResult('The passage does not match. Reset the sample and type it exactly as shown.')
      return
    }
    if (capture.entries.length < PASSAGE.length || capture.entries.some((entry) => !Number.isFinite(entry.dwellMs))) {
      setResult('The timing sample is incomplete. Reset and type the passage again using the keyboard.')
      return
    }
    capture.stop()
    if (capture.inputRef.current) capture.inputRef.current.value = ''
    setComplete(true)
    setBusy(true)
    setResult('Sending timing sample securely…')
    const payload = {
      sample_id: crypto.randomUUID(),
      session_id: crypto.randomUUID(),
      consent: true,
      passage_completed: true,
      events: capture.entries.map(({ id, code, downMs, upMs, dwellMs, flightMs }) => ({ id, code, downMs, upMs, dwellMs, flightMs })),
    }
    try {
      const response = mode === 'enroll' ? await api.enroll(token, payload) : await api.authenticate(token, payload)
      setResult(resultLabel(mode, response))
      onOutcome?.(response)
    } catch (error) {
      setResult(`Sample was not accepted: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    capture.reset()
    setComplete(false)
    setTypedLength(0)
    setResult('')
  }

  return <section className="practice-card" aria-labelledby="practice-title">
    <div className="practice-heading">
      <div><div className="eyebrow"><Activity size={13}/> {mode === 'enroll' ? 'BIOMETRIC ENROLLMENT' : 'IDENTITY VERIFICATION'}</div><h2 id="practice-title">Type the passage</h2></div>
      <span className={`capture-pill ${capture.active ? 'is-active' : ''}`}><i />{capture.active ? 'CAPTURING' : complete ? 'SAMPLE SENT' : 'IDLE'}</span>
    </div>
    <p className="passage">{PASSAGE}</p>
    <label className="consent-row"><input type="checkbox" checked={consented} onChange={(event) => { setConsented(event.target.checked); reset() }} /><span><ShieldCheck size={15}/> I consent to sending key categories and timing for this sample. The typed passage is not sent; biometric timing is encrypted at rest.</span></label>
    <label className="sr-only" htmlFor="typing-sample">Type the fixed passage here</label>
    <input ref={capture.inputRef} id="typing-sample" className="typing-input" type="text" autoComplete="off" autoCapitalize="off" spellCheck="false" placeholder={consented ? 'Select Start capture, then type the passage…' : 'Consent to enable the typing field'} disabled={!consented || !capture.active || complete} onInput={(event) => setTypedLength(event.currentTarget.value.length)} onKeyDown={capture.onKeyDown} onKeyUp={capture.onKeyUp} onBlur={capture.stop} onPaste={(event) => event.preventDefault()} aria-describedby="capture-message" />
    <div className="practice-actions">
      {!capture.active && !complete && <button className="primary-button" type="button" disabled={!consented} onClick={begin}>Start capture</button>}
      {capture.active && <button className="primary-button" type="button" disabled={busy} onMouseDown={(event) => event.preventDefault()} onClick={finish}>{busy ? 'Sending…' : 'Finish sample'}</button>}
      <button className="quiet-button" type="button" onClick={reset}><RotateCcw size={14}/> Reset</button>
      <span className="event-count">{capture.entries.length} / {PASSAGE.length} key events</span>
    </div>
    {result && <p id="capture-message" className={`capture-message ${result.startsWith('Verified.') || result.startsWith('Biometric profile ready') ? 'success' : ''}`} role="status">{result}</p>}
    {capture.interrupted && <p className="capture-message" role="alert">Capture stopped because focus left the typing field while a key was still held.</p>}
    <div className="timing-section">
      <div className="section-label">KEYSTROKE TIMING <span>DWELL / FLIGHT</span></div>
      <div className="timing-list" aria-live="polite">
        {capture.entries.slice(-MAX_BARS).map((entry) => <div className="timing-row" key={entry.id}><span className="key-chip">{labelFor(entry.code)}</span><div className="timing-track"><i style={{ width: `${Math.max(2, (entry.dwellMs / maxDwell) * 100)}%` }} /></div><span className="timing-value">{Math.round(entry.dwellMs)}<small>ms</small></span><span className="flight-value">{entry.flightMs === null ? '—' : `${Math.round(entry.flightMs)} ms`}</span></div>)}
        {capture.entries.length === 0 && <div className="empty-timing">Timing bars appear here as keys are released.</div>}
      </div>
      <div className="stats-grid"><div><small>AVG DWELL</small><strong>{formatMs(capture.stats.averageDwell)}</strong></div><div><small>AVG FLIGHT</small><strong>{formatMs(capture.stats.averageFlight)}</strong></div><div><small>PACE</small><strong>{cpm === null ? '—' : `${cpm} CPM`}</strong></div></div>
    </div>
  </section>
}
