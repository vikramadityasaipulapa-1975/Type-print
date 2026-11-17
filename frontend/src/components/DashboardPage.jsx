import { useState } from 'react'
import { Activity, Fingerprint, ShieldCheck, Trash2 } from 'lucide-react'
import TypingPractice from './TypingPractice.jsx'
import { AuthenticationChart, TimingSequenceChart, TypingTrendChart } from './AnalyticsCharts.jsx'

function formatDate(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}

function MetricCard({ label, value, unit = '' }) {
  return <div className="metric-card"><span>{label}</span><strong>{value === null || value === undefined ? '—' : `${value}${unit}`}</strong></div>
}

function rounded(value) {
  return Number.isFinite(value) ? Math.round(value) : null
}

function HistoryTable({ history }) {
  if (!history.length) return <div className="empty-panel"><Fingerprint size={22}/><strong>No authentication attempts yet</strong><span>Results will appear here after an identity verification.</span></div>
  return <div className="table-scroll"><table><thead><tr><th>Date and time</th><th>Result</th><th>Similarity</th><th>Session</th></tr></thead><tbody>{history.map((item, index) => <tr key={`${item.session_id}-${index}`}><td>{formatDate(item.timestamp)}</td><td><span className={`result-badge ${item.authentication_status}`}>{item.authentication_status.replaceAll('_', ' ')}</span></td><td>{item.similarity_score === null ? '—' : `${Math.round(item.similarity_score)}%`}</td><td className="mono-cell">{item.session_id || '—'}</td></tr>)}</tbody></table></div>
}

export default function DashboardPage({
  view, user, dashboard, profile, history, timing, modelStatus, modelMetrics,
  loading, onOutcome, onDeleteProfile, onNavigate,
}) {
  const [deleteMessage, setDeleteMessage] = useState('')
  const latest = history[0]
  const mostRecentTiming = timing[0]
  const verification = modelMetrics?.verification?.test_verification
  const rfTest = modelMetrics?.random_forest?.test

  if (view === 'enroll') return <section className="workspace-page"><div className="page-heading"><div><div className="eyebrow"><span className="pulse"/> ENROLLMENT WORKFLOW</div><h2>Build your typing profile</h2><p>Type the same fixed passage naturally across separate samples. Each sample is encrypted and stored for your profile.</p></div><span className="capture-pill">{profile?.enrollment_samples || 0} / {profile?.minimum_samples || 5} SAMPLES</span></div><TypingPractice token={user.token} mode="enroll" onOutcome={onOutcome}/></section>

  if (view === 'verify') return <section className="workspace-page"><div className="page-heading"><div><div className="eyebrow"><span className="pulse"/> SECOND FACTOR</div><h2>Verify with your typing</h2><p>Sign-in remains the primary account check. This step compares a fresh passage against your enrolled typing profile.</p></div></div>{!profile?.profile_ready || !modelStatus?.verification_ready ? <div className="notice-card"><ShieldCheck size={19}/><div><strong>Verification is not ready yet</strong><p>{!profile?.profile_ready ? `Enroll at least ${profile?.minimum_samples || 5} samples. Current count: ${profile?.enrollment_samples || 0}.` : 'A trained LSTM model and held-out verification calibrator must be installed before verification can run.'}</p><button className="quiet-button" onClick={() => onNavigate('enroll')}>Go to enrollment</button></div></div> : <TypingPractice token={user.token} mode="authenticate" onOutcome={onOutcome}/>}</section>

  if (view === 'history') return <section className="workspace-page"><div className="page-heading"><div><div className="eyebrow"><span className="pulse"/> SESSION RECORD</div><h2>Authentication history</h2><p>Results returned by the verification API.</p></div></div><div className="panel"><div className="panel-title"><span>RECENT ATTEMPTS</span><small>{history.length} records</small></div><HistoryTable history={history}/></div></section>

  if (view === 'privacy') return <section className="workspace-page"><div className="page-heading"><div><div className="eyebrow"><span className="pulse"/> DATA CONTROL</div><h2>Biometric privacy</h2><p>Your account password remains the primary authentication factor. Typing patterns provide additional verification only.</p></div></div><div className="privacy-grid"><div className="panel"><div className="panel-title"><span>STORED PROFILE</span></div><p className="panel-copy">{profile?.enrolled ? `${profile.enrollment_samples} encrypted enrollment samples are stored for this account.` : 'No biometric enrollment samples are stored.'} Sample summaries and profile templates are encrypted at rest. The typed passage itself is not sent by the browser.</p><p className="panel-copy">Deleting the biometric profile also removes saved typing samples and authentication history for this account.</p>{deleteMessage && <p className="capture-message" role="status">{deleteMessage}</p>}<button className="danger-button" disabled={!profile?.enrolled} onClick={async () => { if (!window.confirm('Delete your encrypted biometric profile, saved samples, and authentication history?')) return; try { await onDeleteProfile(); setDeleteMessage('Biometric profile and saved typing records deleted.') } catch (error) { setDeleteMessage(error.message) } }}><Trash2 size={15}/> Delete biometric data</button></div><div className="panel"><div className="panel-title"><span>MODEL READINESS</span></div><dl className="readiness-list"><div><dt>LSTM checkpoint</dt><dd>{modelStatus?.lstm_available ? 'Available' : 'Not installed'}</dd></div><div><dt>Verification calibrator</dt><dd>{modelStatus?.verification_calibrator_available ? 'Available' : 'Not installed'}</dd></div><div><dt>Profile</dt><dd>{profile?.profile_ready ? 'Ready' : 'Incomplete'}</dd></div></dl></div></div></section>

  return <section className="workspace-page">
    <div className="page-heading"><div><div className="eyebrow"><span className="pulse"/> SECURITY OVERVIEW</div><h2>Good to see you, {user?.username}</h2><p>Typing dynamics are an additional verification signal for your account.</p></div><button className="quiet-button" onClick={() => onNavigate('verify')} disabled={!profile?.profile_ready || !modelStatus?.verification_ready}><ShieldCheck size={15}/> Start verification</button></div>
    {loading && <div className="loading-strip">Loading account analytics…</div>}
    <div className="overview-grid">
      <section className="identity-panel panel"><div className="panel-title"><span>IDENTITY MATCH</span><span className={`result-badge ${latest?.authentication_status || 'neutral'}`}>{latest ? latest.authentication_status.replaceAll('_', ' ') : 'NO ATTEMPTS'}</span></div><div className="identity-score">{latest?.similarity_score === null || latest?.similarity_score === undefined ? '—' : Math.round(latest.similarity_score)}<small>{latest?.similarity_score === null || latest?.similarity_score === undefined ? '' : '%'}</small></div><p>{latest ? `Last attempt · ${formatDate(latest.timestamp)}` : 'Your match score will appear after an actual verification attempt.'}</p><div className="identity-meter"><i style={{ width: `${Math.max(0, Math.min(100, latest?.similarity_score || 0))}%` }}/></div><div className="profile-foot"><span>PROFILE</span><strong>{profile?.profile_ready ? 'ENROLLED' : profile?.enrolled ? 'COLLECTING SAMPLES' : 'NOT ENROLLED'}</strong><button onClick={() => onNavigate('enroll')}>Manage <span>↗</span></button></div></section>
      <section className="panel timing-panel"><div className="panel-title"><span>KEYSTROKE TIMING</span><small>DWELL / FLIGHT · ENCRYPTED SAMPLE HISTORY</small></div><TimingSequenceChart sequence={mostRecentTiming?.sequence || []}/><div className="quick-stats"><MetricCard label="AVG DWELL" value={rounded(dashboard?.average_dwell_ms)} unit=" ms"/><MetricCard label="AVG FLIGHT" value={rounded(dashboard?.average_flight_ms)} unit=" ms"/><MetricCard label="PACE" value={rounded(dashboard?.typing_speed_cpm)} unit=" CPM"/></div></section>
    </div>
    <div className="metric-grid"><MetricCard label="TYPING CONSISTENCY" value={dashboard?.typing_consistency === null ? null : `${Math.round((dashboard?.typing_consistency || 0) * 100)}%`}/><MetricCard label="AUTH ATTEMPTS" value={dashboard?.authentication_attempts}/><MetricCard label="VERIFIED" value={dashboard?.successful_authentications}/><MetricCard label="NOT VERIFIED" value={dashboard?.failed_authentications}/></div>
    <div className="charts-grid"><section className="panel"><div className="panel-title"><span>TIMING TRENDS</span><small>REAL SAVED SAMPLES</small></div><TypingTrendChart sessions={timing}/></section><section className="panel"><div className="panel-title"><span>AUTHENTICATION OUTCOMES</span><small>RECENT API RESULTS</small></div><AuthenticationChart history={history}/></section></div>
    <div className="lower-grid"><section className="panel"><div className="panel-title"><span>AUTHENTICATION HISTORY</span><button className="text-button" onClick={() => onNavigate('history')}>View all <span>↗</span></button></div><HistoryTable history={history.slice(0, 5)}/></section><section className="panel stack-panel"><div className="panel-title"><span>MODEL PERFORMANCE</span><small>HELD-OUT EVALUATION</small></div><div className="performance-row"><span>Verification EER</span><strong>{verification?.eer == null ? '—' : `${(verification.eer * 100).toFixed(1)}%`}</strong></div><div className="performance-row"><span>Verification ROC-AUC</span><strong>{verification?.roc_auc == null ? '—' : `${(verification.roc_auc * 100).toFixed(1)}%`}</strong></div><div className="performance-row"><span>RF test accuracy</span><strong>{rfTest?.accuracy == null ? '—' : `${(rfTest.accuracy * 100).toFixed(1)}%`}</strong></div><div className="technology-list"><div className="panel-title"><span>TECHNOLOGY STACK</span></div>{['Python', 'PyTorch', 'Time Series', 'Biometrics'].map((item, index) => <div className={`technology-item ${index === (modelStatus?.lstm_available ? 1 : 0) ? 'active' : ''}`} key={item}><span>0{index + 1}</span><strong>{item}</strong><i>{index === 1 && modelStatus?.lstm_available ? 'ACTIVE' : ''}</i></div>)}</div></section></div>
  </section>
}
