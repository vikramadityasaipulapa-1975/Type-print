import { Suspense, lazy, useEffect, useState } from 'react'
import { Activity, FileClock, Fingerprint, LayoutDashboard, LogOut, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import AuthPanel from './components/AuthPanel.jsx'
import { api } from './services/api.js'

const DashboardPage = lazy(() => import('./components/DashboardPage.jsx'))

const navigation = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'enroll', label: 'Enroll typing', icon: Activity },
  { id: 'verify', label: 'Verify identity', icon: ShieldCheck },
  { id: 'history', label: 'Authentication history', icon: FileClock },
  { id: 'privacy', label: 'Privacy controls', icon: SlidersHorizontal },
]

export default function App() {
  const [token, setToken] = useState('')
  const [view, setView] = useState('overview')
  const [user, setUser] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [profile, setProfile] = useState(null)
  const [history, setHistory] = useState([])
  const [timing, setTiming] = useState([])
  const [modelStatus, setModelStatus] = useState(null)
  const [modelMetrics, setModelMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const [apiError, setApiError] = useState('')

  useEffect(() => {
    if (!token) return undefined
    let active = true
    setLoading(true)
    setApiError('')
    Promise.all([
      api.me(token), api.dashboard(token), api.profile(token), api.history(token), api.timing(token), api.modelStatus(token), api.modelMetrics(token),
    ]).then(([currentUser, dashboardData, profileData, historyData, timingData, statusData, metricsData]) => {
      if (!active) return
      setUser(currentUser)
      setDashboard(dashboardData)
      setProfile(profileData)
      setHistory(historyData)
      setTiming(timingData)
      setModelStatus(statusData)
      setModelMetrics(metricsData)
    }).catch((error) => {
      if (!active) return
      setApiError(error.message)
      if (error.status === 401) {
        setToken('')
        setUser(null)
      }
    }).finally(() => {
      if (active) setLoading(false)
    })
    return () => { active = false }
  }, [token, refreshKey])

  async function signOut() {
    try { await api.logout(token) } catch { /* Session is cleared locally even if the API is offline. */ }
    setToken('')
    setUser(null)
    setDashboard(null)
    setProfile(null)
    setHistory([])
    setTiming([])
    setView('overview')
  }

  async function deleteBiometricData() {
    await api.deleteProfile(token)
    setRefreshKey((value) => value + 1)
  }

  if (!token) return <AuthPanel onAuthenticated={(value) => { setToken(value); setView('overview') }}/>

  return <main className="app-shell">
    <header className="topbar app-topbar">
      <a className="brand" href="/"><span className="brand-mark"><Fingerprint size={19}/></span>TYPING FINGERPRINT</a>
      <div className="account-area"><span className="api-state"><i className={apiError ? 'offline' : ''}/> {apiError ? 'API ISSUE' : 'SECURE SESSION'}</span><span className="account-name">{user?.username || 'Account'}</span><button className="icon-button" onClick={signOut} title="Sign out"><LogOut size={16}/><span>Sign out</span></button></div>
    </header>
    {apiError && <div className="api-error" role="alert">{apiError}</div>}
    <div className="app-layout">
      <aside className="side-nav"><div className="nav-caption">WORKSPACE</div>{navigation.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${view === id ? 'active' : ''}`} onClick={() => setView(id)}><Icon size={16}/><span>{label}</span></button>)}<div className="side-note"><div className="side-note-icon"><ShieldCheck size={15}/></div><strong>Second factor</strong><p>Typing rhythm supplements your password. It does not replace it.</p></div></aside>
      <Suspense fallback={<div className="loading-strip">Loading dashboard…</div>}><DashboardPage
        view={view}
        user={{ ...user, token }}
        dashboard={dashboard}
        profile={profile}
        history={history}
        timing={timing}
        modelStatus={modelStatus}
        modelMetrics={modelMetrics}
        loading={loading}
        onOutcome={() => setRefreshKey((value) => value + 1)}
        onDeleteProfile={deleteBiometricData}
        onNavigate={setView}
      /></Suspense>
    </div>
    <footer className="app-footer">TYPING FINGERPRINT <span>·</span> BEHAVIORAL BIOMETRICS <span>·</span> ADDITIONAL VERIFICATION ONLY</footer>
  </main>
}
