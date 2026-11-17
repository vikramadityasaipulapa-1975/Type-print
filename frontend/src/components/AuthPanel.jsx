import { useState } from 'react'
import { Fingerprint, LockKeyhole } from 'lucide-react'
import { api } from '../services/api.js'

export default function AuthPanel({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (mode === 'register') await api.register({ username, email, password })
      const session = await api.login({ identifier: mode === 'register' ? email : email || username, password })
      onAuthenticated(session.access_token)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return <main className="auth-shell">
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark"><Fingerprint size={19}/></span>TYPING FINGERPRINT</a><span className="phase-tag">BEHAVIORAL BIOMETRICS</span></header>
    <section className="auth-layout">
      <div className="auth-intro"><div className="eyebrow"><span className="pulse"/> ADDITIONAL VERIFICATION</div><h1>Your rhythm.<br/><em>Your identity.</em></h1><p>Conventional account authentication comes first. Typing dynamics add a second verification signal after enrollment.</p><div className="privacy-note"><LockKeyhole size={16}/><span>Biometric timing data is sensitive. You choose whether to enroll, and you can delete your profile.</span></div></div>
      <form className="auth-card" onSubmit={submit}>
        <div className="auth-tabs"><button type="button" className={mode === 'login' ? 'selected' : ''} onClick={() => { setMode('login'); setError('') }}>Sign in</button><button type="button" className={mode === 'register' ? 'selected' : ''} onClick={() => { setMode('register'); setError('') }}>Create account</button></div>
        <h2>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h2><p className="form-hint">{mode === 'login' ? 'Sign in to continue to your security dashboard.' : 'Set up a conventional account before biometric enrollment.'}</p>
        {mode === 'register' && <label className="form-field">Username<input required minLength="3" maxLength="48" pattern="[A-Za-z0-9_.-]+" value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label>}
        <label className="form-field">{mode === 'login' ? 'Email or username' : 'Email'}<input type={mode === 'login' ? 'text' : 'email'} required value={email} onChange={(event) => setEmail(event.target.value)} autoComplete={mode === 'register' ? 'email' : 'username'} /></label>
        <label className="form-field">Password<input type="password" required minLength={mode === 'register' ? 12 : 1} maxLength="128" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === 'register' ? 'new-password' : 'current-password'} /></label>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button auth-submit" type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
        <small className="form-security">Passwords are protected with Argon2. Sessions expire automatically.</small>
      </form>
    </section>
    <footer>LOCAL SECURITY CONSOLE <span>·</span> TYPING BIOMETRICS ARE AN ADDITIONAL FACTOR</footer>
  </main>
}
