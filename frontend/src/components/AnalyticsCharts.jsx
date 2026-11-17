import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const chartColor = '#59dce3'
const chartBlue = '#6388f5'

export function TimingSequenceChart({ sequence = [] }) {
  if (!sequence.length) return <div className="chart-empty">Complete a sample to see its measured timing sequence.</div>
  const data = sequence.slice(-24).map((point, index) => ({ ...point, label: `${index + 1} · ${point.key_family}` }))
  return <div className="chart-wrap timing-chart"><ResponsiveContainer width="100%" height={260}><BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 75 }} barGap={2}>
    <CartesianGrid stroke="#1d303b" strokeDasharray="3 3" horizontal={false}/><XAxis type="number" stroke="#71858f" tick={{ fontSize: 9 }} unit=" ms"/><YAxis type="category" dataKey="label" stroke="#71858f" tick={{ fontSize: 9 }} width={76}/><Tooltip contentStyle={{ background: '#0b141a', border: '1px solid #30434d', borderRadius: 8, fontSize: 11 }} formatter={(value) => [`${Math.round(value)} ms`]}/><Bar dataKey="dwell_ms" name="Dwell" fill={chartColor} radius={[0, 4, 4, 0]} barSize={7}/><Bar dataKey="flight_ms" name="Flight" fill={chartBlue} radius={[0, 4, 4, 0]} barSize={7}/>
  </BarChart></ResponsiveContainer></div>
}

export function TypingTrendChart({ sessions = [] }) {
  const data = [...sessions].reverse().filter((item) => item.average_dwell_ms !== null || item.average_flight_ms !== null).map((item, index) => ({
    sample: `${index + 1}`,
    dwell: item.average_dwell_ms,
    flight: item.average_flight_ms,
  }))
  if (!data.length) return <div className="chart-empty">Timing trends will appear after samples are saved.</div>
  return <div className="chart-wrap"><ResponsiveContainer width="100%" height={205}><BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
    <CartesianGrid stroke="#1d303b" strokeDasharray="3 3" vertical={false}/><XAxis dataKey="sample" stroke="#71858f" tick={{ fontSize: 9 }}/><YAxis stroke="#71858f" tick={{ fontSize: 9 }}/><Tooltip contentStyle={{ background: '#0b141a', border: '1px solid #30434d', borderRadius: 8, fontSize: 11 }} formatter={(value) => [`${Math.round(value)} ms`]}/><Bar dataKey="dwell" name="Dwell" fill={chartColor} radius={[4, 4, 0, 0]}/><Bar dataKey="flight" name="Flight" fill={chartBlue} radius={[4, 4, 0, 0]}/>
  </BarChart></ResponsiveContainer></div>
}

export function AuthenticationChart({ history = [] }) {
  const successful = history.filter((item) => item.authentication_status === 'authenticated').length
  const rejected = history.filter((item) => item.authentication_status === 'rejected').length
  const uncertain = history.filter((item) => !['authenticated', 'rejected'].includes(item.authentication_status)).length
  if (!history.length) return <div className="chart-empty">No authentication attempts recorded yet.</div>
  const data = [{ name: 'Recent attempts', successful, rejected, uncertain }]
  return <div className="chart-wrap"><ResponsiveContainer width="100%" height={205}><BarChart data={data} margin={{ top: 18, right: 8, left: -20, bottom: 0 }}>
    <CartesianGrid stroke="#1d303b" strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name" stroke="#71858f" tick={{ fontSize: 9 }}/><YAxis allowDecimals={false} stroke="#71858f" tick={{ fontSize: 9 }}/><Tooltip contentStyle={{ background: '#0b141a', border: '1px solid #30434d', borderRadius: 8, fontSize: 11 }}/><Bar dataKey="successful" name="Verified" stackId="a" fill="#5ee0bf"/><Bar dataKey="rejected" name="Rejected" stackId="a" fill="#f27d83"/><Bar dataKey="uncertain" name="Low confidence" stackId="a" fill="#e3b668" radius={[4, 4, 0, 0]}/>
  </BarChart></ResponsiveContainer></div>
}
