import { useState, useEffect } from 'react';
import { AppSidebar } from '@/components/shared/AppSidebar';
import { SidebarProvider } from '@/components/ui/sidebar';
import {
  adminGetUsers, adminGetJobs, adminGetStats,
  adminDeleteUser, AdminUser, AdminJob, AdminStats
} from '@/api';
import { Users, BarChart3, FileText, Trash2, RefreshCw, Loader2, Shield, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';

type Tab = 'stats' | 'users' | 'jobs';

export default function Admin() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>('stats');
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [jobs, setJobs] = useState<AdminJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingUser, setDeletingUser] = useState<string | null>(null);

  // Redirect dacă nu e admin
  useEffect(() => {
    if (!loading && user && user.role !== 'admin') {
      navigate('/');
    }
  }, [user, loading, navigate]);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [s, u, j] = await Promise.all([adminGetStats(), adminGetUsers(), adminGetJobs()]);
      setStats(s);
      setUsers(u);
      setJobs(j);
    } catch (err) {
      console.error('Admin fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const handleDeleteUser = async (username: string) => {
    if (!confirm(`Ești sigur că vrei să ștergi userul "${username}"? Această acțiune nu poate fi anulată.`)) return;
    setDeletingUser(username);
    try {
      await adminDeleteUser(username);
      setUsers(prev => prev.filter(u => u.username !== username));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Eroare la ștergere';
      alert(msg);
    } finally {
      setDeletingUser(null);
    }
  };

  const getVerdictBadge = (verdict: string) => {
    switch ((verdict || '').toUpperCase()) {
      case 'FAKE': return <span className="px-2 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400">FAKE</span>;
      case 'REAL': return <span className="px-2 py-0.5 rounded text-xs font-bold bg-green-500/20 text-green-400">REAL</span>;
      case 'INCONCLUSIVE': return <span className="px-2 py-0.5 rounded text-xs font-bold bg-yellow-500/20 text-yellow-400">NECONCLUDENT</span>;
      default: return <span className="px-2 py-0.5 rounded text-xs bg-muted text-muted-foreground">—</span>;
    }
  };

  if (loading) {
    return (
      <SidebarProvider>
        <div className="min-h-screen flex w-full bg-background">
          <AppSidebar />
          <main className="flex-1 flex items-center justify-center">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </main>
        </div>
      </SidebarProvider>
    );
  }

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar />
        <main className="flex-1 p-6">
          <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Shield className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-foreground">Admin Panel</h1>
                  <p className="text-muted-foreground text-sm">Gestionează platforma DeepGuard Pro</p>
                </div>
              </div>
              <button onClick={fetchAll} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border hover:bg-accent transition-colors text-sm">
                <RefreshCw className="h-4 w-4" /> Reîncarcă
              </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 mb-6 bg-muted/30 rounded-lg p-1 w-fit">
              {([
                { id: 'stats', label: 'Statistici', icon: BarChart3 },
                { id: 'users', label: `Utilizatori (${users.length})`, icon: Users },
                { id: 'jobs', label: `Analize (${jobs.length})`, icon: FileText },
              ] as { id: Tab; label: string; icon: React.ElementType }[]).map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === tab.id ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                >
                  <tab.icon className="h-4 w-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* STATS TAB */}
            {activeTab === 'stats' && stats && (
              <div className="space-y-6">
                {/* KPI Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Total Analize', value: stats.total_jobs, color: 'text-blue-400', bg: 'bg-blue-500/10' },
                    { label: 'Utilizatori', value: stats.total_users, color: 'text-purple-400', bg: 'bg-purple-500/10' },
                    { label: 'FAKE Detectate', value: stats.fake_count, color: 'text-red-400', bg: 'bg-red-500/10' },
                    { label: 'REAL Confirmate', value: stats.real_count, color: 'text-green-400', bg: 'bg-green-500/10' },
                  ].map(card => (
                    <div key={card.label} className={`${card.bg} border border-border rounded-xl p-5`}>
                      <p className="text-muted-foreground text-sm mb-1">{card.label}</p>
                      <p className={`text-3xl font-bold ${card.color}`}>{card.value}</p>
                    </div>
                  ))}
                </div>

                {/* Secondary KPIs */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Completate', value: stats.completed, color: 'text-green-400' },
                    { label: 'În așteptare', value: stats.pending, color: 'text-yellow-400' },
                    { label: 'Eșuate', value: stats.failed, color: 'text-red-400' },
                    { label: 'Confidence Medie', value: `${stats.avg_confidence}%`, color: 'text-blue-400' },
                  ].map(card => (
                    <div key={card.label} className="border border-border rounded-xl p-5 bg-card">
                      <p className="text-muted-foreground text-sm mb-1">{card.label}</p>
                      <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
                    </div>
                  ))}
                </div>

                {/* Breakdown Bar */}
                {stats.total_jobs > 0 && (
                  <div className="border border-border rounded-xl p-5 bg-card">
                    <h3 className="font-medium text-foreground mb-4">Distribuție Verdicturi</h3>
                    <div className="flex rounded-full overflow-hidden h-6 gap-0.5">
                      {stats.fake_count > 0 && (
                        <div className="bg-red-500 flex items-center justify-center text-xs text-white font-bold" style={{ width: `${(stats.fake_count / stats.total_jobs) * 100}%` }}>
                          {Math.round((stats.fake_count / stats.total_jobs) * 100)}%
                        </div>
                      )}
                      {stats.real_count > 0 && (
                        <div className="bg-green-500 flex items-center justify-center text-xs text-white font-bold" style={{ width: `${(stats.real_count / stats.total_jobs) * 100}%` }}>
                          {Math.round((stats.real_count / stats.total_jobs) * 100)}%
                        </div>
                      )}
                      {stats.inconclusive_count > 0 && (
                        <div className="bg-yellow-500 flex items-center justify-center text-xs text-white font-bold" style={{ width: `${(stats.inconclusive_count / stats.total_jobs) * 100}%` }}>
                          {Math.round((stats.inconclusive_count / stats.total_jobs) * 100)}%
                        </div>
                      )}
                    </div>
                    <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> FAKE</span>
                      <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> REAL</span>
                      <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-yellow-500 inline-block" /> NECONCLUDENT</span>
                    </div>
                  </div>
                )}

                <p className="text-xs text-muted-foreground">Generat la: {new Date(stats.generated_at).toLocaleString('ro-RO')}</p>
              </div>
            )}

            {/* USERS TAB */}
            {activeTab === 'users' && (
              <div className="border border-border rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="text-left p-4 font-medium text-muted-foreground">Utilizator</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Email</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Rol</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Analize</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Înregistrat</th>
                      <th className="text-right p-4 font-medium text-muted-foreground">Acțiuni</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((u, idx) => (
                      <tr key={u.username} className={`border-t border-border ${idx % 2 === 0 ? '' : 'bg-muted/20'}`}>
                        <td className="p-4 font-medium text-foreground">{u.username}</td>
                        <td className="p-4 text-muted-foreground">{u.email || '—'}</td>
                        <td className="p-4">
                          {u.role === 'admin'
                            ? <span className="px-2 py-0.5 rounded text-xs font-bold bg-primary/20 text-primary border border-primary/30">Admin</span>
                            : <span className="px-2 py-0.5 rounded text-xs bg-muted text-muted-foreground">User</span>
                          }
                        </td>
                        <td className="p-4 text-muted-foreground">{u.total_jobs || 0}</td>
                        <td className="p-4 text-muted-foreground">{u.created_at ? new Date(u.created_at).toLocaleDateString('ro-RO') : '—'}</td>
                        <td className="p-4 text-right">
                          {u.username !== 'admin' && (
                            <button
                              onClick={() => handleDeleteUser(u.username)}
                              disabled={deletingUser === u.username}
                              className="flex items-center gap-1 ml-auto px-3 py-1.5 rounded-lg border border-red-500/30 hover:bg-red-500/10 text-red-400 transition-colors text-xs disabled:opacity-50"
                            >
                              {deletingUser === u.username ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                              Șterge
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {users.length === 0 && (
                  <div className="text-center py-10 text-muted-foreground">Niciun utilizator găsit</div>
                )}
              </div>
            )}

            {/* JOBS TAB */}
            {activeTab === 'jobs' && (
              <div className="border border-border rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="text-left p-4 font-medium text-muted-foreground">Fișier</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Utilizator</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Verdict</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Confidence</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Status</th>
                      <th className="text-left p-4 font-medium text-muted-foreground">Data</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job, idx) => (
                      <tr key={job.job_id} className={`border-t border-border cursor-pointer hover:bg-muted/30 ${idx % 2 === 0 ? '' : 'bg-muted/20'}`}
                        onClick={() => navigate(`/results/${job.job_id}`)}>
                        <td className="p-4 font-medium text-foreground max-w-[160px] truncate">{job.filename || '—'}</td>
                        <td className="p-4 text-muted-foreground">{job.username || '—'}</td>
                        <td className="p-4">{getVerdictBadge(job.verdict)}</td>
                        <td className="p-4 text-muted-foreground">{job.confidence ? `${Math.round(job.confidence * 100)}%` : '—'}</td>
                        <td className="p-4">
                          <span className={`text-xs font-medium ${job.status === 'completed' ? 'text-green-400' : job.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                            {job.status}
                          </span>
                        </td>
                        <td className="p-4 text-muted-foreground">{job.created_at ? new Date(job.created_at).toLocaleDateString('ro-RO') : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {jobs.length === 0 && (
                  <div className="text-center py-10 text-muted-foreground">Nicio analiză găsită</div>
                )}
              </div>
            )}

            {/* Warning banner */}
            <div className="mt-6 flex items-center gap-2 text-xs text-muted-foreground bg-yellow-500/5 border border-yellow-500/20 rounded-lg p-3">
              <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0" />
              <span>Acțiunile din Admin Panel sunt ireversibile. Ștergerea unui utilizator elimină și toate analizele sale.</span>
            </div>
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}
