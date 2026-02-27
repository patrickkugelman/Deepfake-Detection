import React, { createContext, useState, useEffect, useContext } from 'react';
import { DEEPGUARD_API_URL } from '@/api';

interface ActiveJob {
  job_id: string;
  filename: string;
  progress: number;
}

interface AnalysisContextType {
  jobs: ActiveJob[];
  addJob: (job: ActiveJob) => void;
}

const AnalysisContext = createContext<AnalysisContextType | null>(null);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [jobs, setJobs] = useState<ActiveJob[]>([]);

  useEffect(() => {
    if (jobs.length === 0) return;
    const token = localStorage.getItem('token');
    if (!token) return;

    const interval = setInterval(async () => {
      for (const job of jobs) {
        try {
          const res = await fetch(`${DEEPGUARD_API_URL}/jobs/${job.job_id}`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          const data = await res.json();
          if (data.status === 'completed') {
            setJobs((prev) => prev.filter((j) => j.job_id !== job.job_id));
          } else {
            setJobs((prev) =>
              prev.map((j) =>
                j.job_id === job.job_id ? { ...j, progress: data.progress ?? j.progress } : j
              )
            );
          }
        } catch {
          // ignore
        }
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [jobs]);

  const addJob = (job: ActiveJob) => {
    setJobs((prev) => [...prev, job]);
  };

  return (
    <AnalysisContext.Provider value={{ jobs, addJob }}>
      {children}
      {jobs.length > 0 && (
        <div
          style={{
            position: 'fixed',
            bottom: 20,
            right: 20,
            background: '#3b82f6',
            color: 'white',
            padding: 16,
            borderRadius: 8,
            zIndex: 9999,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
          }}
        >
          <div className="font-bold text-sm">Analize active: {jobs.length}</div>
          {jobs.map((j) => (
            <div key={j.job_id} className="text-xs mt-1 opacity-90">
              {j.filename}: {j.progress}%
            </div>
          ))}
        </div>
      )}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysis must be used within AnalysisProvider');
  return ctx;
};
