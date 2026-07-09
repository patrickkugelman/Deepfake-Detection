import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { WS_URL, API_URL, getMyActiveJobs, ActiveJob } from '@/api';

interface WSMessage {
  type: string;
  status: 'info' | 'success' | 'warning' | 'error' | 'thinking' | 'processing';
  message: string;
  timestamp: string;
  job_id?: string;
  progress?: number;
}

export interface AnalysisState {
  isProcessing: boolean;
  progress: number;
  stage: string;
  logs: WSMessage[];
  jobId: string | null;
}

interface WebSocketContextType {
  messages: WSMessage[];
  isConnected: boolean;
  clearMessages: () => void;
  analysisState: AnalysisState;
  setAnalysisState: React.Dispatch<React.SetStateAction<AnalysisState>>;
  resetAnalysis: () => void;
  completedJobId: string | null;  // When analysis completes, for redirect
  clearCompletedJob: () => void;
  activeJobs: ActiveJob[];  // Jobs active on the account (e.g. started in another tab), from REST poll
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);

  const [analysisState, setAnalysisState] = useState<AnalysisState>(() => {
    const saved = sessionStorage.getItem('analysisState');
    return saved ? JSON.parse(saved) : {
      isProcessing: false,
      progress: 0,
      stage: 'Așteptare...',
      logs: [],
      jobId: null
    };
  });
  const [completedJobId, setCompletedJobId] = useState<string | null>(null);
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const tokenRef = useRef<string | null>(null);

  useEffect(() => {
    sessionStorage.setItem('analysisState', JSON.stringify(analysisState));
  }, [analysisState]);

  // Single account-wide poll for active jobs (e.g. started in another tab/session)
  useEffect(() => {
    const fetchActive = async () => {
      if (!localStorage.getItem('token')) return;
      try {
        const jobs = await getMyActiveJobs();
        setActiveJobs(jobs);
      } catch {
        // silently fail
      }
    };

    fetchActive();
    const interval = setInterval(fetchActive, 5000);
    return () => clearInterval(interval);
  }, []);

  // Background polling when job is in progress (persists across page navigation)
  const jobId = analysisState.jobId;
  const isProcessing = analysisState.isProcessing;
  useEffect(() => {
    if (!jobId || !isProcessing) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) return;

    const poll = async () => {
      try {
        const res = await fetch(`${API_URL}/jobs/${jobId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        const data = await res.json();
        if (data.status === 'completed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setAnalysisState(prev => ({ ...prev, isProcessing: false, progress: 100, stage: 'Complet!' }));
          setCompletedJobId(jobId);
        } else if (data.status === 'failed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setAnalysisState(prev => ({ ...prev, isProcessing: false, stage: 'Eroare la procesare' }));
        } else if (data.progress !== undefined) {
          setAnalysisState(prev => ({ ...prev, progress: data.progress, stage: data.message || prev.stage }));
        }
      } catch (e) {
        console.error('Background poll error:', e);
      }
    };

    pollIntervalRef.current = setInterval(poll, 3000);
    poll();
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [jobId, isProcessing]);

  const resetAnalysis = useCallback(() => {
    setAnalysisState({
      isProcessing: false,
      progress: 0,
      stage: 'Gata pentru o nouă analiză',
      logs: [],
      jobId: null
    });
    setMessages([]);
  }, []);

  const connect = useCallback(() => {
    // Clear existing socket
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      const wsUrl = WS_URL || 'ws://localhost:8000/ws';
      console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
      
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data: WSMessage = JSON.parse(event.data);
          
          // Ignore pong messages
          if (data.type === 'pong') return;
          
          console.log('📡 WebSocket message:', data);

          // Add to messages list
          setMessages((prev) => [...prev, data]);

          // Update analysis state based on message
          if (data.type === 'status' && data.job_id) {
            setAnalysisState(prev => {
              // Only update if this is our current job
              if (prev.jobId && prev.jobId !== data.job_id) {
                return prev;
              }

              let newProgress = data.progress ?? prev.progress;
              let isProc = true;

              // Map status to progress if not provided
              if (data.progress === undefined) {
                if (data.status === 'thinking') newProgress = 25;
                if (data.status === 'processing') newProgress = 65;
                if (data.status === 'success') {
                  newProgress = 100;
                  isProc = false;
                }
                if (data.status === 'error') isProc = false;
              } else {
                // Use explicit progress
                isProc = newProgress < 100;
              }

              return {
                ...prev,
                isProcessing: isProc,
                progress: newProgress,
                stage: data.message,
                logs: [...prev.logs, data],
                jobId: prev.jobId || data.job_id || null
              };
            });
          }
        } catch (error) {
          console.error('❌ WebSocket message parse error:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setIsConnected(false);
      };

      ws.onclose = () => {
        setIsConnected(false);
        socketRef.current = null;
        
        // Exponential backoff for reconnection
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectAttempts.current++;
        
        console.log(`🔌 WebSocket disconnected. Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})...`);
        
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      };

      socketRef.current = ws;
    } catch (error) {
      console.error('❌ WebSocket connection failed:', error);
      setIsConnected(false);
      
      // Retry connection
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
      reconnectAttempts.current++;
      reconnectTimeoutRef.current = setTimeout(connect, delay);
    }
  }, []);

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [connect]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setAnalysisState(prev => ({ ...prev, logs: [] }));
  }, []);

  const clearCompletedJob = useCallback(() => setCompletedJobId(null), []);

  return (
    <WebSocketContext.Provider value={{ 
      messages, 
      isConnected, 
      clearMessages, 
      analysisState, 
      setAnalysisState, 
      resetAnalysis,
      completedJobId,
      clearCompletedJob,
      activeJobs
    }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) throw new Error('useWebSocket must be used within a WebSocketProvider');
  return context;
};