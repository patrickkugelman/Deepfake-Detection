import { useState, useEffect } from 'react';
import { useWebSocket } from '@/context/WebSocketContext';

export const useAnalysis = (jobId: string | null) => {
  const { messages } = useWebSocket();
  const [status, setStatus] = useState<'idle' | 'processing' | 'completed' | 'failed'>('idle');
  
  useEffect(() => {
    if (!jobId) return;
    setStatus('processing');

    // Verificăm dacă în fluxul de mesaje WebSocket a apărut "success" pentru acest proces
    const isFinished = messages.some(
      (m) => m.status === 'success' && m.message.toLowerCase().includes('finalizată')
    );

    if (isFinished) {
      setStatus('completed');
    }
  }, [messages, jobId]);

  return { status, messages };
};