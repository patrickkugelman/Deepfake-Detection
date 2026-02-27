import { useContext } from 'react';
// FIX: Importăm contextul corect din locația sa
import { WebSocketContext } from '@/context/WebSocketContext';

/**
 * Hook personalizat pentru a accesa mesajele live și starea de procesare de la Cortex.
 */
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  
  if (!context) {
    throw new Error('useWebSocket trebuie utilizat în interiorul unui WebSocketProvider');
  }
  
  return context;
};