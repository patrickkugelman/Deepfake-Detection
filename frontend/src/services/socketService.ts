/**
 * Service pentru gestionarea conexiunii de tip WebSocket cu Cortex.
 */
class SocketService {
  private socket: WebSocket | null = null;
  private url: string = 'ws://localhost:8000/ws';

  connect(
    onMessage: (data: any) => void, 
    onOpen?: () => void, 
    onClose?: () => void
  ) {
    if (this.socket) this.socket.close();

    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      if (onOpen) onOpen();
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (err) {
        console.error("Socket parse error:", err);
      }
    };

    this.socket.onclose = () => {
      if (onClose) onClose();
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  sendMessage(msg: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(msg));
    }
  }
}

export const socketService = new SocketService();