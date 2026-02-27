/**
 * DeepGuard Frontend API Client
 * Centralizează comunicarea cu Backend-ul (Port 8000)
 * 
 * Features:
 * - Async Job Processing (Celery)
 * - Real-time Progress Tracking
 * - Gemini Vision Frame Analysis
 * - WebSocket Status Updates
 */

import axios, { AxiosError } from 'axios';

// ============================================================================
// CONFIGURATION
// ============================================================================

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
export const DEEPGUARD_API_URL = API_URL;

// Axios Instance
export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json'
    }
});

// JWT Token Interceptor
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface Evidence {
    type: "frame" | "heatmap" | "chart" | "alert" | "annotated_frame";
    title: string;
    url?: string;
    description: string;
    confidence?: number;
    timestamp?: number;
}

export interface GeminiFrameAnalysis {
    frame_number: number;
    timestamp: number;
    extracted_index: number;
    verdict: "REAL" | "FAKE" | "SUSPICIOUS";
    confidence: number;
    reasoning: string;
    artifacts_detected: string[];
    authenticity_indicators: string[];
}

export interface GeminiVisionResult {
    verdict: "REAL" | "FAKE" | "SUSPICIOUS";
    confidence: number;
    reasoning: string;
    frame_analyses?: GeminiFrameAnalysis[];
    annotated_frames?: string[];
    statistics?: {
        total_frames: number;
        fake_frames: number;
        suspicious_frames: number;
        real_frames: number;
    };
    artifacts_detected: string[];
}

export interface AnalysisResponse {
    verdict: "REAL" | "FAKE" | "SUSPICIOUS" | "INCONCLUSIVE";
    report?: string;
    reasoning: string;
    confidence: number;
    agents_used: string[];
    details: {
        [agentName: string]: any;
        gemini_vision?: GeminiVisionResult;
        synthid?: {
            detected: boolean;
            confidence: number;
            version?: string;
        };
    };
    cached?: boolean;
    metadata?: {
        filename?: string;
        file_type?: string;
        size?: string;
        resolution?: string;
        duration?: string;
        fps?: number;
        codec?: string;
    };
    evidence?: Evidence[];
    annotated_frames?: string[];
    timestamp?: string;
}

export interface JobResponse {
    job_id: string;
    status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed';
    message?: string;
}

export interface JobStatusResponse {
    status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed';
    progress: number;
    message?: string;
    result?: string | AnalysisResponse;
    final_verdict?: string;
    reasoning?: string;
    confidence?: number;
    created_at?: string;
    completed_at?: string;
    celery_state?: string;
    error?: string;
}

export interface HistoryRecord {
    job_id?: string;
    id?: string;
    file_hash?: string;
    filename: string;
    verdict: "REAL" | "FAKE" | "SUSPICIOUS" | "INCONCLUSIVE" | "PENDING" | "ERROR";
    confidence: number;
    timestamp?: string | number;
    status: string;
    progress?: number;
    url?: string;
    created_at?: string;
}

export interface UserHistoryResponse {
    records: HistoryRecord[];
    total: number;
}

export interface DashboardStats {
    totalAnalyses: number;
    todayAnalyses: number;
    weeklyTrend: number;
    deepfakesDetected: number;
    authenticVideos: number;
    inconclusiveResults: number;
    averageProcessingTime: number;
    weeklyData: { day: string; analyses: number; deepfakes: number }[];
}

export interface SystemStats {
    total: number;
    completed: number;
    failed: number;
    pending: number;
}

// ============================================================================
// AUTHENTICATION API
// ============================================================================

export interface AuthResponse {
    access_token: string;
    token_type: string;
}

export interface User {
    username: string;
    email?: string;
    disabled?: boolean;
    role?: string;
}

/**
 * Register new user
 */
export const register = async (username: string, password: string, email?: string): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/register', { username, password, email });
    return response.data;
};

/**
 * Login user
 */
export const login = async (username: string, password: string): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/login', { username, password });
    return response.data;
};

/**
 * Get current user info
 */
export const getMe = async (token: string): Promise<User> => {
    const response = await api.get<User>('/auth/me', {
        headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
};

// ============================================================================
// ANALYSIS & JOBS API (UPDATED FOR CELERY)
// ============================================================================

/**
 * Start new analysis (Async with Celery)
 * Returns job_id for polling
 */
export const analyzeFile = async (
    file: File,
    token: string,
    query?: string,
    forceRefresh: boolean = false
): Promise<JobResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (query) formData.append('query', query);
    if (forceRefresh) formData.append('force_refresh', 'true');

    const response = await axios.post<JobResponse>(`${API_URL}/chat_analyze`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `Bearer ${token}`
        }
    });
    return response.data;
};

/**
 * Poll job status with progress tracking
 * Call this every 2-3 seconds until status is 'completed' or 'failed'
 */
export const getJobStatus = async (jobId: string, token: string): Promise<JobStatusResponse> => {
    const response = await axios.get<JobStatusResponse>(`${API_URL}/jobs/${jobId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });

    // Parse JSON result if it's a string
    if (response.data.result && typeof response.data.result === 'string') {
        try {
            response.data.result = JSON.parse(response.data.result);
        } catch (e) {
            console.warn('Failed to parse result JSON:', e);
        }
    }

    return response.data;
};

/**
 * Chat with bot (Text + Optional File) - Starts a NEW Job
 */
export const chatWithBot = async (
    message: string,
    token: string,
    fileHash?: string,
    file?: File
): Promise<JobResponse> => {
    const formData = new FormData();
    formData.append('query', message);
    if (file) formData.append('file', file);
    if (fileHash) formData.append('file_hash', fileHash);

    const response = await axios.post<JobResponse>(`${API_URL}/chat_analyze`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `Bearer ${token}`
        }
    });
    return response.data;
};

export interface ChatEvidence {
    type: string;
    title: string;
    url?: string;
    description?: string;
    confidence?: number;
}

export interface ChatMessageResponse {
    role: string;
    content: string;
    evidence?: ChatEvidence[];
    error?: string;
}

/**
 * Send a chat message about an EXISTING job
 */
export const sendChatMessage = async (
    jobId: string,
    message: string,
    token: string
): Promise<ChatMessageResponse> => {
    const response = await api.post<ChatMessageResponse>('/chat/message', {
        job_id: jobId,
        message: message
    }, {
        headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
};

// ============================================================================
// STATS & HISTORY API
// ============================================================================

/**
 * Get user analysis history
 */
export const queryUserHistory = async (token: string, limit: number = 50): Promise<UserHistoryResponse> => {
    const response = await axios.get<UserHistoryResponse>(`${API_URL}/history?limit=${limit}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.data;
};

/**
 * Get dashboard statistics
 */
export const getDashboardStats = async (token: string): Promise<DashboardStats> => {
    try {
        const response = await axios.get<DashboardStats>(`${API_URL}/stats/dashboard`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        return response.data;
    } catch (error) {
        // Fallback to /stats if /stats/dashboard doesn't exist
        const statsResponse = await axios.get<SystemStats>(`${API_URL}/stats`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        // Convert SystemStats to DashboardStats
        return {
            totalAnalyses: statsResponse.data.total,
            todayAnalyses: 0,
            weeklyTrend: 0,
            deepfakesDetected: 0,
            authenticVideos: statsResponse.data.completed,
            inconclusiveResults: 0,
            averageProcessingTime: 0,
            weeklyData: []
        };
    }
};

/**
 * Get system statistics
 */
export const getSystemStats = async (token: string): Promise<SystemStats> => {
    const response = await axios.get<SystemStats>(`${API_URL}/stats`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.data;
};

/**
 * Get chat history
 */
export const getChatHistory = async (token: string) => {
    const response = await api.get('/chat/history', {
        headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
};

// ============================================================================
// FEEDBACK & SYSTEM
// ============================================================================

/**
 * Send feedback for model training
 */
export const sendFeedback = async (data: any, token: string): Promise<void> => {
    try {
        await api.post('/feedback', data, {
            headers: { Authorization: `Bearer ${token}` }
        });
    } catch (error) {
        console.error("Feedback failed:", error);
    }
};

/**
 * Health check
 */
export const checkHealth = async () => {
    try {
        const response = await axios.get(`${API_URL}/health`);
        return response.data;
    } catch {
        return { status: "offline", neo4j: "disconnected", redis: "disconnected", celery: "disabled" };
    }
};

// ============================================================================
// WEBSOCKET HELPER
// ============================================================================

/**
 * Create WebSocket connection for real-time updates
 * Usage:
 * 
 * const ws = createWebSocket((message) => {
 *   console.log('Status update:', message);
 *   if (message.job_id === myJobId) {
 *     setProgress(message.progress);
 *   }
 * });
 */
export const createWebSocket = (onMessage: (data: any) => void): WebSocket => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            onMessage(data);
        } catch (e) {
            console.error('WebSocket message parse error:', e);
        }
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
    };

    return ws;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Download annotated frame
 */
export const downloadAnnotatedFrame = async (frameUrl: string, filename: string) => {
    try {
        const response = await axios.get(frameUrl, { responseType: 'blob' });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.remove();
    } catch (error) {
        console.error('Download failed:', error);
    }
};

/**
 * Format verdict for display
 */
export const formatVerdict = (verdict: string): { text: string; color: string; emoji: string } => {
    switch (verdict.toUpperCase()) {
        case 'FAKE':
            return { text: 'DEEPFAKE DETECTAT', color: 'red', emoji: '🔴' };
        case 'SUSPICIOUS':
            return { text: 'SUSPECT', color: 'orange', emoji: '⚠️' };
        case 'REAL':
            return { text: 'AUTENTIC', color: 'green', emoji: '✅' };
        case 'PENDING':
        case 'QUEUED':
        case 'PROCESSING':
            return { text: 'ÎN PROCESARE', color: 'blue', emoji: '⏳' };
        case 'ERROR':
        case 'FAILED':
            return { text: 'EROARE', color: 'gray', emoji: '❌' };
        default:
            return { text: 'NECONCLUDENT', color: 'gray', emoji: '❓' };
    }
};

/**
 * Format confidence percentage
 */
export const formatConfidence = (confidence: number): string => {
    return `${Math.round(confidence * 100)}%`;
};

// ============================================================================
// ACTIVE JOBS API (pentru banner frontend)
// ============================================================================

export interface ActiveJob {
    job_id: string;
    status: string;
    progress: number;
    status_message: string;
    filename: string;
    created_at: string;
}

export const getMyActiveJobs = async (): Promise<ActiveJob[]> => {
    const response = await api.get<{ active_jobs: ActiveJob[] }>('/jobs/my-active');
    return response.data.active_jobs || [];
};

// ============================================================================
// ADMIN API
// ============================================================================

export interface AdminUser {
    username: string;
    email: string;
    role: string;
    created_at: string;
    total_jobs: number;
}

export interface AdminJob {
    job_id: string;
    status: string;
    progress: number;
    filename: string;
    created_at: string;
    completed_at: string;
    verdict: string;
    confidence: number;
    username: string;
}

export interface AdminStats {
    total_jobs: number;
    completed: number;
    failed: number;
    pending: number;
    fake_count: number;
    real_count: number;
    inconclusive_count: number;
    avg_confidence: number;
    total_users: number;
    generated_at: string;
}

export const adminGetUsers = async (): Promise<AdminUser[]> => {
    const response = await api.get<{ users: AdminUser[] }>('/admin/users');
    return response.data.users || [];
};

export const adminGetJobs = async (limit = 50, offset = 0): Promise<AdminJob[]> => {
    const response = await api.get<{ jobs: AdminJob[] }>(`/admin/jobs?limit=${limit}&offset=${offset}`);
    return response.data.jobs || [];
};

export const adminGetStats = async (): Promise<AdminStats> => {
    const response = await api.get<AdminStats>('/admin/stats');
    return response.data;
};

export const adminDeleteUser = async (username: string): Promise<void> => {
    await api.delete(`/admin/users/${username}`);
};

export default api;