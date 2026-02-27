import { api, AnalysisResponse, JobResponse, JobStatusResponse, DashboardStats } from "@/api";

/**
 * Service pentru gestionarea operațiunilor de analiză forensic.
 */
export const analysisService = {
  // Trimite un fișier spre analiză către Coordinator
  async uploadFile(file: File, query?: string): Promise<JobResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (query) formData.append('query', query);

    const { data } = await api.post<JobResponse>('/chat_analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return data;
  },

  // Verifică statusul unui job specific
  async getStatus(jobId: string): Promise<JobStatusResponse> {
    const { data } = await api.get<JobStatusResponse>(`/jobs/${jobId}`);
    return data;
  },

  // Preia statisticile pentru Dashboard
  async getStats(): Promise<DashboardStats> {
    const { data } = await api.get<DashboardStats>('/stats/dashboard');
    return data;
  },

  // Preia istoricul utilizatorului
  async getUserHistory(limit: number = 20) {
    const { data } = await api.get('/query_user_history', { params: { limit } });
    return data;
  }
};