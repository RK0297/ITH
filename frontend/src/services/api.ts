// API service for backend communication

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ChatMessage {
  query: string;
  conversation_id?: string;
  top_k?: number;
}

export interface ChatResponse {
  response: string;
  sources: Array<{
    title: string;
    source: string;
    category: string;
    url: string;
    preview: string;
  }>;
  conversation_id: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  vector_database_status: string;
  vector_database_count: number;
  ollama_status: string;
  timestamp: string;
}

export interface StatsResponse {
  total_documents: number;
  categories: string[];
  document_types: string[];
  collection_name: string;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Chat endpoint
  async sendChatMessage(message: ChatMessage): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  // Health check
  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  // Get stats
  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/api/stats');
  }

  // Search documents
  async searchDocuments(query: string, top_k: number = 5): Promise<any> {
    return this.request(`/api/search?query=${encodeURIComponent(query)}&top_k=${top_k}`);
  }

  // Get document by ID
  async getDocument(docId: string): Promise<any> {
    return this.request(`/api/doc/${docId}`);
  }

  // Build database (admin)
  async buildDatabase(reset: boolean = false, maxItems?: number, filename?: string): Promise<any> {
    return this.request('/api/build-db', {
      method: 'POST',
      body: JSON.stringify({ reset, max_items: maxItems, filename }),
    });
  }
}

export const apiService = new ApiService();
