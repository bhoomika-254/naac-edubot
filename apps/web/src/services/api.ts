import axios, { AxiosInstance, AxiosResponse } from 'axios'
import {
  ComplianceResponse,
  QueryRequest,
  SystemHealth,
  SystemStats,
  SchedulerStatus,
  UploadResponse,
  IngestRequest,
  IngestStatusRequest,
  IngestStatusResponse,
  UpdateRequest,
  ScheduleRequest,
  ApiError
} from '../types'

const SUPABASE_STAGED_PREFIX = 'supabase://'
const SUPABASE_URL = String(import.meta.env.VITE_SUPABASE_URL || '').trim().replace(/\/+$/, '')
const SUPABASE_BROWSER_KEY = String(
  import.meta.env.VITE_SUPABASE_ANON_KEY ||
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ||
  ''
).trim()
const SUPABASE_UPLOAD_BUCKET = String(import.meta.env.VITE_SUPABASE_UPLOAD_BUCKET || 'edubot-uploads').trim()
const SUPABASE_BROWSER_KEY_IS_JWT = SUPABASE_BROWSER_KEY.split('.').length === 3

const DIRECT_SUPABASE_UPLOAD_ENABLED =
  Boolean(SUPABASE_URL) && Boolean(SUPABASE_BROWSER_KEY) && Boolean(SUPABASE_UPLOAD_BUCKET)

class ApiService {
  private api: AxiosInstance

  private ensureDirectUploadConfigured(): void {
    if (DIRECT_SUPABASE_UPLOAD_ENABLED) {
      return
    }

    const apiError: ApiError = {
      detail:
        'Direct Supabase upload is not configured. Set VITE_SUPABASE_URL, ' +
        'VITE_SUPABASE_UPLOAD_BUCKET, and one of VITE_SUPABASE_ANON_KEY or VITE_SUPABASE_PUBLISHABLE_KEY in Vercel environment variables.',
      status_code: 500,
      timestamp: new Date().toISOString(),
    }
    throw apiError
  }

  private sanitizeFileName(fileName: string): string {
    const trimmed = String(fileName || 'uploaded.pdf').trim() || 'uploaded.pdf'
    const lower = trimmed.toLowerCase()
    return lower.replace(/[^a-z0-9._-]+/g, '_')
  }

  private buildSupabaseObjectPath(documentType: 'naac_requirement' | 'mvsr_evidence', fileName: string): string {
    const stamp = new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)
    const nonce = Math.random().toString(36).slice(2, 10)
    const safeName = this.sanitizeFileName(fileName)
    return `${documentType}/${stamp}_${nonce}_${safeName}`
  }

  private encodeStoragePath(path: string): string {
    return String(path || '')
      .split('/')
      .map((segment) => encodeURIComponent(segment))
      .join('/')
  }

  private async uploadToSupabaseStorage(
    file: File,
    documentType: 'naac_requirement' | 'mvsr_evidence'
  ): Promise<{ bucket: string; objectPath: string }> {
    this.ensureDirectUploadConfigured()

    const bucket = SUPABASE_UPLOAD_BUCKET
    const objectPath = this.buildSupabaseObjectPath(documentType, file.name)
    const encodedBucket = encodeURIComponent(bucket)
    const encodedPath = this.encodeStoragePath(objectPath)
    const uploadUrl = `${SUPABASE_URL}/storage/v1/object/${encodedBucket}/${encodedPath}`

    const uploadHeaders: Record<string, string> = {
      apikey: SUPABASE_BROWSER_KEY,
      'Content-Type': 'application/pdf',
      'x-upsert': 'false',
    }
    if (SUPABASE_BROWSER_KEY_IS_JWT) {
      uploadHeaders.Authorization = `Bearer ${SUPABASE_BROWSER_KEY}`
    }

    const response = await fetch(uploadUrl, {
      method: 'POST',
      headers: uploadHeaders,
      body: file,
    })

    if (!response.ok) {
      let detail = `Supabase upload failed with status ${response.status}`
      try {
        const payload = await response.json()
        detail = payload?.error || payload?.message || detail
      } catch {
        try {
          const text = await response.text()
          if (text) detail = text
        } catch {
          // noop
        }
      }

      const apiError: ApiError = {
        detail:
          `Failed to upload directly to Supabase Storage: ${detail}. ` +
          'Confirm bucket insert policy for anon and verify the browser key belongs to the same Supabase project.',
        status_code: response.status,
        timestamp: new Date().toISOString(),
      }
      throw apiError
    }

    return { bucket, objectPath }
  }

  private shouldAttachAuth(url?: string): boolean {
    const raw = String(url || '').trim()
    if (!raw) return true

    const normalized = raw.startsWith('/') ? raw : `/${raw}`

    // Upload + ingest routes are intentionally unauthenticated on the backend.
    // Avoid sending a stale bearer token that can trigger intermittent 401s.
    if (normalized === '/upload' || normalized.startsWith('/upload?')) return false
    if (normalized === '/upload/reference' || normalized.startsWith('/upload/reference?')) return false
    if (normalized === '/ingest' || normalized.startsWith('/ingest?')) return false
    if (normalized === '/ingest/status' || normalized.startsWith('/ingest/status?')) return false

    return true
  }

  constructor() {
    const configuredBaseRaw =
      (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
      ((import.meta as any).env?.REACT_APP_API_BASE_URL as string | undefined)
    const configuredBase = configuredBaseRaw?.trim()
    let normalizedBase = '/api'

    if (configuredBase) {
      const trimmed = configuredBase.replace(/\/+$/, '')
      // If user provides full backend host without /api, force API prefix.
      normalizedBase = trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`
    }

    this.api = axios.create({
      baseURL: normalizedBase,
      timeout: 180000, // 3 minutes timeout for LLM responses on CPU
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Add request interceptor for logging
    this.api.interceptors.request.use(
      (config) => {
        const token = sessionStorage.getItem('auth_token')
        if (token && this.shouldAttachAuth(config.url)) {
          const headers = config.headers as any
          if (headers && typeof headers.set === 'function') {
            headers.set('Authorization', `Bearer ${token}`)
          } else {
            config.headers = {
              ...(config.headers || {}),
              Authorization: `Bearer ${token}`,
            } as any
          }
        }
        console.log(`[API] Making request: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`)
        console.log(`[API] Full URL: ${config.baseURL}${config.url}`)
        return config
      },
      (error) => {
        console.error('API Request Error:', error)
        return Promise.reject(error)
      }
    )

    // Add response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        const responseStatus = error?.response?.status
        const requestUrlRaw = String(error?.config?.url || '').trim()
        const requestUrl = requestUrlRaw.startsWith('/') ? requestUrlRaw : `/${requestUrlRaw}`

        const isUploadOrIngestRoute =
          requestUrl === '/upload' ||
          requestUrl.startsWith('/upload?') ||
          requestUrl === '/upload/reference' ||
          requestUrl.startsWith('/upload/reference?') ||
          requestUrl === '/ingest' ||
          requestUrl.startsWith('/ingest?') ||
          requestUrl === '/ingest/status' ||
          requestUrl.startsWith('/ingest/status?')

        // If infrastructure/proxy returns a transient 401 for upload routes,
        // retry once without Authorization header.
        if (responseStatus === 401 && isUploadOrIngestRoute && !error?.config?._retryWithoutAuth) {
          const retryConfig = { ...(error.config || {}), _retryWithoutAuth: true }
          const headers = retryConfig.headers as any
          if (headers && typeof headers.delete === 'function') {
            headers.delete('Authorization')
          } else if (headers && typeof headers === 'object') {
            delete headers.Authorization
          }
          return this.api.request(retryConfig)
        }

        const responseData = error?.response?.data
        let detail = responseData?.detail || responseData?.error?.message || error.message || 'Unknown error'

        if (error?.response?.status === 413) {
          detail =
            'The uploaded payload is too large for the current gateway/storage limits. ' +
            'Please compress the PDF or split it into smaller parts, then upload again.'
        }

        const apiError: ApiError = {
          detail,
          status_code: error.response?.status || 500,
          timestamp: new Date().toISOString(),
        }
        console.error('API Response Error:', apiError)
        return Promise.reject(apiError)
      }
    )
  }

  // Query endpoints
  async queryCompliance(request: QueryRequest): Promise<ComplianceResponse> {
    const response: AxiosResponse<ComplianceResponse> = await this.api.post('/query', request)
    return response.data
  }

  // System health endpoints
  async getSystemHealth(): Promise<SystemHealth> {
    const response: AxiosResponse<SystemHealth> = await this.api.get('/health')
    return response.data
  }

  async getSystemStats(): Promise<SystemStats> {
    const response: AxiosResponse<SystemStats> = await this.api.get('/stats')
    return response.data
  }

  async getLastSync(): Promise<any> {
    const response = await this.api.get('/last-sync')
    return response.data
  }

  // Document ingestion endpoints
  async ingestDocuments(request: IngestRequest): Promise<any> {
    const response = await this.api.post('/ingest', request)
    return response.data
  }

  async getIngestionStatuses(request: IngestStatusRequest): Promise<IngestStatusResponse> {
    const response: AxiosResponse<IngestStatusResponse> = await this.api.post('/ingest/status', request)
    return response.data
  }

  async uploadDocument(file: File, documentType: 'naac_requirement' | 'mvsr_evidence'): Promise<UploadResponse> {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      const apiError: ApiError = {
        detail: 'Only PDF files are supported.',
        status_code: 400,
        timestamp: new Date().toISOString(),
      }
      return Promise.reject(apiError)
    }

    const { bucket, objectPath } = await this.uploadToSupabaseStorage(file, documentType)

    const response: AxiosResponse<UploadResponse> = await this.api.post('/upload/reference', {
      filename: file.name,
      file_size: file.size,
      document_type: documentType,
      bucket,
      object_path: objectPath,
    })

    // Ensure staged path always encodes bucket/object for downstream ingest.
    if (!response.data.stored_path) {
      response.data.stored_path = `${SUPABASE_STAGED_PREFIX}${bucket}/${objectPath}`
    }
    return response.data
  }

  async deleteStagedUpload(storedPath: string): Promise<any> {
    const response = await this.api.delete('/upload', {
      data: { stored_path: storedPath },
    })
    return response.data
  }

  // Update endpoints
  async forceUpdate(request: UpdateRequest): Promise<any> {
    const response = await this.api.post('/force-update', request)
    return response.data
  }

  // Scheduler endpoints
  async getSchedulerStatus(): Promise<SchedulerStatus> {
    const response: AxiosResponse<SchedulerStatus> = await this.api.get('/scheduler/status')
    return response.data
  }

  async scheduleJob(request: ScheduleRequest): Promise<any> {
    const response = await this.api.post('/scheduler/schedule', request)
    return response.data
  }

  async pauseJob(jobId: string): Promise<any> {
    const response = await this.api.post(`/scheduler/jobs/${jobId}/pause`)
    return response.data
  }

  async resumeJob(jobId: string): Promise<any> {
    const response = await this.api.post(`/scheduler/jobs/${jobId}/resume`)
    return response.data
  }

  async removeJob(jobId: string): Promise<any> {
    const response = await this.api.delete(`/scheduler/jobs/${jobId}`)
    return response.data
  }

  // Mapping analysis endpoint
  async analyzeQueryMapping(query: string): Promise<any> {
    const response = await this.api.get('/mapping/analyze', {
      params: { query },
    })
    return response.data
  }

  // Utility method for checking API connectivity
  async checkConnectivity(): Promise<boolean> {
    try {
      await this.api.get('/health')
      return true
    } catch (error) {
      console.error('API connectivity check failed:', error)
      return false
    }
  }

  // Download file helper (for future use with reports)
  async downloadFile(url: string, filename: string): Promise<void> {
    const response = await this.api.get(url, {
      responseType: 'blob',
    })

    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }
}

// Create singleton instance
const apiService = new ApiService()
export default apiService

// Export types and utility functions
export type { ApiError }

export const isApiError = (error: any): error is ApiError => {
  return error && typeof error === 'object' && 'detail' in error && 'status_code' in error
}

export const getErrorMessage = (error: unknown): string => {
  if (isApiError(error)) {
    return error.detail
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}
