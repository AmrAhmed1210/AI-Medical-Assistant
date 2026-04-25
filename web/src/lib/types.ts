// ============================================================================
// 🔐 Auth & Users
// ============================================================================

// ── أدوار المستخدمين ──────────────────────────────────────────────────
export type UserRole = 'Admin' | 'Doctor' | 'Patient'

// ── مستخدم (يتطابق مع رد الـ API من Swagger) ─────────────────────────
export type UserDto = {
  id: number
  name: string
  email: string
  role: UserRole
  isActive: boolean
  photoUrl?: string
  createdAt?: string
}

// ── مستخدم (تنسيق داخلي للتطبيق - للراحة في المكونات) ─────────────────
export type MappedUserDto = {
  id: number
  name: string
  email: string
  role: UserRole
  isActive: boolean
  photoUrl?: string
  createdAt?: string
}

// ── دالة تحويل من تنسيق الـ API للتنسيق الداخلي ───────────────────────
export const mapUserDto = (user: UserDto): MappedUserDto => ({
  id: user.id,
  name: user.name,
  email: user.email,
  role: user.role as UserRole,
  isActive: user.isActive,
  photoUrl: user.photoUrl,
  createdAt: user.createdAt,
})

// ── دالة تحويل عكسي (من الداخلي للـ API) ──────────────────────────────
export const unmapUserDto = (user: MappedUserDto): Omit<UserDto, 'createdAt'> => ({
  id: user.id as number,
  name: user.name,
  email: user.email,
  role: user.role,
  isActive: user.isActive,
})

// ── طلب إنشاء مستخدم ─────────────────────────────────────────────────
export type CreateUserRequest = {
  fullName: string
  email: string
  password: string
  role: UserRole
  phoneNumber?: string
  specialityName?: string
  specialityNameAr?: string
  consultationFee?: number
  yearsExperience?: number
  bio?: string
}

// ── إحصائيات النظام ──────────────────────────────────────────────────
export type SystemStatsDto = {
  totalUsers: number
  totalDoctors: number
  totalPatients: number
  totalSessions: number
  totalAppointments: number
  activeModels: number
  avgResponseTimeMs: number
  highUrgencyToday: number
  sessionsToday: number
  sessionsThisWeek: number
  urgencyDistribution: { LOW: number; MEDIUM: number; HIGH: number; EMERGENCY: number }
  sessionsPerDay: Array<{ date: string; count: number }>
  userGrowth: Array<{ date: string; count: number }>
}

// ── استجابة مصفوفة ────────────────────────────────────────────────────
export type PagedResponse<T> = {
  items: T[]
  total: number
  page: number
  pageSize: number
}

// ============================================================================
// 🔑 Auth Requests & Responses
// ============================================================================

export interface RegisterRequest {
  fullName: string
  email: string
  password: string
  role: 'Patient' | 'Doctor'
  phoneNumber?: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface AuthUserDto {
  id: string
  fullName: string
  email: string
  role: UserRole
}

export interface LoginResponse {
  accessToken: string
  refreshToken: string
  expiresIn: number
  user: AuthUserDto
}

// ============================================================================
// 💬 Sessions / Consult
// ============================================================================

export type UrgencyLevel = 'LOW' | 'MEDIUM' | 'HIGH'
export type MessageRole = 'user' | 'assistant' | 'doctor' | 'admin'

export interface SymptomDto {
  term: string
  icd11: string
  severity?: number
}

export interface SessionDto {
  id: string
  title: string | null
  createdAt: string
  updatedAt: string
  messageCount: number
  urgencyLevel: UrgencyLevel | null
  lastMessage?: string | null
  lastMessageAt?: string | null
  patientName?: string | null
  patientPhotoUrl?: string | null
  type?: string
  userId?: number
}

export interface MessageDto {
  id: string
  sessionId?: string | number
  role: MessageRole
  content: string
  messageType?: string
  attachmentUrl?: string | null
  fileName?: string | null
  senderName?: string
  senderPhotoUrl?: string | null
  timestamp: string
}

export interface AnalysisResultDto {
  symptoms: SymptomDto[]
  urgencyLevel: UrgencyLevel
  disclaimer: string
}

export interface SessionDetailDto extends SessionDto {
  messages: MessageDto[]
  analysisResult: AnalysisResultDto | null
}

export interface StartSessionRequest {
  message: string
}

export interface SendMessageRequest {
  content: string
}

// ============================================================================
// 📅 Appointments
// ============================================================================

export type AppointmentStatus = 'Pending' | 'Confirmed' | 'Cancelled' | 'Completed'

export interface AppointmentDto {
  id: string
  patientName: string
  doctorName: string
  scheduledAt: string
  date?: string
  time?: string
  specialty?: string
  paymentMethod?: string
  status: AppointmentStatus
  notes: string | null
}

export interface BookAppointmentRequest {
  doctorId: string
  scheduledAt: string
  sessionId?: string
}

export interface CancelAppointmentRequest {
  reason: string
}

export interface CompleteAppointmentRequest {
  notes: string
}

// ============================================================================
// 🩺 Doctor
// ============================================================================

export type DayOfWeek = 0 | 1 | 2 | 3 | 4 | 5 | 6 // 0=Sun … 6=Sat

// Dashboard ──────────────────────────────────────────────────────────────────

export interface UpcomingAppointmentDto {
  id: string
  patientName: string
  patientPhotoUrl?: string | null
  scheduledAt: string
  status: AppointmentStatus
}

export interface WeeklySessionChartDto {
  day: string
  count: number
}

export interface AIReportDto {
  reportId: string
  patientId: string
  patientName: string
  sessionId: string
  urgencyLevel: UrgencyLevel
  symptoms: SymptomDto[]
  disclaimer: string
  createdAt: string
}

export interface DoctorDashboardDto {
  todayAppointments: number
  pendingAppointments: number
  totalPatients: number
  unreadReports: number
  weekAppointments: number
  upcomingAppointments: UpcomingAppointmentDto[]
  todayAppointmentsList: UpcomingAppointmentDto[]
  weeklySessionsChart: WeeklySessionChartDto[]
  recentReports: AIReportDto[]
}

// Profile ────────────────────────────────────────────────────────────────────

export interface DoctorDetailDto {
  id: string
  userId: string
  fullName: string
  email: string
  specialty: string
  license: string
  bio: string | null
  photoUrl: string | null
  consultFee: number | null
  yearsExperience: number | null
  createdAt: string
  updatedAt: string | null
  isAvailable?: boolean
  isScheduleVisible?: boolean
  specialityNameAr?: string
  rating?: number
}

// Patients ───────────────────────────────────────────────────────────────────

export type Gender = 'Male' | 'Female' | 'Other'
export type BloodType = 'A+' | 'A-' | 'B+' | 'B-' | 'AB+' | 'AB-' | 'O+' | 'O-'

export interface PatientSummaryDto {
  id: string
  fullName: string
  email: string
  phoneNumber: string | null
  dateOfBirth: string | null
  gender: Gender | null
  bloodType: BloodType | null
  allergies: string | null
  totalAppointments: number
  lastVisit: string | null
  photoUrl?: string | null
}

// Availability ───────────────────────────────────────────────────────────────

export interface AvailabilityDto {
  dayOfWeek: DayOfWeek
  dayName?: string
  startTime: string   // "09:00"
  endTime: string     // "17:00"
  isAvailable: boolean
  isActive?: boolean
  slotDurationMinutes?: number
}

// ============================================================================
// 👑 Admin
// ============================================================================

export interface GetUsersParams {
  role?: UserRole
  search?: string
  page?: number
  pageSize?: number
}

export type AgentName = 'symptom-extractor' | 'rag-retriever' | 'response-generator'

export interface ModelVersionDto {
  id: string
  agentName: AgentName
  version: string
  filePath: string
  isActive: boolean
  loadedAt: string | null
  createdAt: string
}

export interface ReloadModelRequest {
  agentName: AgentName
}

// ============================================================================
// ⚠️ Error Response
// ============================================================================

export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'UNAUTHORIZED'
  | 'FORBIDDEN'
  | 'NOT_FOUND'
  | 'CONFLICT'
  | 'UNSAFE_INPUT'
  | 'RATE_LIMITED'
  | 'INTERNAL_ERROR'
  | 'AI_UNAVAILABLE'

export interface ApiErrorResponse {
  statusCode: number
  error: ErrorCode
  message: string
  traceId: string
}

export type NotificationType = 'info' | 'success' | 'warning' | 'error'
export interface NotificationDto {
  id: string
  type: NotificationType
  title: string
  message: string
  read: boolean
  createdAt: string
}

// ============================================================================
// 👨‍⚕️ Reviews
// ============================================================================

export interface ReviewDto {
  id: string
  author: string
  patientName?: string
  rating: number
  comment: string
  createdAt: string | Date
}
