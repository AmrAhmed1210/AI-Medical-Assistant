// ==================== Auth Types ====================
export type UserRole = 'Admin' | 'Doctor' | 'Patient'

export interface UserDto {
  userId: string
  fullName: string
  email: string
  role: UserRole
  isActive: boolean
  createdAt: string
}

export interface AuthResponseDto {
  token: string
  refreshToken: string
  expiresAt: string
  user: UserDto
}

export interface UserProfileDto extends UserDto {
  phone?: string
  avatarUrl?: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  fullName: string
  email: string
  password: string
  role: UserRole
}

// ==================== Doctor Types ====================
export interface DoctorDetailDto {
  doctorId: string
  fullName: string
  specialityName: string
  specialityNameAr: string
  rating: number
  consultationFee: number
  yearsExperience: number
  isAvailable: boolean
  bio: string
  photoUrl?: string
  email?: string
  phone?: string
}

export interface DoctorDashboardDto {
  todayAppointments: number
  pendingAppointments: number
  totalPatients: number
  weekAppointments: number
  todayAppointmentsList: AppointmentDto[]
  recentReports: AIReportDto[]
  weeklySessionsChart: { day: string; count: number }[]
}

export interface AvailabilityDto {
  availabilityId?: string
  dayOfWeek: number // 0=Sunday, 1=Monday, ...
  dayName: string
  startTime: string
  endTime: string
  slotDurationMinutes: number
  isActive: boolean
}

// ==================== Appointment Types ====================
export type AppointmentStatus = 'Pending' | 'Confirmed' | 'Cancelled' | 'Completed'

export interface AppointmentDto {
  appointmentId: string
  patientName: string
  patientId: string
  doctorName: string
  doctorId: string
  specialityName: string
  scheduledAt: string
  status: AppointmentStatus
  notes?: string
  fee: number
}

// ==================== AI Report Types ====================
export type UrgencyLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'EMERGENCY'

export interface SymptomDto {
  symptom: string
  icdCode?: string
  description?: string
}

export interface CitationDto {
  title: string
  url: string
  source: string
}

export interface AIReportDto {
  reportId: string
  patientName: string
  patientId: string
  createdAt: string
  urgencyLevel: UrgencyLevel
  symptomsJson: SymptomDto[]
  responseSummary: string
  recommendedSpecialty: string
  fullResponseText: string
  hallucinationScore: number
  citations: CitationDto[]
}

// ==================== Patient Types ====================
export interface PatientSummaryDto {
  patientId: string
  fullName: string
  phone?: string
  lastVisit?: string
  totalSessions: number
  urgencyTrend: UrgencyLevel
  age?: number
  gender?: 'male' | 'female'
  email?: string
}

// ==================== Session Types ====================
export type MessageRole = 'user' | 'assistant' | 'doctor'

export interface MessageDto {
  msgId: string
  sessionId: string
  role: MessageRole
  content: string
  timestamp: string
  urgencyLevel?: UrgencyLevel
}

export interface SessionDto {
  sessionId: string
  userId: string
  patientName: string
  startTime: string
  endTime?: string
  status: 'Active' | 'Completed' | 'Abandoned'
  finalUrgency?: UrgencyLevel
  messages: MessageDto[]
}

export interface SessionDetailDto extends SessionDto {
  report?: AIReportDto
}

// ==================== Admin Types ====================
export interface UrgencyDistribution {
  LOW: number
  MEDIUM: number
  HIGH: number
  EMERGENCY: number
}

export interface SystemStatsDto {
  totalUsers: number
  totalDoctors: number
  totalPatients: number
  sessionsToday: number
  sessionsThisWeek: number
  urgencyDistribution: UrgencyDistribution
  sessionsPerDay: { date: string; count: number }[]
  userGrowth: { date: string; count: number }[]
}

export interface ModelVersionDto {
  modelId: string
  agentName: string
  version: string
  filePath: string
  isActive: boolean
  deployedAt: string
}

// ==================== Consult Types ====================
export interface ConsultResponseDto {
  sessionId: string
  message: string
  urgencyLevel: UrgencyLevel
  isComplete: boolean
}

// ==================== Notification Types ====================
export type NotificationType = 'success' | 'error' | 'warning' | 'info'

export interface NotificationDto {
  id: string
  type: NotificationType
  title: string
  message: string
  createdAt: string
  read: boolean
}

// ==================== Create User (Admin) ====================
export interface CreateUserRequest {
  fullName: string
  email: string
  password: string
  role: UserRole
  // Doctor-specific
  specialityName?: string
  specialityNameAr?: string
  consultationFee?: number
  yearsExperience?: number
  bio?: string
}
