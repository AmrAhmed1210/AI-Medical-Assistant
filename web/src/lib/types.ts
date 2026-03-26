

// ── أدوار المستخدمين ──────────────────────────────────────────────────
export type UserRole = 'Admin' | 'Doctor' | 'Patient'

// ── مستخدم (يتطابق مع رد الـ API من Swagger) ─────────────────────────
export type UserDto = {
userId:string
fullName:string
email:string
role:string
isActive:boolean
}
// ── مستخدم (تنسيق داخلي للتطبيق - للراحة في المكونات) ─────────────────
// ده عشان نستخدمه في الـ Components من غير ما نغير كل الكود القديم
export type MappedUserDto = {
  userId: string               // string دايماً عشان نستخدمه كـ key
  fullName: string             // اسم واضح
  email: string
  role: UserRole
  isActive: boolean
  createdAt?: string
}

// ── دالة تحويل من تنسيق الـ API للتنسيق الداخلي ───────────────────────
export const mapUserDto = (user: UserDto): MappedUserDto => ({
  userId: String(user.userId),     // تحويل الـ id لـ string
  fullName: user.fullName,         // name → fullName
  email: user.email,
  role: user.role as UserRole, // Type assertion للـ role
  isActive: user.isActive,
  
})

// ── دالة تحويل عكسي (من الداخلي للـ API) ──────────────────────────────
export const unmapUserDto = (user: MappedUserDto): Omit<UserDto, 'createdAt'> => ({
  userId: user.userId,
  fullName: user.fullName,
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
  specialityName?: string      // للأطباء فقط
  specialityNameAr?: string    // للأطباء فقط
  consultationFee?: number     // للأطباء فقط
  yearsExperience?: number     // للأطباء فقط
  bio?: string                 // للأطباء فقط
}

// ── إحصائيات النظام ──────────────────────────────────────────────────
export type SystemStatsDto = {
  totalUsers: number
  totalDoctors: number
  totalPatients: number
  totalAppointments: number
  sessionsToday?: number
  urgencyDistribution?: Record<string, number>
  sessionsPerDay?: Array<{ date: string; count: number }>
  userGrowth?: Array<{ date: string; count: number }>
}

// ── استجابة مصفوفة (لو الـ Backend عدّل ورجع pagination) ─────────────
export type PagedResponse<T> = {
  items: T[]
  total: number
  page: number
  pageSize: number
}