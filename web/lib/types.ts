export type UserRole = "patient" | "doctor" | "admin"

export interface User {
  id: string; name: string; email: string; role: UserRole; avatar?: string;
}

export interface AdminStats {
  totalUsers: number; totalDoctors: number; totalPatients: number;
  totalAppointments: number; revenue: number; activeConsultations: number;
}
