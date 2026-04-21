import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface Appointment {
  id: number;
  patientId: number;
  doctorId: number;
  doctorName: string;
  specialty: string;
  date: string;
  time: string;
  paymentMethod: string;
  status: string;
  notes?: string;
}

export interface BookAppointmentPayload {
  doctorId: number;
  date: string;
  time: string;
  paymentMethod: "visa" | "cash";
  notes?: string;
}

// ============================================
// Book Appointment
// ============================================
export const bookAppointment = async (payload: BookAppointmentPayload): Promise<Appointment> => {
  return apiFetch<Appointment>(
    API.appointments.book,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    true
  );
};

// ============================================
// Get My Appointments
// ============================================
export const getMyAppointments = async (): Promise<Appointment[]> => {
  const data = await apiFetch<any>(API.appointments.my, { method: "GET" }, true);
  // Handle both array and single object response
  return Array.isArray(data) ? data : [data];
};

// ============================================
// Cancel Appointment
// ============================================
export const cancelAppointment = async (id: number): Promise<void> => {
  await apiFetch<unknown>(API.appointments.cancel(id), { method: "DELETE" }, true);
};