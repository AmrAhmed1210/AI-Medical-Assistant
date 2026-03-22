import { API } from "../constants/api";
import { getToken } from "./authService";

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
  const token = await getToken();
  const res = await fetch(API.appointments.book, {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.message || "Booking failed");
  return data;
};

// ============================================
// Get My Appointments
// ============================================
export const getMyAppointments = async (): Promise<Appointment[]> => {
  const token = await getToken();
  const res = await fetch(API.appointments.my, {
    headers: { "Authorization": `Bearer ${token}` },
  });

  if (!res.ok) throw new Error("Failed to load appointments");

  const data = await res.json();
  // Handle both array and single object response
  return Array.isArray(data) ? data : [data];
};

// ============================================
// Cancel Appointment
// ============================================
export const cancelAppointment = async (id: number): Promise<void> => {
  const token = await getToken();
  const res = await fetch(API.appointments.cancel(id), {
    method:  "DELETE",
    headers: { "Authorization": `Bearer ${token}` },
  });

  if (!res.ok) throw new Error("Failed to cancel appointment");
};