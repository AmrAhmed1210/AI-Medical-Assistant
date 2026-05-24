import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface Doctor {
  id: number;
  name: string;
  specialty: string;
  rating: number;
  reviewCount: number;
  location: string;
  consultationFee: number;
  isAvailable: boolean;
  imageUrl: string;
  bio?: string | null;
  yearsExperience?: number | null;
  photoUrl?: string | null;
  consultFee?: number | null;
}

export interface DoctorDetails extends Doctor {
  experience?: number;
  bio?: string;
  schedule?: {
    doctorId: number;
    doctorName: string;
    isMobileEnabled: boolean;
    isProfileComplete: boolean;
    hasSchedule: boolean;
    days: Array<{
      dayOfWeek: number;
      dayName: string;
      startTime: string;
      endTime: string;
      isAvailable: boolean;
      slotDurationMinutes: number;
      timeSlots: string[];
    }>;
    bookedSlots: Array<{
      date: string;
      time: string;
    }>;
  };
}

export interface Review {
  id: string | number;
  author: string;
  rating: number;
  comment: string;
  date: string;
  isMine?: boolean;
}

// ============================================
// Get All Doctors
// ============================================
export const getAllDoctors = async (specialtyId?: number): Promise<Doctor[]> => {
  const url = specialtyId
    ? API.doctors.getBySpecialty(specialtyId)
    : API.doctors.getAll;
  return apiFetch<Doctor[]>(url, { method: "GET" }, false);
};

// ============================================
// Get Doctor by ID
// ============================================
export const getDoctorById = async (id: number | string): Promise<DoctorDetails> => {
  return apiFetch<DoctorDetails>(API.doctors.getById(id), { method: "GET" }, false);
};

// ============================================
// Get Reviews by Doctor
// ============================================
export const getReviewsByDoctor = async (doctorId: number | string): Promise<Review[]> => {
  const data = await apiFetch<any[]>(API.reviews.getByDoctor(doctorId), { method: "GET" }, false);
  const reviews = Array.isArray(data) ? data : [];
  return reviews.map((item) => ({
    id: item.id ?? "",
    author: item.author ?? item.patientName ?? "Anonymous",
    rating: Number(item.rating ?? 0),
    comment: item.comment ?? "",
    date: item.date ?? item.createdAt ?? "",
    isMine: !!item.isMine,
  }));
};

// ============================================
// Check if Review Exists
// ============================================
export const checkReviewExists = async (doctorId: number | string): Promise<Review | null> => {
  const reviews = await getReviewsByDoctor(doctorId);
  return reviews.find(r => r.isMine) || null;
};

// ============================================
// Add Review
// ============================================
export const addReview = async (
  doctorId: number,
  rating: number,
  comment: string,
  author?: string
): Promise<Review> => {
  const data = await apiFetch<any>(
    API.reviews.add,
    {
      method: "POST",
      body: JSON.stringify({
        doctorId,
        rating,
        comment,
      }),
    },
    true
  );
  return {
    id: data?.id ?? "",
    author: data?.author ?? data?.patientName ?? author ?? "Anonymous",
    rating: Number(data?.rating ?? rating ?? 0),
    comment: data?.comment ?? comment,
    date: data?.date ?? data?.createdAt ?? "",
  };
};

export const updateMyReview = async (
  doctorId: number,
  rating: number,
  comment: string,
  reviewId?: number | string
): Promise<Review> => {
  let data: any;
  try {
    data = await apiFetch<any>(
      API.reviews.updateMine(doctorId),
      {
        method: "PUT",
        body: JSON.stringify({ rating, comment }),
      },
      true
    );
  } catch (error: any) {
    if (error?.status === 404 && reviewId != null) {
      data = await apiFetch<any>(
        API.reviews.updateById(reviewId),
        {
          method: "PUT",
          body: JSON.stringify({ rating, comment }),
        },
        true
      );
    } else {
      throw error;
    }
  }

  return {
    id: data?.id ?? "",
    author: data?.author ?? data?.patientName ?? "Anonymous",
    rating: Number(data?.rating ?? rating ?? 0),
    comment: data?.comment ?? comment,
    date: data?.date ?? data?.createdAt ?? "",
  };
};

export const submitDoctorReview = addReview;
export const updateDoctorReview = updateMyReview;

export const deleteMyReview = async (
  doctorId: number,
  reviewId?: number | string
): Promise<void> => {
  try {
    await apiFetch<unknown>(
      API.reviews.deleteMine(doctorId),
      { method: "DELETE" },
      true
    );
  } catch (error: any) {
    if (error?.status === 404 && reviewId != null) {
      await apiFetch<unknown>(
        API.reviews.deleteById(reviewId),
        { method: "DELETE" },
        true
      );
      return;
    }
    throw error;
  }
};

export type DoctorProfileDto = {
  id: number;
  fullName: string;
  specialty: string;
  specialityNameAr?: string | null;
  bio?: string | null;
  photoUrl?: string | null;
  consultFee?: number | null;
  yearsExperience?: number | null;
  isAvailable?: boolean;
  email?: string;
};

export type TodayAppointmentItem = {
  id: number;
  patientId: number;
  patientName?: string;
  patientPhotoUrl?: string;
  scheduledAt?: string;
  status?: string;
  time?: string;
  notes?: string;
};

export type DoctorDashboardDto = {
  todayAppointments: number;
  pendingAppointments: number;
  totalPatients: number;
  weekAppointments: number;
  todayAppointmentsList: TodayAppointmentItem[];
  weeklySessionsChart: Array<{ day: string; count: number }>;
  recentReports: Array<any>;
};

export const getDoctorProfile = async (): Promise<DoctorProfileDto> =>
  apiFetch<DoctorProfileDto>(API.doctors.profile, { method: "GET" }, true);

export const getDoctorDashboard = async (): Promise<DoctorDashboardDto> =>
  apiFetch<DoctorDashboardDto>(API.doctors.dashboard, { method: "GET" }, true);

export const updateDoctorProfile = async (data: Partial<DoctorProfileDto>): Promise<void> => {
  await apiFetch(API.doctors.profile, {
    method: "PUT",
    body: JSON.stringify({
      fullName: data.fullName,
      bio: data.bio,
      yearsExperience: data.yearsExperience,
      consultationFee: data.consultFee,
      isAvailable: data.isAvailable
    })
  }, true);
};

// ============================================
// Appointments & Schedule
// ============================================
export const getDoctorSchedule = async (doctorId: string | number): Promise<any[]> => {
  return apiFetch<any[]>(`${API.doctors.availability}/${doctorId}`, { method: "GET" }, false);
};

export const bookAppointment = async (
  doctorId: string | number,
  date: string,
  time: string,
  paymentMethod: "visa" | "cash"
): Promise<any> => {
  return apiFetch<any>(API.appointments.book, {
    method: "POST",
    body: JSON.stringify({
      doctorId,
      scheduledAt: date,
      time,
      paymentMethod
    })
  }, true);
};
export const uploadDoctorPhoto = async (uri: string): Promise<string> => {
  const formData = new FormData();
  const filename = uri.split("/").pop() || "photo.jpg";
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : `image`;

  formData.append("file", { uri, name: filename, type } as any);

  const res = await apiFetch<any>(API.doctors.uploadPhoto, {
    method: "POST",
    body: formData,
  }, true);

  return res.photoUrl || res.imageUrl;
};
