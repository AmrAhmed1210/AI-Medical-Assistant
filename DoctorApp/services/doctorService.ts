import { API } from "../constants/api";
import { getToken } from "./authService";

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
}

export interface DoctorDetails extends Doctor {
  experience: number;
  bio: string;
}

export interface Review {
  id: string;
  author: string;
  rating: number;
  comment: string;
  date: string;
}

// ============================================
// Get All Doctors
// ============================================
export const getAllDoctors = async (specialtyId?: number): Promise<Doctor[]> => {
  const url = specialtyId
    ? API.doctors.getBySpecialty(specialtyId)
    : API.doctors.getAll;

  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load doctors");
  return res.json();
};

// ============================================
// Get Doctor by ID
// ============================================
export const getDoctorById = async (id: number | string): Promise<DoctorDetails> => {
  const res = await fetch(API.doctors.getById(id));
  if (!res.ok) throw new Error("Doctor not found");
  return res.json();
};

// ============================================
// Get Reviews by Doctor
// ============================================
export const getReviewsByDoctor = async (doctorId: number | string): Promise<Review[]> => {
  const res = await fetch(API.reviews.getByDoctor(doctorId));
  if (!res.ok) throw new Error("Failed to load reviews");
  return res.json();
};

// ============================================
// Add Review
// ============================================
export const addReview = async (doctorId: number, rating: number, comment: string): Promise<Review> => {
  const token = await getToken();
  const res = await fetch(API.reviews.add, {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({ doctorId, rating, comment }),
  });
  if (!res.ok) throw new Error("Failed to add review");
  return res.json();
};