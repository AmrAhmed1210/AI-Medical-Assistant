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
  followerCount?: number;
  fairScore?: number;
  location?: string | null;
  consultationFee?: number | null;
  isAvailable: boolean;
  imageUrl: string;
  bio?: string | null;
  yearsExperience?: number | null;
  photoUrl?: string | null;
  consultFee?: number | null;
  phoneNumber?: string | null;
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

const SPECIALTY_KEYWORDS: Array<{ specialty: string; keywords: string[] }> = [
  { specialty: "Cardiology", keywords: ["cardio", "heart", "قلب", "صدر", "ضغط", "blood pressure", "hypertension"] },
  { specialty: "Endocrinology", keywords: ["diabetes", "sugar", "سكر", "غدة", "thyroid", "endocrine"] },
  { specialty: "Neurology", keywords: ["neuro", "brain", "headache", "migraine", "اعصاب", "عصب", "مخ", "صداع", "دوخة"] },
  { specialty: "Orthopedics", keywords: ["ortho", "bone", "joint", "back", "عظام", "مفاصل", "ظهر", "ركبة"] },
  { specialty: "Dermatology", keywords: ["derma", "skin", "rash", "جلد", "حساسية جلد", "طفح"] },
  { specialty: "Ophthalmology", keywords: ["eye", "vision", "عين", "نظر"] },
  { specialty: "Pulmonology", keywords: ["lung", "asthma", "breath", "صدر", "ربو", "تنفس"] },
  { specialty: "Gastroenterology", keywords: ["stomach", "colon", "liver", "معدة", "قولون", "كبد", "هضم"] },
  { specialty: "Nephrology", keywords: ["kidney", "renal", "كلى", "كلوي"] },
];

const normalize = (value?: string | null) =>
  (value || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u064B-\u065F]/g, "")
    .replace(/[أإآ]/g, "ا")
    .replace(/ة/g, "ه")
    .replace(/ى/g, "ي")
    .trim();

const SPECIALTY_ALIASES: Record<string, string[]> = {
  Cardiology: ["cardiology", "cardiologist", "heart", "قلب"],
  Endocrinology: ["endocrinology", "endocrinologist", "diabetes", "سكر", "غدد"],
  Neurology: ["neurology", "neurologist", "أعصاب", "اعصاب"],
  Orthopedics: ["orthopedics", "orthopedic", "ortho", "عظام"],
  Dermatology: ["dermatology", "dermatologist", "جلدية", "جلد"],
  Ophthalmology: ["ophthalmology", "ophthalmologist", "eye", "عيون", "عين"],
  Pulmonology: ["pulmonology", "pulmonologist", "chest", "صدر", "رئة"],
  Gastroenterology: ["gastroenterology", "gastroenterologist", "باطنة", "جهاز هضمي"],
  Nephrology: ["nephrology", "nephrologist", "kidney", "كلى"],
  Pediatrics: ["pediatrics", "pediatrician", "children", "أطفال", "اطفال"],
  Gynecology: ["gynecology", "gynecologist", "نساء", "نسا"],
  ENT: ["ent", "ear nose throat", "أنف وأذن", "انف واذن"],
  General: ["general", "internal medicine", "family medicine", "عام", "باطنة"],
};

const EXTRA_SPECIALTY_KEYWORDS: Array<{ specialty: string; keywords: string[] }> = [
  { specialty: "Cardiology", keywords: ["قلب", "صدر", "ضغط", "خفقان", "نهجان", "blood pressure", "hypertension"] },
  { specialty: "Endocrinology", keywords: ["سكر", "غدة", "غدد", "هرمون", "diabetes", "thyroid"] },
  { specialty: "Neurology", keywords: ["أعصاب", "اعصاب", "عصب", "مخ", "صداع", "دوخة", "تنميل", "migraine"] },
  { specialty: "Orthopedics", keywords: ["عظام", "مفاصل", "ظهر", "ركبة", "كسر", "خشونة", "joint"] },
  { specialty: "Dermatology", keywords: ["جلد", "جلدية", "حساسية جلد", "طفح", "حبوب", "اكزيما", "rash"] },
  { specialty: "Ophthalmology", keywords: ["عين", "عيون", "نظر", "زغللة", "vision"] },
  { specialty: "Pulmonology", keywords: ["صدر", "رئة", "ربو", "تنفس", "كحة", "سعال", "asthma"] },
  { specialty: "Gastroenterology", keywords: ["معدة", "قولون", "كبد", "هضم", "بطن", "حموضة", "stomach"] },
  { specialty: "Nephrology", keywords: ["كلى", "كلوي", "كلية", "بول", "kidney"] },
  { specialty: "Pediatrics", keywords: ["أطفال", "اطفال", "طفل", "رضيع", "child"] },
  { specialty: "Gynecology", keywords: ["حمل", "دورة", "نساء", "نسا", "رحم", "pregnancy"] },
  { specialty: "ENT", keywords: ["أذن", "اذن", "أنف", "انف", "حنجرة", "زور", "ear", "nose", "throat"] },
  { specialty: "General", keywords: ["حمى", "سخونية", "تعب", "إرهاق", "ارهاق", "fever", "fatigue"] },
];

const GREETING_PATTERNS = [
  "السلام عليكم",
  "وعليكم السلام",
  "صباح الخير",
  "مساء الخير",
  "اهلا",
  "اهلاً",
  "مرحبا",
  "hello",
  "hi",
  "hey",
];

const DOCTOR_REQUEST_TERMS = [
  "doctor",
  "specialist",
  "clinic",
  "appointment",
  "book",
  "دكتور",
  "دكتورة",
  "طبيب",
  "طبيبة",
  "تخصص",
  "استشارة",
  "اكشف",
  "كشف",
  "احجز",
  "حجز",
  "عيادة",
];

const MEDICAL_SIGNAL_TERMS = [
  "pain",
  "ache",
  "fever",
  "rash",
  "cough",
  "vomit",
  "nausea",
  "dizzy",
  "bleeding",
  "swelling",
  "severe",
  "chronic",
  "symptom",
  "medicine",
  "medication",
  "وجع",
  "ألم",
  "الم",
  "تعب",
  "سخونية",
  "حرارة",
  "حمى",
  "كحة",
  "سعال",
  "قيء",
  "ترجيع",
  "غثيان",
  "دوخة",
  "صداع",
  "نزيف",
  "تورم",
  "حساسية",
  "طفح",
  "حكة",
  "حرقان",
  "تنميل",
  "ضيق",
  "نهجان",
  "خفقان",
  "اسهال",
  "إسهال",
  "امساك",
  "إمساك",
  "دواء",
  "علاج",
  "حبوب",
  "جرعة",
  "حمل",
  "دورة",
  "متأخرة",
  "متاخره",
  "رحم",
  "سكر",
  "ضغط",
  "قلب",
  "صدر",
  "بطن",
  "معدة",
  "عين",
  "جلد",
  "ظهر",
  "ركبة",
  "اذن",
  "أذن",
  "انف",
  "أنف",
  "زور",
];

const DETAIL_SIGNAL_TERMS = [
  "منذ",
  "بقال",
  "بقالي",
  "يوم",
  "يومين",
  "اسبوع",
  "أسبوع",
  "شهر",
  "ساعات",
  "ساعة",
  "شديد",
  "جامد",
  "قوي",
  "مستمر",
  "متكرر",
  "يزيد",
  "بيزيد",
  "مفاجئ",
  "sudden",
  "severe",
  "days",
  "weeks",
  "hours",
  "persistent",
  "recurrent",
];

const includesAny = (source: string, terms: string[]) =>
  terms.some((term) => source.includes(normalize(term)));

const countMatches = (source: string, terms: string[]) =>
  terms.reduce((count, term) => count + (source.includes(normalize(term)) ? 1 : 0), 0);

const isGreetingOnly = (source: string, wordCount: number) =>
  wordCount <= 7 && includesAny(source, GREETING_PATTERNS) && !includesAny(source, MEDICAL_SIGNAL_TERMS);

export const shouldAttemptDoctorRecommendation = (text: string): boolean => {
  const source = normalize(text);
  const words = source.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  if (words.length === 0 || isGreetingOnly(source, words.length)) return false;

  if (includesAny(source, DOCTOR_REQUEST_TERMS)) return true;

  const medicalSignalCount = countMatches(source, MEDICAL_SIGNAL_TERMS);
  const hasDetails = includesAny(source, DETAIL_SIGNAL_TERMS) || /\d/.test(source);

  return medicalSignalCount >= 2 || (medicalSignalCount >= 1 && hasDetails);
};

const inferSpecialtyFromText = (text: string): string | null => {
  const source = normalize(text);
  const extraMatch = EXTRA_SPECIALTY_KEYWORDS.find((item) =>
    item.keywords.some((keyword) => source.includes(normalize(keyword)))
  );
  if (extraMatch) return extraMatch.specialty;

  const match = SPECIALTY_KEYWORDS.find((item) =>
    item.keywords.some((keyword) => source.includes(normalize(keyword)))
  );
  return match?.specialty ?? null;
};

const doctorMatchesSpecialty = (doctorSpecialty: string | null | undefined, specialty: string) => {
  const source = normalize(doctorSpecialty);
  const aliases = SPECIALTY_ALIASES[specialty] ?? [specialty];
  return aliases.some((alias) => source.includes(normalize(alias)));
};

export const getFairDoctorScore = (doctor: Pick<Doctor, "rating" | "reviewCount">) => {
  const rating = Math.max(0, Math.min(5, Number(doctor.rating) || 0));
  const reviewCount = Math.max(0, Number(doctor.reviewCount) || 0);
  if (reviewCount <= 0) return 0;

  const platformAverage = 3.8;
  const minimumConfidenceReviews = 5;
  const bayesianRating =
    (platformAverage * minimumConfidenceReviews + rating * reviewCount) /
    (minimumConfidenceReviews + reviewCount);
  const confidenceBoost = Math.min(0.35, Math.log10(reviewCount + 1) * 0.12);
  return bayesianRating + confidenceBoost;
};

export const sortDoctorsFairly = (items: Doctor[]) =>
  items.slice().sort((a, b) => {
    const scoreDiff = getFairDoctorScore(b) - getFairDoctorScore(a);
    if (Math.abs(scoreDiff) > 0.0001) return scoreDiff;
    return (Number(b.reviewCount) || 0) - (Number(a.reviewCount) || 0);
  });

export const enrichDoctorsWithReviewStats = async (items: Doctor[]): Promise<Doctor[]> => {
  return Promise.all(
    items.map(async (doctor) => {
      try {
        const reviews = await getReviewsByDoctor(doctor.id);
        if (reviews.length === 0) {
          return {
            ...doctor,
            rating: 0,
            reviewCount: 0,
            fairScore: 0,
          };
        }

        const average =
          reviews.reduce((sum, review) => sum + Number(review.rating || 0), 0) / reviews.length;
        const enriched = {
          ...doctor,
          rating: Number(average.toFixed(1)),
          reviewCount: reviews.length,
        };

        return {
          ...enriched,
          fairScore: getFairDoctorScore(enriched),
        };
      } catch {
        const safeDoctor = {
          ...doctor,
          rating: Number(doctor.rating || 0),
          reviewCount: Number(doctor.reviewCount || 0),
        };
        return {
          ...safeDoctor,
          fairScore: getFairDoctorScore(safeDoctor),
        };
      }
    })
  );
};

export const getRecommendedDoctorsForNeed = async (
  needText: string,
  limit = 5
): Promise<{ specialty: string | null; doctors: Doctor[] }> => {
  if (!shouldAttemptDoctorRecommendation(needText)) return { specialty: null, doctors: [] };

  const specialty = inferSpecialtyFromText(needText);
  // If we can't infer a specialty, don't recommend random doctors
  if (!specialty) return { specialty: null, doctors: [] };

  const doctors = await enrichDoctorsWithReviewStats(await getAllDoctors());
  const filtered = doctors.filter((doctor) => doctorMatchesSpecialty(doctor.specialty, specialty));

  // If no doctors match this specialty, return specialty with empty array
  // so the chatbot can tell the user what specialty they need
  if (filtered.length === 0) return { specialty, doctors: [] };

  const ranked = sortDoctorsFairly(filtered).slice(0, limit);
  return { specialty, doctors: ranked };
};

export const formatDoctorRecommendationsForAi = (items: Doctor[]) =>
  items.map((doctor) => ({
    id: doctor.id,
    name: doctor.name,
    specialty: doctor.specialty,
    rating: Number(doctor.rating || 0),
    reviewCount: Number(doctor.reviewCount || 0),
    consultationFee: doctor.consultationFee ?? doctor.consultFee ?? null,
    location: doctor.location,
  }));

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
