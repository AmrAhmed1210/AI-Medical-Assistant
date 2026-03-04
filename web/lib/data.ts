export interface Doctor {
  id: string
  name: string
  specialty: string
  qualifications: string
  rating: number
  reviews: number
  fees: number
  image: string
  tags: string[]
  bio: string
  availableDates: { date: number; day: string }[]
  timeSlots: {
    morning: string[]
    afternoon: string[]
    evening: string[]
  }
}

export interface Category {
  id: string
  name: string
  icon: string
}

export interface Patient {
  id: string
  name: string
  age: number
  gender: string
  condition: string
  lastVisit: string
  status: "stable" | "critical" | "recovering"
  avatar: string
}

export interface Hospital {
  id: string
  name: string
  address: string
  phone: string
  doctorsCount: number
  rating: number
  reviews: number
  image: string
  status: "active" | "inactive"
  facilities: string[]
  doctors : Doctor[]
}

export interface AIReport {
  id: string
  patientId: string
  patientName: string
  date: string
  diagnosis: string
  severity: "low" | "medium" | "high" | "critical"
  urgency: "routine" | "soon" | "urgent" | "emergency"
  symptoms: string[]
  recommendations: string[]
  confidence: number
}

export interface AdminStats {
  totalUsers: number
  totalDoctors: number
  totalPatients: number
  totalHospitals: number
  totalAppointments: number
  revenue: number
  activeConsultations: number
}

export const categories: Category[] = [
  { id: "neurology", name: "Neurology", icon: "brain" },
  { id: "cardiology", name: "Cardiology", icon: "heart-pulse" },
  { id: "orthopedics", name: "Orthopedics", icon: "bone" },
  { id: "pathology", name: "Pathology", icon: "microscope" },
  { id: "dermatology", name: "Dermatology", icon: "scan-face" },
  { id: "pediatrics", name: "Pediatrics", icon: "baby" },
]

export const hospitals: Hospital[] = [
  {
    id: "hosp-001",
    name: "MediCare Central Hospital",
    address: "123 Healthcare Ave, Downtown",
    phone: "+1 (555) 100-2000",
    doctorsCount: 45,
    rating: 4.8,
    reviews: 1200,
    image: "/images/hospital-1.jpg",
    status: "active",
    facilities: ["Emergency 24/7", "MRI Scan", "Pharmacy", "ICU"],
    doctors: [
      {
        id: "dr-eion-morgan",
        name: "Dr. Eion Morgan",
        specialty: "Neurology",
        qualifications: "MBBS, MD (Neurology)",
        rating: 4.5,
        reviews: 2530,
        fees: 50.99,
        image: "/images/doctor-profile.jpg",
        tags: ["Neurologist", "Neuromedicine"],
        bio: "Dedicated neurologist with over 15 years of experience.",
        availableDates: [{ date: 15, day: "Mon" }],
        timeSlots: { morning: ["09:00 AM"], afternoon: ["01:00 PM"], evening: ["05:00 PM"] }
      }
    ]
  },
  {
    id: "hosp-002",
    name: "City Medical Center",
    address: "456 Medical Dr, Midtown",
    phone: "+1 (555) 200-3000",
    doctorsCount: 32,
    rating: 4.6,
    reviews: 850,
    image: "/images/hospital-2.jpg",
    status: "active",
    facilities: ["Cardiology Unit", "Laboratory", "Pediatrics"],
    doctors: [
      {
        id: "dr-james-patel",
        name: "Dr. James Patel",
        specialty: "Cardiology",
        qualifications: "MBBS, DM (Cardiology)",
        rating: 4.8,
        reviews: 3120,
        fees: 65.99,
        image: "/images/doctor-3.jpg",
        tags: ["Cardiologist", "Heart Specialist"],
        bio: "Leading cardiologist with over 20 years of experience.",
        availableDates: [{ date: 16, day: "Tue" }],
        timeSlots: { morning: ["08:30 AM"], afternoon: ["01:00 PM"], evening: ["04:30 PM"] }
      }
    ]
  },
  {
    id: "hosp-003",
    name: "NeuroScience Institute",
    address: "654 Brain Ave, Eastside",
    phone: "+1 (555) 500-6000",
    doctorsCount: 15,
    rating: 4.7,
    reviews: 450,
    image: "/images/hospital-3.jpg",
    status: "inactive",
    facilities: ["Neurology Lab", "Rehabilitation Center"],
    doctors: []
  }
]

export const doctors: Doctor[] = [
  {
    id: "dr-eion-morgan",
    name: "Dr. Eion Morgan",
    specialty: "Neurology",
    qualifications: "MBBS, MD (Neurology)",
    rating: 4.5,
    reviews: 2530,
    fees: 50.99,
    image: "/images/doctor-profile.jpg",
    tags: ["Neurologist", "Neuromedicine", "Medicine"],
    bio: "Dr. Eion Morgan is a dedicated neurologist with over 15 years of experience in caring for patients with complex neurological conditions.",
    availableDates: [
      { date: 15, day: "Mon" },
      { date: 16, day: "Tue" },
    ],
    timeSlots: {
      morning: ["09:00 AM", "09:30 AM"],
      afternoon: ["01:00 PM", "01:30 PM"],
      evening: ["05:00 PM", "05:30 PM"],
    },
  },
  {
    id: "dr-james-patel",
    name: "Dr. James Patel",
    specialty: "Cardiology",
    qualifications: "MBBS, DM (Cardiology)",
    rating: 4.8,
    reviews: 3120,
    fees: 65.99,
    image: "/images/doctor-3.jpg",
    tags: ["Cardiologist", "Heart Specialist"],
    bio: "Dr. James Patel is a leading cardiologist with over 20 years of experience in interventional cardiology.",
    availableDates: [
      { date: 15, day: "Mon" },
      { date: 16, day: "Tue" },
    ],
    timeSlots: {
      morning: ["08:30 AM", "09:00 AM"],
      afternoon: ["01:00 PM", "02:00 PM"],
      evening: ["04:30 PM", "06:00 PM"],
    },
  },
]

export const patients: Patient[] = [
  { id: "p-001", name: "Mr. Williamson", age: 42, gender: "Male", condition: "Chronic Migraine", lastVisit: "2024-01-15", status: "stable", avatar: "MW" },
  { id: "p-003", name: "Ahmed Hassan", age: 58, gender: "Male", condition: "Atrial Fibrillation", lastVisit: "2024-01-20", status: "critical", avatar: "AH" },
]

export const adminStats: AdminStats = {
  totalUsers: 12480,
  totalDoctors: 342,
  totalPatients: 11850,
  totalHospitals: 3,
  totalAppointments: 4230,
  revenue: 284500,
  activeConsultations: 89,
}