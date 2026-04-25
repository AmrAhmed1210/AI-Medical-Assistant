import { API } from "../constants/api";
import { apiFetch } from "./http";

function buildIsoDate(year: number, month: number, day: number): string {
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    throw new Error("Date of birth is invalid.");
  }
  return `${year.toString().padStart(4, "0")}-${month.toString().padStart(2, "0")}-${day.toString().padStart(2, "0")}T00:00:00Z`;
}

function normalizeDateOfBirth(value?: string): string | null {
  const input = value?.trim();
  if (!input) return null;

  const isoMatch = input.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (isoMatch) {
    const [, year, month, day] = isoMatch;
    return buildIsoDate(Number(year), Number(month), Number(day));
  }

  const slashMatch = input.match(/^(\d{2})[/-](\d{2})[/-](\d{4})$/);
  if (slashMatch) {
    const [, day, month, year] = slashMatch;
    return buildIsoDate(Number(year), Number(month), Number(day));
  }

  const parsed = new Date(input);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toISOString();
  }
  return null;
}

// ============================================
// Types
// ============================================
export interface Profile {
  id: string;
  name: string;
  email: string;
  phone: string;
  role: string;
  dateOfBirth?: string;
  photoUrl?: string;
}

// ============================================
// Get My Profile
// ============================================
export const getMyProfile = async (): Promise<Profile> => {
  return apiFetch<Profile>(API.profile.get, { method: "GET" }, true);
};

// ============================================
// Update My Profile
// ============================================
export const updateMyProfile = async (
  name: string,
  email: string,
  phone: string,
  dateOfBirth: string
): Promise<void> => {
  const normalized = normalizeDateOfBirth(dateOfBirth);
  await apiFetch<unknown>(
    API.profile.update,
    {
      method: "PUT",
      body: JSON.stringify({
        name,
        email,
        phone,
        birthDate: normalized,
        dateOfBirth: normalized,
      }),
    },
    true
  );
};

// ============================================
// Upload Profile Photo
// ============================================
export const uploadProfilePhoto = async (uri: string): Promise<string> => {
  const formData = new FormData();
  
  // Expo File format
  const filename = uri.split('/').pop() || 'photo.jpg';
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : `image/jpeg`;

  formData.append('file', {
    uri,
    name: filename,
    type,
  } as any);

  const response = await apiFetch<{ photoUrl: string }>(
    API.profile.photo, // api/profile/photo
    {
      method: "POST",
      body: formData,
    },
    true
  );

  return response.photoUrl;
};
