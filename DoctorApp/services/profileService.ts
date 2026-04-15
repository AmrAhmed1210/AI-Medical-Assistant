import { API } from "../constants/api";
import { getToken } from "./authService";

// ============================================
// Types
// ============================================
export interface Profile {
  id: string;
  name: string;
  email: string;
  phone: string;
  role: string;
}

// ============================================
// Get My Profile
// ============================================
export const getMyProfile = async (): Promise<Profile> => {
  const token = await getToken();
  const res = await fetch(API.profile.get, {
    headers: { "Authorization": `Bearer ${token}` },
  });

  if (!res.ok) throw new Error("Failed to load profile");
  return res.json();
};

// ============================================
// Update My Profile
// ============================================
export const updateMyProfile = async (name: string, phone: string): Promise<void> => {
  const token = await getToken();
  const res = await fetch(API.profile.update, {
    method:  "PUT",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({ name, phone }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.message || "Update failed");
};