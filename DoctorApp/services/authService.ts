import AsyncStorage from "@react-native-async-storage/async-storage";
import { API } from "../constants/api";

// ============================================
// Types
// ============================================
export interface RegisterPayload {
  name:         string;
  email:        string;
  passwordHash: string;
  role:         string;
  phone?:       string;
}

export interface LoginPayload {
  email:        string;
  passwordHash: string;
}

export interface AuthResponse {
  token: string;
  name:  string;
  email: string;
  role:  string;
  phone: string;
}

// ============================================
// Register
// ============================================
export const registerApi = async (payload: RegisterPayload): Promise<AuthResponse> => {
  const res = await fetch(API.auth.register, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.message || "Registration failed");
  }

  return data as AuthResponse;
};

// ============================================
// Login
// ============================================
export const loginApi = async (payload: LoginPayload): Promise<AuthResponse> => {
  const res = await fetch(API.auth.login, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.message || "Login failed");
  }

  return data as AuthResponse;
};

// ============================================
// Save session to AsyncStorage
// ============================================
export const saveSession = async (auth: AuthResponse) => {
  await AsyncStorage.multiSet([
    ["token",      auth.token],
    ["userToken",  auth.token],
    ["userName",   auth.name],
    ["userEmail",  auth.email],
    ["userPhone",  auth.phone],
    ["userRole",   auth.role],
    ["isLoggedIn", "true"],
    ["user", JSON.stringify({
      name:  auth.name,
      email: auth.email,
      phone: auth.phone,
      role:  auth.role,
    })],
  ]);
};

// ============================================
// Get token for protected requests
// ============================================
export const getToken = async (): Promise<string | null> => {
  return await AsyncStorage.getItem("token");
};

// ============================================
// Logout
// ============================================
export const logout = async () => {
  await AsyncStorage.multiRemove([
    "token", "userToken", "userName", "userEmail",
    "userPhone", "userRole", "isLoggedIn", "user",
  ]);
};