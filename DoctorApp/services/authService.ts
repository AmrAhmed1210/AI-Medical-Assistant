import AsyncStorage from "@react-native-async-storage/async-storage";
import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface RegisterPayload {
  fullName:    string;
  email:       string;
  password:    string;
  role:        string;
  phoneNumber?: string;
  dateOfBirth?: string; // ISO format: YYYY-MM-DD
}

export interface LoginPayload {
  email:    string;
  password: string;
}

export interface AuthResponse {
  token: string;
  id?:   number;
  name:  string;
  email: string;
  role:  string;
  phone: string;
}

function buildIsoDate(year: number, month: number, day: number): string {
  const date = new Date(Date.UTC(year, month - 1, day));

  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    throw new Error("Date of birth is invalid. Use a real date like 2000-05-21.");
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

  throw new Error("Date of birth format is invalid. Use YYYY-MM-DD.");
}

// ============================================
// Register
// ============================================
export const registerApi = async (payload: RegisterPayload): Promise<AuthResponse> => {
  try {
    const normalizedDateOfBirth = normalizeDateOfBirth(payload.dateOfBirth);

    // Transform payload to match backend field names
    const requestBody = {
      fullName: payload.fullName,
      email: payload.email,
      password: payload.password,
      role: payload.role || "Patient",
      phoneNumber: payload.phoneNumber || "",
      dateOfBirth: normalizedDateOfBirth,
    };

    const dto = await apiFetch<any>(
      API.auth.register,
      {
        method: "POST",
        body: JSON.stringify(requestBody),
      },
      false
    );

    // Backend returns: { accessToken, user: { id, fullName, email, role } }
    const token = dto.accessToken || dto.token || "";
    const user = dto.user || {};

    if (!token) {
      throw new Error("No token received from server");
    }

    return {
      token: token,
      id: user.id || 0,
      name: user.fullName || payload.fullName,
      email: user.email || payload.email,
      role: user.role || payload.role || "Patient",
      phone: user.phoneNumber || payload.phoneNumber || "",
    };
  } catch (err: any) {
    const msg = err.message || "";

    // Better error messages for common issues
    if (msg.includes("Email already registered")) {
      throw new Error("This email is already registered. Please login instead.");
    }
    if (msg.includes("dateOfBirth") || msg.includes("DateTime")) {
      throw new Error("Date of birth format is invalid. Use YYYY-MM-DD.");
    }
    if (msg.includes("The dto field is required")) {
      throw new Error("Registration data is invalid. Please check the form and try again.");
    }
    throw err;
  }
};

// ============================================
// Login
// ============================================
export const loginApi = async (payload: LoginPayload): Promise<AuthResponse> => {
  try {
    const dto = await apiFetch<any>(
      API.auth.login,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      false
    );

    // Backend returns: { accessToken, user: { id, fullName, email, role } }
    const token = dto.accessToken || dto.token || "";
    const user = dto.user || {};

    if (!token) {
      throw new Error("No token received from server");
    }

    return {
      token: token,
      id: user.id || 0,
      name: user.fullName || "",
      email: user.email || payload.email,
      role: user.role || "Patient",
      phone: user.phoneNumber || "",
    };
  } catch (err: any) {
    const msg = err.message || "";
    // Pass through specific account status messages from backend
    if (msg.includes("deleted") || msg.includes("deactivated") || msg.includes("contact admin")) {
      throw err; // Pass through as-is
    }
    if (msg.includes("Unauthorized") || msg.includes("Invalid email or password")) {
      throw new Error("Invalid email or password.");
    }
    throw err;
  }
};

// ============================================
// Save session to AsyncStorage
// ============================================
export const saveSession = async (auth: AuthResponse) => {
  try {
    console.log('💾 Saving session for:', auth.email);
    console.log('💾 Token length:', auth.token?.length || 0);
    console.log('💾 Role:', auth.role);

    await AsyncStorage.multiSet([
      ["token",      auth.token],
      ["userToken",  auth.token],
      ["userName",   auth.name],
      ["userEmail",  auth.email],
      ["userPhone",  auth.phone || ""],
      ["userRole",   auth.role || "Patient"],
      ["isLoggedIn", "true"],
      ["user", JSON.stringify({
        id:    auth.id || 0,
        name:  auth.name,
        email: auth.email,
        phone: auth.phone || "",
        role:  auth.role || "Patient",
      })],
    ]);

    // Verify it was saved
    const savedToken = await AsyncStorage.getItem('token');
    const savedIsLoggedIn = await AsyncStorage.getItem('isLoggedIn');
    console.log('✅ Session saved! Token exists:', !!savedToken, 'isLoggedIn:', savedIsLoggedIn);
  } catch (err) {
    console.error("❌ Failed to save session:", err);
    throw err;
  }
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
