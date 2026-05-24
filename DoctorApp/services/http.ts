import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import { BASE_URL } from "../constants/api";

type ApiFetchOptions = RequestInit & {
  allowedStatusCodes?: number[];
};

type ErrorPayload = {
  errors?: Record<string, string[]>;
  message?: string;
  title?: string;
};

type ApiRequestError = Error & {
  status?: number;
};

function isExpectedNotFound(status: number, message: string): boolean {
  if (status !== 404) return false;
  const normalized = (message || "").toLowerCase();
  return normalized.includes("no reading found") || normalized.includes("not found");
}

async function extractErrorMessage(response: Response): Promise<string> {
  const rawText = await response.text();
  if (!rawText) {
    return `HTTP ${response.status}`;
  }

  try {
    const payload = JSON.parse(rawText) as ErrorPayload;

    const validationMessages = Object.values(payload.errors ?? {})
      .flat()
      .filter(Boolean);

    if (validationMessages.length > 0) {
      return validationMessages[0];
    }

    return payload.message || payload.title || rawText;
  } catch {
    return rawText;
  }
}

export async function apiFetch<T>(
  url: string,
  options: ApiFetchOptions = {},
  requiresAuth: boolean = false
): Promise<T> {
  try {
    const isFormData = options.body instanceof FormData;

    const headers: Record<string, string> = {
      // Don't set Content-Type for FormData - fetch needs to set it automatically
      // with the correct multipart boundary. For everything else, default to JSON.
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers as Record<string, string>),
    }

    // Remove Content-Type if FormData, even if caller passed one manually
    if (isFormData) {
      delete headers['Content-Type'];
      delete headers['content-type'];
    }

    if (requiresAuth) {
      const token = await AsyncStorage.getItem('token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      } else {
        console.warn('[API] No token found for protected request:', url)
      }
    }

    const response = await fetch(url, { ...options, headers })

    if (options.allowedStatusCodes?.includes(response.status)) {
      return undefined as T
    }
    
    if (response.status === 401) {
      const errorMessage = await extractErrorMessage(response)
      console.warn('[API] Unauthorized - clearing session')
      if (requiresAuth) {
        await AsyncStorage.multiRemove(['token', 'isLoggedIn', 'userRole'])
        router.replace("/(auth)/login")
      }
      const unauthorizedError = new Error(errorMessage) as ApiRequestError
      unauthorizedError.status = response.status
      throw unauthorizedError
    }

    if (!response.ok) {
      const errorMessage = await extractErrorMessage(response)
      if (response.status === 403) {
        console.warn(`[API] Forbidden ${response.status}:`, url)
      } else if (isExpectedNotFound(response.status, errorMessage)) {
        console.warn(`[API] Not found (expected) ${response.status}:`, errorMessage)
      } else {
        console.error(`[API] Error ${response.status}:`, errorMessage)
      }
      const requestError = new Error(errorMessage) as ApiRequestError
      requestError.status = response.status
      throw requestError
    }

    if (response.status === 204) {
      return undefined as T
    }

    const data = await response.json()
    return data as T
    
  } catch (error) {
    const status = (error as ApiRequestError)?.status
    const rawMessage = error instanceof Error ? error.message : String(error)
    const isNetworkFailure =
      rawMessage.includes("Network request failed") ||
      rawMessage.includes("Failed to fetch") ||
      rawMessage.includes("Network Error")

    if (isNetworkFailure) {
      const networkError = new Error(
        `تعذر الاتصال بالسيرفر.\n` +
          `• تأكد أن الموبايل والكمبيوتر على نفس شبكة الواي فاي\n` +
          `• شغّل Backend: dotnet run (منفذ 5194)\n` +
          `• العنوان الحالي: ${BASE_URL}`
      ) as ApiRequestError
      console.error("[API] Network error:", url, "→", BASE_URL)
      throw networkError
    }

    if (typeof status === "number" && status >= 400 && status < 500) {
      console.warn('[API] Request failed:', url, error)
    } else {
      console.error('[API] Fetch failed:', url, error)
    }
    throw error
  }
}
