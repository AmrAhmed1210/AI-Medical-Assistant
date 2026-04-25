import { BASE_URL } from "../constants/api";
import { apiFetch } from "./http";

export interface SessionMessage {
  id: number;
  sessionId: number;
  role: string;
  content: string;
  messageType: string;
  attachmentUrl?: string | null;
  fileName?: string | null;
  senderName?: string;
  senderPhotoUrl?: string | null;
  timestamp: string;
}

export interface SessionItem {
  id: number;
  userId: number;
  title?: string | null;
  createdAt: string;
  updatedAt?: string | null;
  messageCount?: number;
  urgencyLevel?: string | null;
  lastMessage?: string | null;
  lastMessageAt?: string | null;
  patientName?: string | null;
  patientPhotoUrl?: string | null;
  doctorPhotoUrl?: string | null;
  type?: string;
}

export interface SessionDetails extends SessionItem {
  messages: SessionMessage[];
}

const SESSIONS_API = `${BASE_URL}/api/sessions`;

export const getMySessions = async (): Promise<SessionItem[]> => {
  const data = await apiFetch<any>(SESSIONS_API, { method: "GET" }, true);
  return Array.isArray(data) ? data : [];
};

export const getSessionDetails = async (sessionId: number | string): Promise<SessionDetails> => {
  return apiFetch<SessionDetails>(`${SESSIONS_API}/${sessionId}`, { method: "GET" }, true);
};

export const deleteSession = async (sessionId: number | string): Promise<void> => {
  return apiFetch<void>(`${SESSIONS_API}/${sessionId}`, { method: "DELETE" }, true);
};

export const startSession = async (doctorId: number, initialMessage?: string): Promise<SessionItem> => {
  return apiFetch<SessionItem>(
    SESSIONS_API,
    {
      method: "POST",
      body: JSON.stringify({
        doctorId,
        initialMessage: initialMessage?.trim() || undefined,
      }),
    },
    true
  );
};

export const sendSessionMessage = async (
  sessionId: number | string, 
  content: string, 
  type: string = "text",
  attachmentUrl?: string | null,
  fileName?: string | null
): Promise<SessionMessage> => {
  return apiFetch<SessionMessage>(
    `${SESSIONS_API}/${sessionId}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ 
        content,
        messageType: type,
        attachmentUrl,
        fileName
      }),
    },
    true
  );
};

export const startSupportSession = async (message?: string): Promise<SessionItem> => {
  return apiFetch<SessionItem>(
    `${SESSIONS_API}/support`,
    {
      method: "POST",
      body: JSON.stringify({ message }),
    },
    true
  );
};

export const parseDoctorIdFromSessionTitle = (title?: string | null): number | null => {
  if (!title) return null;
  const match = title.match(/\|d:(\d+)\|/i);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) && value > 0 ? value : null;
};
