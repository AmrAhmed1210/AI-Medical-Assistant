import { API } from "../constants/api";
import { apiFetch } from "./http";

export interface ChatMessage {
  id: number;
  sessionId: number;
  role: "user" | "assistant" | string;
  content: string;
  timestamp: string;
}

export interface ChatSession {
  id: number;
  title: string;
  createdAt: string;
  lastMessage?: string;
  lastMessageAt?: string;
}

export const chatService = {
  getSessions: async () => {
    return apiFetch<ChatSession[]>(API.chat.sessions, { method: "GET" }, true);
  },

  getMessages: async (sessionId: number) => {
    return apiFetch<ChatMessage[]>(API.chat.messages(sessionId), { method: "GET" }, true);
  },

  ask: async (question: string, sessionId?: number) => {
    return apiFetch<{ reply: string; sessionId: number }>(
      API.chat.ask,
      {
        method: "POST",
        body: JSON.stringify({ question, sessionId }),
      },
      true
    );
  },
};
