import { API } from "../constants/api";
import { apiFetch } from "./http";
import { getSessionDetails } from "./sessionService";

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
    return apiFetch<ChatSession[]>(API.sessions.list, { method: "GET" }, true);
  },

  getMessages: async (sessionId: number) => {
    const session = await getSessionDetails(sessionId);
    return (session.messages ?? []).map((m) => ({
      id: m.id,
      sessionId: m.sessionId,
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
    })) as ChatMessage[];
  },

  ask: async (question: string, sessionId?: number, history?: ChatMessage[]) => {
    return apiFetch<{ reply: string; sessionId?: number }>(
      API.chat.ask,
      {
        method: "POST",
        body: JSON.stringify({
          question,
          sessionId,
          history: history?.map((message) => ({
            role: message.role === "assistant" ? "assistant" : "user",
            content: message.content,
          })),
        }),
      },
      true
    );
  },

  analyzeImage: async (imageUri: string, sessionId?: number) => {
    const formData = new FormData();
    const filename = imageUri.split("/").pop() || "image.jpg";
    const match = /\.(\w+)$/.exec(filename);
    const mimeType = match ? `image/${match[1]}` : "image/jpeg";

    formData.append("file", { uri: imageUri, name: filename, type: mimeType } as any);
    if (sessionId) formData.append("SessionId", sessionId.toString());

    return apiFetch<unknown>(
      API.chat.analyzeImage,
      {
        method: "POST",
        body: formData,
        headers: { "Content-Type": "multipart/form-data" },
      },
      true
    );
  },
};
