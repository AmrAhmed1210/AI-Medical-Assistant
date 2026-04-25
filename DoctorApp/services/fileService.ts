import { BASE_URL } from "../constants/api";
import { apiFetch } from "./http";

export const uploadMessageFile = async (uri: string, filename?: string): Promise<{ url: string; fileName: string }> => {
  const formData = new FormData();
  const name = filename || uri.split("/").pop() || "file";
  const match = /\.(\w+)$/.exec(name);
  const type = match ? `application/${match[1]}` : `application/octet-stream`;

  // For images, use image/jpeg or similar if possible
  const finalType = name.match(/\.(jpg|jpeg|png|gif)$/i) 
    ? `image/${match ? match[1] : 'jpeg'}` 
    : type;

  formData.append("file", { uri, name, type: finalType } as any);

  return apiFetch<any>(`${BASE_URL}/api/profile/upload`, {
    method: "POST",
    body: formData,
  }, true);
};
