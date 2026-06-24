import AsyncStorage from "@react-native-async-storage/async-storage";
import { BASE_URL } from "../constants/api";
import { apiFetch } from "./http";

export const getFollowKey = async (doctorId: number) => {
  const patientId = await AsyncStorage.getItem("patientId") || await AsyncStorage.getItem("userId") || 'guest';
  return `follow_${patientId}_doctor_${doctorId}`;
};

export const checkIfFollowed = async (doctorId: number): Promise<boolean> => {
  if (!Number.isFinite(doctorId) || doctorId <= 0) return false;
  try {
    const followed = await getFollowedDoctorIds();
    return followed.includes(doctorId);
  } catch {
    // Fall through to local cache below.
  }

  const key = await getFollowKey(doctorId);
  const value = await AsyncStorage.getItem(key);
  return value === "true";
};

export const setFollowed = async (doctorId: number, followed: boolean): Promise<void> => {
  if (!Number.isFinite(doctorId) || doctorId <= 0) return;
  try {
    await apiFetch(
      followed
        ? `${BASE_URL}/api/users/follow/${doctorId}`
        : `${BASE_URL}/api/users/unfollow/${doctorId}`,
      { method: followed ? "POST" : "DELETE" },
      true
    );
  } catch {
    // Keep the UI responsive offline; local state is still scoped per patient.
  }

  const key = await getFollowKey(doctorId);
  await AsyncStorage.setItem(key, followed ? "true" : "false");
};

export const toggleFollowed = async (doctorId: number): Promise<boolean> => {
  const current = await checkIfFollowed(doctorId);
  const next = !current;
  await setFollowed(doctorId, next);
  return next;
};

export const getFollowedDoctorIds = async (): Promise<number[]> => {
  const patientId = await AsyncStorage.getItem("patientId") || await AsyncStorage.getItem("userId") || 'guest';
  const prefix = `follow_${patientId}_doctor_`;
  try {
    const doctors = await apiFetch<Array<{ id?: number | string }>>(
      `${BASE_URL}/api/users/followed-doctors`,
      { method: "GET" },
      true
    );
    if (!Array.isArray(doctors)) {
      throw new Error("Invalid followed doctors response");
    }

    const keys = await AsyncStorage.getAllKeys();
    const existingFollowKeys = keys.filter((key) => key.startsWith(prefix));
    if (existingFollowKeys.length > 0) {
      await AsyncStorage.multiRemove(existingFollowKeys);
    }

    const serverIds = doctors
      .map((doctor) => Number(doctor.id))
      .filter((id) => Number.isFinite(id) && id > 0);

    if (serverIds.length > 0) {
      await AsyncStorage.multiSet(serverIds.map((id) => [`${prefix}${id}`, "true"]));
    }
    return serverIds;
  } catch {
    // Fall back to locally cached follows.
  }
  
  const keys = await AsyncStorage.getAllKeys();
  const followKeys = keys.filter((key) => key.startsWith(prefix));
  if (followKeys.length === 0) return [];

  const entries = await AsyncStorage.multiGet(followKeys);
  return entries
    .filter(([, value]) => value === "true")
    .map(([key]) => Number(key.replace(prefix, "")))
    .filter((id) => Number.isFinite(id) && id > 0);
};
export const getSubscriptionKey = async (doctorId: number) => {
  const patientId = await AsyncStorage.getItem("patientId") || await AsyncStorage.getItem("userId") || 'guest';
  return `notify_sub_${patientId}_doctor_${doctorId}`;
};

export const checkIfSubscribed = async (doctorId: number): Promise<boolean> => {
  if (!Number.isFinite(doctorId) || doctorId <= 0) return false;
  const key = await getSubscriptionKey(doctorId);
  const value = await AsyncStorage.getItem(key);
  return value === "true";
};

export const setSubscribed = async (doctorId: number, subscribed: boolean): Promise<void> => {
  if (!Number.isFinite(doctorId) || doctorId <= 0) return;
  const key = await getSubscriptionKey(doctorId);
  await AsyncStorage.setItem(key, subscribed ? "true" : "false");
};

export const getSubscribedDoctorIds = async (): Promise<number[]> => {
  const patientId = await AsyncStorage.getItem("patientId") || await AsyncStorage.getItem("userId") || 'guest';
  const prefix = `notify_sub_${patientId}_doctor_`;
  
  const keys = await AsyncStorage.getAllKeys();
  const subKeys = keys.filter((key) => key.startsWith(prefix));
  if (subKeys.length === 0) return [];

  const entries = await AsyncStorage.multiGet(subKeys);
  return entries
    .filter(([, value]) => value === "true")
    .map(([key]) => Number(key.replace(prefix, "")))
    .filter((id) => Number.isFinite(id) && id > 0);
};

export const getAllNotificationDoctorIds = async (): Promise<number[]> => {
  const follows = await getFollowedDoctorIds();
  const subs = await getSubscribedDoctorIds();
  return Array.from(new Set([...follows, ...subs]));
};

export const shouldReceiveDoctorNotifications = async (doctorId: number): Promise<boolean> => {
  const [followed, subscribed] = await Promise.all([
    checkIfFollowed(doctorId),
    checkIfSubscribed(doctorId)
  ]);
  return followed || subscribed;
};
