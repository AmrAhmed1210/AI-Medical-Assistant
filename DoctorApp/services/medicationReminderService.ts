import AsyncStorage from "@react-native-async-storage/async-storage";

let Notifications: any = null;
try {
  Notifications = require("expo-notifications");
  Notifications.setNotificationHandler?.({
    handleNotification: async () => ({
      shouldShowAlert: true,
      shouldPlaySound: true,
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true,
    }),
  });
} catch {
  // expo-notifications not available (e.g. Expo Go SDK 53+)
}

const NOTIF_KEY = "@medication_notifs";

function notifsAvailable(): boolean {
  return Notifications != null && Notifications.scheduleNotificationAsync != null;
}

export async function requestNotificationPermissions(): Promise<boolean> {
  if (!notifsAvailable()) return false;
  const { status } = await Notifications.requestPermissionsAsync();
  return status === "granted";
}

export async function scheduleMedicationReminders(
  medicationId: number,
  medicationName: string,
  dosage: string,
  doseTimes: string,
  daysOfWeek: string,
  startDateStr: string,
  endDateStr?: string,
  daysAhead = 7
): Promise<void> {
  if (!notifsAvailable()) return;

  // Cancel existing for this med
  await cancelMedicationReminders(medicationId);

  const days = daysOfWeek?.split(",").map(d => d.trim()).filter(Boolean) ?? [];
  if (days.length === 0) return;

  const times = doseTimes?.split(",").map(t => t.trim()).filter(Boolean) ?? [];
  if (times.length === 0) return;

  const start = new Date(startDateStr);
  const end = endDateStr ? new Date(endDateStr) : new Date(Date.now() + daysAhead * 86400000);
  const maxEnd = new Date(Date.now() + daysAhead * 86400000);
  const limit = end < maxEnd ? end : maxEnd;

  const identifiers: string[] = [];

  for (let d = new Date(); d <= limit; d.setDate(d.getDate() + 1)) {
    const dayName = d.toLocaleDateString("en-US", { weekday: "long" });
    if (!days.includes(dayName)) continue;

    for (const timeStr of times) {
      const [h, m] = timeStr.split(":").map(Number);
      if (isNaN(h) || isNaN(m)) continue;

      const trigger = new Date(d);
      trigger.setHours(h, m, 0, 0);
      if (trigger <= new Date()) continue; // skip past

      const id = await Notifications.scheduleNotificationAsync({
        content: {
          title: `💊 Time for ${medicationName}`,
          body: `${dosage} — Don't forget your dose!`,
          data: { medicationId, type: "medication_reminder" },
          sound: "default",
        },
        trigger: { type: Notifications.SchedulableTriggerInputTypes?.DATE ?? "date", date: trigger },
      });
      identifiers.push(id);
    }
  }

  // Save identifiers
  const stored = await AsyncStorage.getItem(NOTIF_KEY);
  const map: Record<string, string[]> = stored ? JSON.parse(stored) : {};
  map[String(medicationId)] = identifiers;
  await AsyncStorage.setItem(NOTIF_KEY, JSON.stringify(map));
}

export async function cancelMedicationReminders(medicationId: number): Promise<void> {
  if (!notifsAvailable()) return;
  const stored = await AsyncStorage.getItem(NOTIF_KEY);
  if (!stored) return;
  const map: Record<string, string[]> = JSON.parse(stored);
  const ids = map[String(medicationId)] ?? [];
  for (const id of ids) {
    await Notifications.cancelScheduledNotificationAsync(id);
  }
  delete map[String(medicationId)];
  await AsyncStorage.setItem(NOTIF_KEY, JSON.stringify(map));
}

export async function cancelAllMedicationReminders(): Promise<void> {
  if (!notifsAvailable()) return;
  await Notifications.cancelAllScheduledNotificationsAsync();
  await AsyncStorage.removeItem(NOTIF_KEY);
}

export async function getScheduledNotificationCount(): Promise<number> {
  if (!notifsAvailable()) return 0;
  return (await Notifications.getAllScheduledNotificationsAsync()).length;
}
