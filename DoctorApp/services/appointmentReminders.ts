import AsyncStorage from "@react-native-async-storage/async-storage";
import Constants from "expo-constants";

let Notifications: any = null;
if (Constants.appOwnership !== "expo") {
  try {
    Notifications = require("expo-notifications");
  } catch {
    // expo-notifications not available
  }
}

const APPT_NOTIF_KEY = "@appointment_notifs";

function notifsAvailable(): boolean {
  return Notifications != null && Notifications.scheduleNotificationAsync != null;
}

export async function scheduleAppointmentReminders(appointments: any[]): Promise<void> {
  if (!notifsAvailable()) return;

  // Cancel old ones
  const stored = await AsyncStorage.getItem(APPT_NOTIF_KEY);
  if (stored) {
    const oldIds: string[] = JSON.parse(stored);
    for (const id of oldIds) {
      await Notifications.cancelScheduledNotificationAsync(id).catch(() => {});
    }
  }

  const newIdentifiers: string[] = [];
  const now = new Date();

  for (const appt of appointments) {
    try {
      const apptDate = new Date(appt.date);
      const [h, m] = (appt.time || "0:0").split(':').map(Number);
      
      if (isNaN(apptDate.getTime())) continue;
      
      const targetTime = new Date(apptDate);
      targetTime.setHours(h, m, 0, 0);

      if (targetTime <= now) continue;

      // 1 Hour Before
      const oneHourBefore = new Date(targetTime.getTime() - 3600000);
      if (oneHourBefore > now) {
        const id1 = await Notifications.scheduleNotificationAsync({
          content: {
            title: "📅 Appointment Reminder",
            body: `Meeting with Dr. ${appt.doctorName} in 1 hour (${appt.time})`,
            data: { appointmentId: appt.id, type: "appointment_reminder" },
            sound: "default",
          },
          trigger: { type: "date", date: oneHourBefore },
        });
        newIdentifiers.push(id1);
      }

      // Morning of (8 AM)
      const morningOf = new Date(targetTime);
      morningOf.setHours(8, 0, 0, 0);
      if (morningOf > now && morningOf < targetTime) {
        const id2 = await Notifications.scheduleNotificationAsync({
          content: {
            title: "🩺 Appointment Today",
            body: `You have an appointment today with Dr. ${appt.doctorName} at ${appt.time}`,
            data: { appointmentId: appt.id, type: "appointment_reminder" },
            sound: "default",
          },
          trigger: { type: "date", date: morningOf },
        });
        newIdentifiers.push(id2);
      }
    } catch (e) {
      console.log("Failed to schedule appt notif", e);
    }
  }

  await AsyncStorage.setItem(APPT_NOTIF_KEY, JSON.stringify(newIdentifiers));
}

export async function cancelAllAppointmentReminders(): Promise<void> {
  if (!notifsAvailable()) return;
  const stored = await AsyncStorage.getItem(APPT_NOTIF_KEY);
  if (stored) {
    const ids = JSON.parse(stored);
    for (const id of ids) {
      await Notifications.cancelScheduledNotificationAsync(id).catch(() => {});
    }
  }
  await AsyncStorage.removeItem(APPT_NOTIF_KEY);
}
