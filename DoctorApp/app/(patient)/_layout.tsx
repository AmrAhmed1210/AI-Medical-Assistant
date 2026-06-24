/**
 * _layout.tsx  —  Patient tab navigator
 *
 * Responsibilities:
 * 1. Bootstrap the SignalR connection once auth is ready
 * 2. Register all real-time event handlers (appointments, schedules, messages)
 * 3. Render the Tabs navigator using our premium CustomTabBar
 *
 * The actual visual tab bar lives in components/TabBar/TabBar.tsx to keep
 * this layout file focused on routing and data concerns.
 */

import { Tabs, usePathname, useRouter } from "expo-router";
import { useEffect, useState } from "react";
import { ActivityIndicator, View } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Toast from "react-native-toast-message";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { CustomTabBar } from "../../components/TabBar/TabBar";
import {
  addNotification,
  createAppointmentUpdatedNotification,
  createScheduleReadyNotification,
} from "../../services/notificationService";
import {
  onDoctorUpdated,
  onAppointmentUpdated,
  onNewMessage,
  onNotificationReceived,
  onScheduleReady,
  onScheduleUpdated,
  subscribeToDoctorSchedule,
  startSignalRConnection,
  stopSignalRConnection,
} from "../../services/signalr";
import {
  getAllNotificationDoctorIds,
  shouldReceiveDoctorNotifications,
} from "../../services/followService";
import { useNotificationStore } from "../../store/notificationStore";

// ─── Types ────────────────────────────────────────────────────────────────────

/** Prevent duplicate schedule toasts within a short burst window */
interface ScheduleDedupeState {
  lastDoctorId: number | null;
  lastTs: number;
}

// ─── Layout ───────────────────────────────────────────────────────────────────

export default function TabsLayout() {
  const pathname = usePathname();
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState(false);
  const unreadMessages      = useNotificationStore((s) => s.unreadMessages);
  const clearAllMessages    = useNotificationStore((s) => s.clearAllMessages);
  const setLatestMsgPayload = useNotificationStore((s) => s.setLatestMessagePayload);
  const incrementSession    = useNotificationStore((s) => s.incrementSessionMessage);

  const insets = useSafeAreaInsets();

  useEffect(() => {
    let mounted = true;

    const requirePatientSession = async () => {
      const [token, isLoggedIn, role] = await Promise.all([
        AsyncStorage.getItem("token"),
        AsyncStorage.getItem("isLoggedIn"),
        AsyncStorage.getItem("userRole"),
      ]);

      if (!mounted) return;

      if (!token || isLoggedIn !== "true") {
        router.replace("/(auth)/login");
        return;
      }

      if (role?.toLowerCase() === "doctor") {
        router.replace("/(doctor)");
        return;
      }

      setAuthChecked(true);
    };

    requirePatientSession().catch(() => {
      if (mounted) router.replace("/(auth)/login");
    });

    return () => {
      mounted = false;
    };
  }, [router]);

  // ── SignalR bootstrap ──────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;

    const tryConnect = async () => {
      // Small delay — lets the auth token settle after login navigation
      await new Promise<void>((r) => setTimeout(r, 1000));

      const token = await AsyncStorage.getItem("token");
      if (!token || !mounted) return;

      const conn = await startSignalRConnection();
      if (!conn || !mounted) return;

      // Restore all doctor schedule subscriptions from local storage
      try {
        const ids = await getAllNotificationDoctorIds();
        await Promise.all(
          ids.map((id) =>
            subscribeToDoctorSchedule(Number(id)).catch(() => undefined)
          )
        );
      } catch {
        // Non-critical — subscriptions will be re-added on next follow
      }

      // ── Deduplication for schedule toasts ──
      const dedup: ScheduleDedupeState = { lastDoctorId: null, lastTs: 0 };

      const pushScheduleToast = (doctorId: number, doctorName: string) => {
        const now = Date.now();
        if (dedup.lastDoctorId === doctorId && now - dedup.lastTs < 1500) return;
        dedup.lastDoctorId = doctorId;
        dedup.lastTs = now;
        addNotification(createScheduleReadyNotification(doctorName));
        Toast.show({
          type: "success",
          text1: "Schedule Updated",
          text2: `Dr. ${doctorName} has updated availability`,
          position: "top",
          topOffset: 60,
        });
      };

      // ── Schedule event handler (reused for Ready + Updated) ──
      const handleScheduleEvent = async (data: any) => {
        if (!mounted) return;
        const doctorId   = Number(data?.doctorId   ?? data?.DoctorId);
        const doctorName = String(data?.doctorName  ?? data?.DoctorName  ?? "Doctor");
        if (!(await shouldReceiveDoctorNotifications(doctorId))) return;
        pushScheduleToast(doctorId, doctorName);
      };

      // ── Appointment status updates ──
      onAppointmentUpdated((data) => {
        if (!mounted) return;
        const status = String(data?.status ?? "").toLowerCase();
        const message = data?.message || "Appointment updated";
        const title =
          status === "confirmed" ? "Appointment Confirmed"
          : status === "cancelled" ? "Appointment Cancelled"
          : "Appointment Updated";
        const type =
          status === "confirmed" ? "appointment_confirmed"
          : status === "cancelled" ? "appointment_cancelled"
          : "appointment_updated";

        addNotification(createAppointmentUpdatedNotification(title, message, type));
        Toast.show({
          type: "info",
          text1: title,
          text2: data?.status ? `Status: ${data.status}` : undefined,
          position: "top",
          topOffset: 60,
        });
      });

      onScheduleReady(handleScheduleEvent);
      onScheduleUpdated(handleScheduleEvent);

      // ── Doctor profile update ──
      onDoctorUpdated(async (payload) => {
        if (!mounted) return;
        const doctorId   = Number(payload?.doctorId  ?? payload?.DoctorId);
        const doctorName = String(payload?.doctorName ?? payload?.DoctorName ?? "Doctor");
        if (!(await shouldReceiveDoctorNotifications(doctorId))) return;

        await addNotification({
          id: `doctor_update_${doctorId}_${Date.now()}`,
          type: "update",
          icon: "👨‍⚕️",
          title: "👨‍⚕️ Doctor Updated",
          message: `Dr. ${doctorName} updated their profile`,
          timestamp: Date.now(),
          doctorId,
          doctorName,
        });

        Toast.show({
          type: "info",
          text1: "Doctor Updated",
          text2: `Dr. ${doctorName} updated their profile`,
          position: "top",
          topOffset: 60,
        });
      });

      // ── Incoming messages ──
      onNewMessage((payload) => {
        if (!mounted) return;
        setLatestMsgPayload(payload);

        const sessionId = payload?.sessionId ?? payload?.SessionId;
        // Only increment unread badge when the messages screen is NOT open
        if (pathname !== "/messages" && sessionId) {
          incrementSession(sessionId);
        }

        const doctorName = String(payload?.doctorName ?? payload?.DoctorName ?? "Doctor");
        const message    = String(payload?.message    ?? payload?.Message    ?? "You received a new message.");
        addNotification(
          createAppointmentUpdatedNotification("New Message", `Dr. ${doctorName}: ${message}`, "message")
        );
        Toast.show({
          type: "info",
          text1: "New Message",
          text2: `Dr. ${doctorName}: ${message}`,
          position: "top",
          topOffset: 60,
        });
      });

      // ── Generic notification hub events ──
      onNotificationReceived((payload) => {
        if (!mounted) return;
        const category = payload?.category ?? payload?.Category;
        const data     = payload?.data     ?? payload?.Data ?? {};

        if (category === "schedule_updated" || category === "schedule_ready") {
          const doctorId   = Number(data?.doctorId   ?? data?.DoctorId);
          const doctorName = String(data?.doctorName ?? data?.DoctorName ?? "Doctor");
          shouldReceiveDoctorNotifications(doctorId)
            .then((ok) => ok && pushScheduleToast(doctorId, doctorName))
            .catch(() => undefined);
          return;
        }

        const APPOINTMENT_CATEGORIES = new Set([
          "appointment_confirmed",
          "appointment_reminder",
          "missed_appointment",
          "rebook_offer",
          "rebook_confirmed",
        ]);

        if (APPOINTMENT_CATEGORIES.has(category)) {
          const title   = String(payload?.title   ?? payload?.Title   ?? "Appointment");
          const message = String(payload?.message ?? payload?.Message ?? "You have a new appointment update.");
          addNotification(createAppointmentUpdatedNotification(title, message, category));
          Toast.show({ type: "info", text1: title, text2: message, position: "top", topOffset: 60 });
          return;
        }

        if (category === "message") {
          incrementSession(0);
          const title   = String(payload?.title   ?? payload?.Title   ?? "New Message");
          const message = String(payload?.message ?? payload?.Message ?? "You received a new message.");
          addNotification(createAppointmentUpdatedNotification(title, message, "message"));
          Toast.show({ type: "info", text1: title, text2: message, position: "top", topOffset: 60 });
        }
      });
    };

    tryConnect();

    return () => {
      mounted = false;
      stopSignalRConnection();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Clear message badge when user navigates to messages ──
  useEffect(() => {
    if (pathname?.includes("/messages")) {
      clearAllMessages();
    }
  }, [pathname, clearAllMessages]);

  // ── Render ────────────────────────────────────────────────────────────────
  if (!authChecked) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <Tabs
      screenOptions={{ headerShown: false }}
      tabBar={(props: any) => (
        <CustomTabBar
          {...props}
          insets={insets}
          unreadCount={unreadMessages}
        />
      )}
    >
      {/* ── Visible tabs ── */}
      <Tabs.Screen name="home"     options={{ title: "Home" }} />
      <Tabs.Screen name="doctors"  options={{ title: "Find" }} />
      <Tabs.Screen name="messages" options={{ title: "Chat" }} />
      <Tabs.Screen name="chatbot"  options={{ title: "AI Bot" }} />
      <Tabs.Screen name="profile"  options={{ title: "Profile" }} />

      {/* ── Detail screens — hidden from tab bar ── */}
      <Tabs.Screen name="vitals"          options={{ href: null }} />
      <Tabs.Screen name="medications"     options={{ href: null }} />
      <Tabs.Screen name="doctor-details"  options={{ href: null }} />
      <Tabs.Screen name="followed-doctors" options={{ href: null }} />
      <Tabs.Screen name="visit-summary"   options={{ href: null }} />
      <Tabs.Screen name="ai-profile-assistant" options={{ href: null }} />
      <Tabs.Screen name="medical-records/index" options={{ href: null }} />
    </Tabs>
  );
}
