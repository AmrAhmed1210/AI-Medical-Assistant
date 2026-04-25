import { Tabs } from "expo-router";
import { usePathname } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { useEffect, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Toast from "react-native-toast-message";
import { View, StyleSheet } from "react-native";
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
import { checkIfFollowed, getAllNotificationDoctorIds, shouldReceiveDoctorNotifications } from "../../services/followService";
import { useNotificationStore } from "../../store/notificationStore";

type TabIconProps = {
  color: string;
  size: number;
  focused: boolean;
};

function TabIcon({ name, color, size, focused }: { name: any; color: string; size: number; focused: boolean }) {
  return (
    <View style={[styles.iconWrap, focused && styles.iconWrapActive]}>
      <Ionicons name={name} size={focused ? size + 1 : size} color={color} />
    </View>
  );
}

export default function TabsLayout() {
  const pathname = usePathname();
  const unreadMessages = useNotificationStore(state => state.unreadMessages);
  const clearAllMessages = useNotificationStore(state => state.clearAllMessages);
  const setLatestMessagePayload = useNotificationStore(state => state.setLatestMessagePayload);
  const incrementSessionMessage = useNotificationStore(state => state.incrementSessionMessage);

  useEffect(() => {
    let mounted = true;

    const tryConnect = async () => {
      // Wait a bit for auth to settle
      await new Promise(r => setTimeout(r, 1000));

      const token = await AsyncStorage.getItem("token");
      if (!token || !mounted) return;

      const conn = await startSignalRConnection();
      if (!conn || !mounted) return;

      try {
        const list = await getAllNotificationDoctorIds();
        await Promise.all(
          list.map((doctorId) => subscribeToDoctorSchedule(Number(doctorId)).catch(() => undefined))
        );
      } catch {
        // ignore subscription restore issues
      }

      let lastScheduleDoctorId: number | null = null;
      let lastScheduleTs = 0;
      const pushScheduleNotification = (doctorId: number, doctorName: string) => {
        const now = Date.now();
        if (lastScheduleDoctorId === doctorId && now - lastScheduleTs < 1500) return;
        lastScheduleDoctorId = doctorId;
        lastScheduleTs = now;
        addNotification(createScheduleReadyNotification(doctorName));
        Toast.show({
          type: "success",
          text1: "Schedule Updated",
          text2: `Dr. ${doctorName} has updated availability`,
          position: "top",
          topOffset: 60,
        });
      };

      const handleScheduleEvent = async (data: any) => {
        if (!mounted) return;
        const payload = data ?? {};
        const doctorId = Number(payload?.doctorId ?? payload?.DoctorId);
        const doctorName = String(payload?.doctorName ?? payload?.DoctorName ?? "Doctor");
        if (!(await shouldReceiveDoctorNotifications(doctorId))) return;
        pushScheduleNotification(doctorId, doctorName);
      };

      onAppointmentUpdated((data) => {
        if (!mounted) return;
        const status = String(data?.status ?? "").toLowerCase();
        const message = data?.message || "Appointment updated";
        const title = status === "confirmed"
          ? "Appointment Confirmed"
          : status === "cancelled"
            ? "Appointment Cancelled"
            : "Appointment Updated";
        addNotification(
          createAppointmentUpdatedNotification(
            title,
            message,
            status === "confirmed"
              ? "appointment_confirmed"
              : status === "cancelled"
                ? "appointment_cancelled"
                : "appointment_updated"
          )
        );
        Toast.show({
          type: "info",
          text1: title,
          text2: data.status ? `Status: ${data.status}` : undefined,
          position: "top",
          topOffset: 60,
        });
      });

      onScheduleReady(handleScheduleEvent);
      onScheduleUpdated(handleScheduleEvent);
      onDoctorUpdated(async (payload) => {
        if (!mounted) return;
        const doctorId = Number(payload?.doctorId ?? payload?.DoctorId);
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
      onNewMessage((payload) => {
        if (!mounted) return;
        setLatestMessagePayload(payload);
        
        const sessionId = payload?.sessionId ?? payload?.SessionId;
        if (pathname !== "/messages" && sessionId) {
          incrementSessionMessage(sessionId);
        }

        const doctorName = String(payload?.doctorName ?? payload?.DoctorName ?? "Doctor");
        const message = String(payload?.message ?? payload?.Message ?? "You received a new message.");
        addNotification(
          createAppointmentUpdatedNotification(
            "New Message",
            `Dr. ${doctorName}: ${message}`,
            "message"
          )
        );
        Toast.show({
          type: "info",
          text1: "New Message",
          text2: `Dr. ${doctorName}: ${message}`,
          position: "top",
          topOffset: 60,
        });
      });
      onNotificationReceived((payload) => {
        if (!mounted) return;
        const category = payload?.category ?? payload?.Category;
        const data = payload?.data ?? payload?.Data ?? {};
        if (category === "schedule_updated" || category === "schedule_ready") {
          const doctorId = Number(data?.doctorId ?? data?.DoctorId);
          const doctorName = String(data?.doctorName ?? data?.DoctorName ?? "Doctor");
          shouldReceiveDoctorNotifications(doctorId)
            .then((allowed) => {
              if (!allowed) return;
              pushScheduleNotification(doctorId, doctorName);
            })
            .catch(() => undefined);
          return;
        }

        if (
          category === "appointment_confirmed" ||
          category === "appointment_reminder" ||
          category === "missed_appointment" ||
          category === "rebook_offer" ||
          category === "rebook_confirmed"
        ) {
          const title = String(payload?.title ?? payload?.Title ?? "Appointment");
          const message = String(payload?.message ?? payload?.Message ?? "You have a new appointment update.");
          addNotification(createAppointmentUpdatedNotification(title, message, category));
          Toast.show({
            type: "info",
            text1: title,
            text2: message,
            position: "top",
            topOffset: 60,
          });
          return;
        }

        if (category === "message") {
          incrementSessionMessage(0); // bump unread count globally
          const title = String(payload?.title ?? payload?.Title ?? "New Message");
          const message = String(payload?.message ?? payload?.Message ?? "You received a new message.");
          addNotification(createAppointmentUpdatedNotification(title, message, "message"));
          Toast.show({
            type: "info",
            text1: title,
            text2: message,
            position: "top",
            topOffset: 60,
          });
        }
      });
    };

    tryConnect();

    return () => {
      mounted = false;
      stopSignalRConnection();
    };
  }, []);

  useEffect(() => {
    if (pathname?.includes("/messages")) {
      clearAllMessages();
    }
  }, [pathname, clearAllMessages]);

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: "#BBBBC0",
        tabBarStyle: {
          height: 70,
          paddingBottom: 10,
          paddingTop: 6,
          backgroundColor: "#fff",
          borderTopWidth: 0,
          shadowColor: "#000",
          shadowOffset: { width: 0, height: -4 },
          shadowOpacity: 0.08,
          shadowRadius: 12,
          elevation: 12,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: "600",
          marginTop: 2,
        },
      }}
    >
      <Tabs.Screen
        name="home"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "home" : "home-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="doctors"
        options={{
          title: "Find",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "search" : "search-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="messages"
        options={{
          title: "Messages",
          tabBarBadge: unreadMessages > 0 ? unreadMessages : undefined,
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "chatbubbles" : "chatbubbles-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="chatbot"
        options={{
          title: "AI Bot",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "medical" : "medical-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="profile"
        options={{
          title: "Profile",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "person" : "person-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="doctor-details"
        options={{
          href: null,
        }}
      />
      <Tabs.Screen
        name="followed-doctors"
        options={{
          href: null,
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  iconWrap: {
    width: 36,
    height: 30,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 10,
  },
  iconWrapActive: {
    backgroundColor: COLORS.primary + "18",
  },
});
