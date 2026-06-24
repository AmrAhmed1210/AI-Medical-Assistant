import { Tabs, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { useEffect, useState } from "react";
import { ActivityIndicator, View } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function DoctorTabsLayout() {
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    let mounted = true;

    const requireDoctorSession = async () => {
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

      if (role?.toLowerCase() !== "doctor") {
        router.replace("/(patient)/home");
        return;
      }

      setAuthChecked(true);
    };

    requireDoctorSession().catch(() => {
      if (mounted) router.replace("/(auth)/login");
    });

    return () => {
      mounted = false;
    };
  }, [router]);

  if (!authChecked) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: "#999",
        tabBarStyle: {
          height: 65,
          paddingBottom: 10,
          paddingTop: 8,
          backgroundColor: '#fff',
          borderTopWidth: 1,
          borderTopColor: '#f0f0f0',
          elevation: 0,
          shadowOpacity: 0,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '500',
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="grid-outline" size={size} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="schedule"
        options={{
          title: "Schedule",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="calendar-outline" size={size} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="ai-reports"
        options={{
          title: "AI Reports",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="analytics-outline" size={size} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="profile"
        options={{
          title: "Profile",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person-outline" size={size} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="workspace"
        options={{ href: null }}
      />
      <Tabs.Screen
        name="visit-summary"
        options={{ href: null }}
      />
    </Tabs>
  );
}
