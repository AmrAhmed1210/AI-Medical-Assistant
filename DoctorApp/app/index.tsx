import { useEffect } from "react";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { View, ActivityIndicator, Text } from "react-native";
import { COLORS } from "../constants/colors";

export default function Index() {
  const router = useRouter();

  useEffect(() => {
    const restoreSession = async () => {
      try {
        // Wait for AsyncStorage to be ready (important!)
        await new Promise(r => setTimeout(r, 1500));

        // Get all session data
        const [token, isLoggedIn, role] = await Promise.all([
          AsyncStorage.getItem('token'),
          AsyncStorage.getItem('isLoggedIn'),
          AsyncStorage.getItem('userRole'),
        ]);

        console.log("🔐 Session check:", { hasToken: !!token, isLoggedIn, role });

        // Require BOTH token AND isLoggedIn flag
        if (token && isLoggedIn === 'true') {
          console.log("✅ User is logged in as:", role || "Patient");
          if (role === 'Doctor') {
            router.replace('/(doctor)');
          } else {
            router.replace('/(patient)/home');
          }
        } else {
          console.log("❌ No valid session, redirecting to login");
          router.replace('/(auth)/login');
        }
      } catch (e) {
        console.error("❌ Session restore failed:", e);
        router.replace('/(auth)/login');
      }
    };

    restoreSession();
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: COLORS.primary }}>
      <ActivityIndicator size="large" color="#fff" />
      <Text style={{ color: "#fff", marginTop: 16, fontSize: 14, opacity: 0.8 }}>
        Loading...
      </Text>
    </View>
  );
}