import { useEffect } from "react";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { View, ActivityIndicator } from "react-native";
import { COLORS } from "../constants/colors";

export default function Index() {
  const router = useRouter();

  useEffect(() => {
    checkSession();
  }, []);

  const checkSession = async () => {
    try {
      const isLoggedIn = await AsyncStorage.getItem("isLoggedIn");
      const role       = await AsyncStorage.getItem("userRole");

      if (isLoggedIn === "true") {
        if (role === "Doctor") {
          router.replace("/(doctor)");
        } else {
          router.replace("/(patient)/home");
        }
      } else {
        router.replace("/(auth)");
      }
    } catch {
      router.replace("/(auth)");
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: COLORS.primary }}>
      <ActivityIndicator size="large" color="#fff" />
    </View>
  );
}