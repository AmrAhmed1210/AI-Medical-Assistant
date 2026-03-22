import { Stack } from "expo-router";
import Toast from "react-native-toast-message";
import { LanguageProvider } from "../context/LanguageContext";

export default function RootLayout() {
  return (
    <LanguageProvider>
      <Stack screenOptions={{ headerShown: false }} />
      <Toast />
    </LanguageProvider>
  );
}