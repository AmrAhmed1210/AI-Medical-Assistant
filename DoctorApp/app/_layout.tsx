import { Stack } from "expo-router";
import Toast from "react-native-toast-message";
import { LanguageProvider } from "../context/LanguageContext";
import { ThemeProvider } from "../context/ThemeContext";

export default function RootLayout() {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <Stack screenOptions={{ headerShown: false }} />
        <Toast />
      </LanguageProvider>
    </ThemeProvider>
  );
}