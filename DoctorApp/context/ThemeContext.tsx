import React, { createContext, useContext, useState, useEffect } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
  colors: typeof LightColors;
}

const LightColors = {
  background: "#F8FAFC",
  surface: "#FFFFFF",
  text: "#1E293B",
  textMuted: "#64748B",
  textLight: "#94A3B8",
  primary: "#059669",
  primaryLight: "#ECFDF5",
  card: "#FFFFFF",
  border: "#F1F5F9",
  statusBar: "dark-content",
  glassBackground: "rgba(255, 255, 255, 0.8)",
  cardBgGradient: ["#F8FAFC", "#FFFFFF"] as string[],
  cardBorder: "rgba(0, 0, 0, 0.05)",
};

const DarkColors = {
  background: "#090D16", // Premium dark background
  surface: "#121B2E",    // Sleek surface color
  text: "#F8FAFC",       // Clean light text
  textMuted: "#94A3B8",  // Muted light gray
  textLight: "#64748B",  // Soft gray
  primary: "#10B981",    // Vibrant emerald for dark mode
  primaryLight: "#064E3B",
  card: "#121B2E",
  border: "#1E293B",
  statusBar: "light-content",
  glassBackground: "rgba(18, 27, 46, 0.8)",
  cardBgGradient: ["#121B2E", "#17233B"] as string[],
  cardBorder: "rgba(255, 255, 255, 0.05)",
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const [theme, setTheme] = useState<Theme>('light');

  useEffect(() => {
    // Load persisted theme preference
    AsyncStorage.getItem('@user_theme').then((savedTheme) => {
      if (savedTheme === 'light' || savedTheme === 'dark') {
        setTheme(savedTheme);
      } else {
        setTheme('light'); // Unconditionally default to light mode on first launch
      }
    });
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
    AsyncStorage.setItem('@user_theme', nextTheme);
  };

  const colors = theme === 'dark' ? DarkColors : LightColors;
  const isDark = theme === 'dark';

  return (
    <ThemeContext.Provider value={{ theme, isDark, toggleTheme, colors }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
