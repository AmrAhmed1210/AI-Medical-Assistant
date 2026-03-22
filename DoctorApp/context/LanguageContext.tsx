import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { I18nManager } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { t, Lang, TranslationKey } from "../constants/i18n";

const LANG_KEY = "app_language";

function getDeviceDefaultLang(): Lang {
  try {
    const locale = Intl.DateTimeFormat().resolvedOptions().locale ?? "";
    return locale.startsWith("ar") ? "ar" : "en";
  } catch {
    return "en";
  }
}

interface LanguageContextType {
  lang: Lang;
  tr: (key: TranslationKey) => string;
  switchLanguage: (newLang: Lang) => Promise<void>;
  isRTL: boolean;
  ready: boolean;
}

const LanguageContext = createContext<LanguageContextType>({
  lang: "en",
  tr: (key) => key,
  switchLanguage: async () => {},
  isRTL: false,
  ready: false,
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(LANG_KEY).then((stored) => {
      const resolved = (stored as Lang) ?? getDeviceDefaultLang();
      setLang(resolved);
      setReady(true);
    });
  }, []);

  const switchLanguage = useCallback(async (newLang: Lang) => {
    await AsyncStorage.setItem(LANG_KEY, newLang);
    setLang(newLang);
    const shouldRTL = newLang === "ar";
    if (I18nManager.isRTL !== shouldRTL) {
      I18nManager.forceRTL(shouldRTL);
    }
  }, []);

  const tr = useCallback((key: TranslationKey) => t(lang, key), [lang]);
  const isRTL = lang === "ar";

  return (
    <LanguageContext.Provider value={{ lang, tr, switchLanguage, isRTL, ready }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}