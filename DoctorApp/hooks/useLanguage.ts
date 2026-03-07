import { useState, useEffect, useCallback } from "react";
import { I18nManager } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { t, Lang, TranslationKey } from "../constants/i18n";

const LANG_KEY = "app_language";

// Detect device language on first launch
function getDeviceDefaultLang(): Lang {
  try {
    const locale = Intl.DateTimeFormat().resolvedOptions().locale ?? "";
    return locale.startsWith("ar") ? "ar" : "en";
  } catch {
    return "en";
  }
}

export function useLanguage() {
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
    // Enable RTL for Arabic
    const shouldRTL = newLang === "ar";
    if (I18nManager.isRTL !== shouldRTL) {
      I18nManager.forceRTL(shouldRTL);
      // Note: Full RTL takes effect after app reload
    }
  }, []);

  const tr = useCallback(
    (key: TranslationKey) => t(lang, key),
    [lang]
  );

  const isRTL = lang === "ar";

  return { lang, tr, switchLanguage, isRTL, ready };
}