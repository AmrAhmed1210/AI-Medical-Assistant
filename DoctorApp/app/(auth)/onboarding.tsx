import React, { useState, useRef } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity,
  SafeAreaView, Animated, StatusBar,
} from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";

const SLIDES = [
  {
    id: 1,
    emoji: "🏥",
    title: "مرحباً بك في MedBook!",
    titleEn: "Welcome to MedBook!",
    description: "شريكك الرقمي لإدارة صحتك وصحة عائلتك بكل سهولة وأمان.",
    descriptionEn: "Your digital partner to manage your health easily and securely.",
    gradient: ["#064E3B", "#059669"] as const,
  },
  {
    id: 2,
    emoji: "🤖",
    title: "مساعد طبي ذكي بالذكاء الاصطناعي",
    titleEn: "Smart AI Medical Assistant",
    description: "حلل الروشتات والتحاليل والأشعة واحصل على إرشادات فورية.",
    descriptionEn: "Analyze prescriptions, lab results, and scans for instant guidance.",
    gradient: ["#0C4A6E", "#0EA5E9"] as const,
  },
  {
    id: 3,
    emoji: "👨‍⚕️",
    title: "احجز مع أفضل الأطباء",
    titleEn: "Book Top Doctors Easily",
    description: "ابحث حسب التخصص، اقرأ مراجعات حقيقية، واحجز في ثوانٍ.",
    descriptionEn: "Search by specialty, read reviews, and book in seconds.",
    gradient: ["#4C1D95", "#8B5CF6"] as const,
  },
  {
    id: 4,
    emoji: "📋",
    title: "ملفك الصحي المتكامل في جيبك",
    titleEn: "Your Complete Health Profile",
    description: "سجّل قياساتك، حساسيتك، أدويتك وتاريخك المرضي وشاركها مع طبيبك.",
    descriptionEn: "Record vitals, allergies, medications, and share with your doctor.",
    gradient: ["#047857", "#10B981"] as const,
  },
];

export default function OnboardingScreen() {
  const router = useRouter();
  const [activeIndex, setActiveIndex] = useState(0);

  const fadeAnim = useRef(new Animated.Value(1)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(0)).current;

  const animateTransition = (nextIndex: number) => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 0, duration: 150, useNativeDriver: true }),
      Animated.timing(scaleAnim, { toValue: 0.92, duration: 150, useNativeDriver: true }),
    ]).start(() => {
      setActiveIndex(nextIndex);
      slideAnim.setValue(24);
      Animated.parallel([
        Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.timing(scaleAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.timing(slideAnim, { toValue: 0, duration: 300, useNativeDriver: true }),
      ]).start();
    });
  };

  const handleNext = () => {
    if (activeIndex < SLIDES.length - 1) {
      animateTransition(activeIndex + 1);
    } else {
      handleComplete();
    }
  };

  const handleSkip = () => handleComplete();

  const handleComplete = async () => {
    const [token, isLoggedIn, role] = await Promise.all([
      AsyncStorage.getItem("token"),
      AsyncStorage.getItem("isLoggedIn"),
      AsyncStorage.getItem("userRole"),
    ]);
    if (!token || isLoggedIn !== "true") {
      router.replace("/(auth)/login");
      return;
    }
    router.replace(role?.toLowerCase() === "doctor" ? "/(doctor)" : "/(patient)/home");
  };

  const slide = SLIDES[activeIndex];
  const isLast = activeIndex === SLIDES.length - 1;

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      <LinearGradient
        colors={slide.gradient as unknown as [string, string, ...string[]]}
        style={StyleSheet.absoluteFillObject}
        start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
      />

      {/* subtle circles */}
      <View style={[s.deco, { top: -80, right: -60, width: 220, height: 220 }]} />
      <View style={[s.deco, { bottom: 80, left: -80, width: 260, height: 260 }]} />

      <SafeAreaView style={s.safe}>

        {/* header */}
        <View style={s.header}>
          <View style={s.pill}><Text style={s.pillTxt}>{activeIndex + 1} / {SLIDES.length}</Text></View>
          <TouchableOpacity onPress={handleSkip} style={s.pill} activeOpacity={0.75}>
            <Text style={s.pillTxt}>تخطي</Text>
          </TouchableOpacity>
        </View>

        {/* content — emoji only, big and centered */}
        <Animated.View style={[s.content, {
          opacity: fadeAnim,
          transform: [{ scale: scaleAnim }, { translateY: slideAnim }],
        }]}>
          <View style={s.emojiWrap}>
            <Text style={s.emoji}>{slide.emoji}</Text>
          </View>

          <Text style={s.titleAr}>{slide.title}</Text>
          <Text style={s.titleEn}>{slide.titleEn}</Text>
          <View style={s.divider} />
          <Text style={s.descAr}>{slide.description}</Text>
          <Text style={s.descEn}>{slide.descriptionEn}</Text>
        </Animated.View>

        {/* footer */}
        <View style={s.footer}>
          <View style={s.dots}>
            {SLIDES.map((_, i) => (
              <View key={i} style={[s.dot, i === activeIndex && s.dotActive, i < activeIndex && s.dotDone]} />
            ))}
          </View>

          <TouchableOpacity style={s.nextBtn} onPress={handleNext} activeOpacity={0.85}>
            <Text style={[s.nextTxt, { color: slide.gradient[0] }]}>
              {isLast ? "ابدأ الآن • Get Started" : "التالي • Next"}
            </Text>
            <Ionicons
              name={isLast ? "checkmark-circle" : "arrow-forward"}
              size={20} color={slide.gradient[0]}
              style={{ marginLeft: 8 }}
            />
          </TouchableOpacity>
        </View>

      </SafeAreaView>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1, justifyContent: "space-between" },
  deco: { position: "absolute", borderRadius: 999, backgroundColor: "rgba(255,255,255,0.07)" },

  header: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", paddingHorizontal: 24, paddingTop: 16,
  },
  pill: {
    paddingVertical: 6, paddingHorizontal: 14,
    borderRadius: 20, backgroundColor: "rgba(255,255,255,0.2)",
  },
  pillTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },

  content: {
    flex: 1, alignItems: "center",
    justifyContent: "center", paddingHorizontal: 32,
  },
  // big circular emoji container — clean, no icon card on top
  emojiWrap: {
    width: 150, height: 150, borderRadius: 75,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderWidth: 2, borderColor: "rgba(255,255,255,0.25)",
    justifyContent: "center", alignItems: "center",
    marginBottom: 36,
    shadowColor: "#000", shadowOpacity: 0.15,
    shadowRadius: 20, shadowOffset: { width: 0, height: 10 },
    elevation: 8,
  },
  emoji: { fontSize: 72 },

  titleAr: {
    fontSize: 26, fontWeight: "800", color: "#fff",
    textAlign: "center", lineHeight: 38,
  },
  titleEn: {
    fontSize: 15, color: "rgba(255,255,255,0.75)",
    fontWeight: "600", textAlign: "center", marginTop: 6,
  },
  divider: {
    width: 44, height: 3,
    backgroundColor: "rgba(255,255,255,0.35)",
    borderRadius: 2, marginVertical: 22,
  },
  descAr: {
    fontSize: 15, color: "#fff",
    textAlign: "center", lineHeight: 26, paddingHorizontal: 8,
  },
  descEn: {
    fontSize: 13, color: "rgba(255,255,255,0.7)",
    textAlign: "center", lineHeight: 20,
    marginTop: 8, paddingHorizontal: 8,
  },

  footer: { paddingHorizontal: 28, paddingBottom: 44, alignItems: "center" },
  dots: { flexDirection: "row", gap: 8, marginBottom: 24 },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: "rgba(255,255,255,0.3)" },
  dotActive: { width: 28, backgroundColor: "#fff" },
  dotDone: { backgroundColor: "rgba(255,255,255,0.6)" },

  nextBtn: {
    width: "100%", height: 58, backgroundColor: "#fff",
    borderRadius: 20, flexDirection: "row",
    justifyContent: "center", alignItems: "center",
    shadowColor: "#000", shadowOpacity: 0.12,
    shadowRadius: 16, shadowOffset: { width: 0, height: 6 }, elevation: 8,
  },
  nextTxt: { fontSize: 16, fontWeight: "800" },
});