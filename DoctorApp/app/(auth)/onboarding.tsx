import React, { useState, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Dimensions,
  Animated,
} from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";

const { width, height } = Dimensions.get("window");

const ONBOARDING_SLIDES = [
  {
    id: 1,
    emoji: "🏥",
    title: "مرحباً بك في MedBook!",
    titleEn: "Welcome to MedBook!",
    description: "شريكك الرقمي لإدارة صحتك وصحة عائلتك بكل سهولة وأمان.",
    descriptionEn: "Your digital partner to manage your health easily and securely.",
    gradient: ["#064E3B", "#059669"] as const,
    accentColor: "#10B981",
  },
  {
    id: 2,
    emoji: "🤖",
    title: "مساعد طبي ذكي بالذكاء الاصطناعي",
    titleEn: "Smart AI Medical Assistant",
    description: "حلل الروشتات والتحاليل والأشعة واحصل على إرشادات فورية.",
    descriptionEn: "Analyze prescriptions, lab results, and scans for instant guidance.",
    gradient: ["#0C4A6E", "#0EA5E9"] as const,
    accentColor: "#38BDF8",
  },
  {
    id: 3,
    emoji: "👨‍⚕️",
    title: "احجز مع أفضل الأطباء",
    titleEn: "Book Top Doctors Easily",
    description: "ابحث حسب التخصص، اقرأ مراجعات حقيقية، واحجز في ثوانٍ.",
    descriptionEn: "Search by specialty, read reviews, and book in seconds.",
    gradient: ["#4C1D95", "#8B5CF6"] as const,
    accentColor: "#A78BFA",
  },
  {
    id: 4,
    emoji: "📋",
    title: "ملفك الصحي المتكامل في جيبك",
    titleEn: "Your Complete Health Profile",
    description: "سجّل قياساتك، حساسيتك، أدويتك وتاريخك المرضي وشاركها مع طبيبك.",
    descriptionEn: "Record vitals, allergies, medications, and share with your doctor.",
    gradient: ["#047857", "#10B981"] as const,
    accentColor: "#34D399",
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
      Animated.timing(scaleAnim, { toValue: 0.8, duration: 150, useNativeDriver: true }),
    ]).start(() => {
      setActiveIndex(nextIndex);
      slideAnim.setValue(30);
      Animated.parallel([
        Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.timing(scaleAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.timing(slideAnim, { toValue: 0, duration: 300, useNativeDriver: true }),
      ]).start();
    });
  };

  const handleNext = () => {
    if (activeIndex < ONBOARDING_SLIDES.length - 1) {
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

  const currentSlide = ONBOARDING_SLIDES[activeIndex];
  const isLast = activeIndex === ONBOARDING_SLIDES.length - 1;

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={currentSlide.gradient as unknown as [string, string, ...string[]]}
        style={StyleSheet.absoluteFillObject}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      />

      {/* Decorative bubbles */}
      <View style={[styles.bubble, { top: -60, right: -40, width: 200, height: 200, backgroundColor: 'rgba(255,255,255,0.06)' }]} />
      <View style={[styles.bubble, { bottom: 100, left: -70, width: 250, height: 250, backgroundColor: 'rgba(255,255,255,0.04)' }]} />
      <View style={[styles.bubble, { top: height * 0.35, right: -30, width: 120, height: 120, backgroundColor: 'rgba(255,255,255,0.08)' }]} />

      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.stepIndicator}>
            <Text style={styles.stepText}>{activeIndex + 1}/{ONBOARDING_SLIDES.length}</Text>
          </View>
          <TouchableOpacity onPress={handleSkip} activeOpacity={0.7} style={styles.skipButton}>
            <Text style={styles.skipText}>تخطي</Text>
          </TouchableOpacity>
        </View>

        {/* Content */}
        <Animated.View style={[
          styles.contentContainer,
          {
            opacity: fadeAnim,
            transform: [
              { scale: scaleAnim },
              { translateY: slideAnim },
            ],
          },
        ]}>
          {/* Big emoji */}
          <View style={styles.emojiContainer}>
            <View style={styles.emojiGlow} />
            <Text style={styles.emoji}>{currentSlide.emoji}</Text>
          </View>

          {/* Text */}
          <View style={styles.textContainer}>
            <Text style={styles.titleAr}>{currentSlide.title}</Text>
            <Text style={styles.titleEn}>{currentSlide.titleEn}</Text>

            <View style={styles.divider} />

            <Text style={styles.descriptionAr}>{currentSlide.description}</Text>
            <Text style={styles.descriptionEn}>{currentSlide.descriptionEn}</Text>
          </View>
        </Animated.View>

        {/* Footer */}
        <View style={styles.footer}>
          {/* Progress dots */}
          <View style={styles.indicatorContainer}>
            {ONBOARDING_SLIDES.map((_, index) => (
              <View
                key={index}
                style={[
                  styles.indicatorDot,
                  index === activeIndex && styles.activeIndicatorDot,
                  index < activeIndex && styles.completedDot,
                ]}
              />
            ))}
          </View>

          {/* Next / Get Started Button */}
          <TouchableOpacity
            style={styles.nextButton}
            onPress={handleNext}
            activeOpacity={0.85}
          >
            <Text style={[styles.nextButtonText, { color: currentSlide.gradient[0] }]}>
              {isLast ? "ابدأ الآن • Get Started" : "التالي • Next"}
            </Text>
            <Ionicons
              name={isLast ? "checkmark-circle" : "arrow-forward"}
              size={20}
              color={currentSlide.gradient[0]}
              style={{ marginLeft: 8 }}
            />
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  bubble: {
    position: "absolute",
    borderRadius: 999,
  },
  safeArea: {
    flex: 1,
    justifyContent: "space-between",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 24,
    paddingTop: 15,
  },
  stepIndicator: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
  },
  stepText: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "700",
  },
  skipButton: {
    paddingVertical: 8,
    paddingHorizontal: 18,
    borderRadius: 20,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
  },
  skipText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
  },
  contentContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 30,
  },
  emojiContainer: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: "rgba(255, 255, 255, 0.12)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 40,
    borderWidth: 2,
    borderColor: "rgba(255, 255, 255, 0.2)",
  },
  emojiGlow: {
    position: "absolute",
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: "rgba(255, 255, 255, 0.06)",
  },
  emoji: {
    fontSize: 72,
  },
  textContainer: {
    alignItems: "center",
  },
  titleAr: {
    fontSize: 24,
    fontWeight: "800",
    color: "#fff",
    textAlign: "center",
    lineHeight: 36,
  },
  titleEn: {
    fontSize: 16,
    color: "rgba(255, 255, 255, 0.75)",
    fontWeight: "600",
    textAlign: "center",
    marginTop: 6,
    letterSpacing: 0.3,
  },
  divider: {
    width: 50,
    height: 3,
    backgroundColor: "rgba(255, 255, 255, 0.35)",
    borderRadius: 2,
    marginVertical: 20,
  },
  descriptionAr: {
    fontSize: 16,
    color: "#fff",
    textAlign: "center",
    lineHeight: 26,
    paddingHorizontal: 10,
  },
  descriptionEn: {
    fontSize: 13,
    color: "rgba(255, 255, 255, 0.7)",
    textAlign: "center",
    lineHeight: 20,
    marginTop: 8,
    paddingHorizontal: 10,
  },
  footer: {
    paddingHorizontal: 30,
    paddingBottom: 45,
    alignItems: "center",
  },
  indicatorContainer: {
    flexDirection: "row",
    justifyContent: "center",
    marginBottom: 25,
    gap: 8,
  },
  indicatorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
  },
  activeIndicatorDot: {
    width: 28,
    backgroundColor: "#fff",
  },
  completedDot: {
    backgroundColor: "rgba(255, 255, 255, 0.6)",
  },
  nextButton: {
    width: "100%",
    height: 58,
    backgroundColor: "#fff",
    borderRadius: 20,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    elevation: 8,
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 15,
    shadowOffset: { width: 0, height: 6 },
  },
  nextButtonText: {
    fontSize: 16,
    fontWeight: "800",
  },
});
