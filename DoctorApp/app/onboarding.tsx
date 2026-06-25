import React, { useState, useRef, useEffect } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, ActivityIndicator, StatusBar, Animated, Easing,
} from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../constants/colors";
import { useLanguage } from "../context/LanguageContext";
import { getMyPatientId } from "../services/authService";
import { createAllergy, createSurgery } from "../services/medicalRecordService";
import Toast from "react-native-toast-message";

// ─── palette – same tokens as login / register ────────────────────────────
const C = {
  primary: "#059669",
  primaryLight: "#10B981",
  bg: "#F8FAFC",
  surface: "#FFFFFF",
  text: "#0F172A",
  textMuted: "#64748B",
  border: "#E2E8F0",
};

// Step meta – icon + gradient accent per step
const STEP_META = [
  { icon: "fitness", color: "#059669", label: "Allergies" },
  { icon: "heart", color: "#0EA5E9", label: "Chronic" },
  { icon: "medkit", color: "#8B5CF6", label: "Medication" },
  { icon: "cut", color: "#F59E0B", label: "Surgery" },
] as const;

export default function OnboardingScreen() {
  const router = useRouter();
  const { tr } = useLanguage();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [focusedInput, setFocusedInput] = useState<string | null>(null);

  // ── entry animation ──────────────────────────────────────────────────────
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(32)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  }, []);

  // step transition animation
  const stepFade = useRef(new Animated.Value(1)).current;
  const stepSlide = useRef(new Animated.Value(0)).current;

  const animateStep = (next: number) => {
    Animated.parallel([
      Animated.timing(stepFade, { toValue: 0, duration: 140, useNativeDriver: true }),
      Animated.timing(stepSlide, { toValue: -20, duration: 140, useNativeDriver: true }),
    ]).start(() => {
      setStep(next);
      stepSlide.setValue(20);
      Animated.parallel([
        Animated.timing(stepFade, { toValue: 1, duration: 280, useNativeDriver: true }),
        Animated.timing(stepSlide, { toValue: 0, duration: 280, useNativeDriver: true }),
      ]).start();
    });
  };

  // ── form state ───────────────────────────────────────────────────────────
  const [allergyName, setAllergyName] = useState("");
  const [allergySeverity, setAllergySeverity] = useState("Mild");
  const [chronicName, setChronicName] = useState("");
  const [medName, setMedName] = useState("");
  const [medDosage, setMedDosage] = useState("");
  const [surgeryName, setSurgeryName] = useState("");
  const [surgeryDate, setSurgeryDate] = useState("");

  const totalSteps = 4;

  // ── logic — untouched ────────────────────────────────────────────────────
  const goToSignedInHome = async () => {
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

  const handleNext = async () => {
    if (step < totalSteps) {
      animateStep(step + 1);
      return;
    }
    await saveAll();
  };

  const handleSkip = () => {
    if (step < totalSteps) {
      animateStep(step + 1);
    } else {
      goToSignedInHome();
    }
  };

  const saveAll = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (pid <= 0) {
        Toast.show({ type: "error", text1: tr("patient_profile_not_found") });
        return;
      }

      const promises: Promise<any>[] = [];
      if (allergyName.trim()) {
        promises.push(createAllergy(pid, {
          allergenName: allergyName, severity: allergySeverity,
          reactionDescription: "", allergyType: "General", isActive: true,
        }));
      }
      if (surgeryName.trim()) {
        promises.push(createSurgery(pid, {
          surgeryName, hospitalName: "", doctorName: "",
          surgeryDate, complications: "", notes: "",
        }));
      }

      if (promises.length > 0) await Promise.all(promises);

      Toast.show({ type: "success", text1: tr("saved") });
      goToSignedInHome();
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || tr("error") });
    } finally {
      setLoading(false);
    }
  };

  // ── helpers ──────────────────────────────────────────────────────────────
  const inputProps = (id: string) => ({
    style: [s.inputField, focusedInput === id && s.inputFieldFocused],
    placeholderTextColor: "#94A3B8",
    onFocus: () => setFocusedInput(id),
    onBlur: () => setFocusedInput(null),
  });

  const wrapperProps = (id: string) => ({
    style: [s.inputWrapper, focusedInput === id && s.inputWrapperFocused],
  });

  const iconColor = (id: string) =>
    focusedInput === id ? C.primary : C.textMuted;

  // ── step content ─────────────────────────────────────────────────────────
  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <View>
            <Text style={s.stepTitle}>{tr("allergies")}</Text>
            <Text style={s.stepDesc}>{tr("onboarding_allergy_desc")}</Text>

            <Text style={s.label}>{tr("allergen_name")}</Text>
            <View {...wrapperProps("allergyName")}>
              <Ionicons name="leaf-outline" size={18} color={iconColor("allergyName")} style={s.icon} />
              <TextInput
                placeholder={tr("allergen_name")}
                value={allergyName} onChangeText={setAllergyName}
                {...inputProps("allergyName")}
              />
            </View>

            <Text style={s.label}>{tr("severity")}</Text>
            <View style={s.chipRow}>
              {["Mild", "Moderate", "Severe"].map((sv) => (
                <TouchableOpacity
                  key={sv}
                  style={[s.chip, allergySeverity === sv && s.chipActive]}
                  onPress={() => setAllergySeverity(sv)}
                  activeOpacity={0.8}
                >
                  <Text style={[s.chipTxt, allergySeverity === sv && s.chipTxtActive]}>{sv}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        );

      case 2:
        return (
          <View>
            <Text style={s.stepTitle}>{tr("chronic_diseases")}</Text>
            <Text style={s.stepDesc}>{tr("onboarding_chronic_desc")}</Text>

            <Text style={s.label}>{tr("disease_name")}</Text>
            <View {...wrapperProps("chronic")}>
              <Ionicons name="pulse-outline" size={18} color={iconColor("chronic")} style={s.icon} />
              <TextInput
                placeholder={tr("disease_name")}
                value={chronicName} onChangeText={setChronicName}
                {...inputProps("chronic")}
              />
            </View>
          </View>
        );

      case 3:
        return (
          <View>
            <Text style={s.stepTitle}>{tr("medications")}</Text>
            <Text style={s.stepDesc}>{tr("onboarding_medication_desc")}</Text>

            <Text style={s.label}>{tr("medication_name")}</Text>
            <View {...wrapperProps("medName")}>
              <Ionicons name="medkit-outline" size={18} color={iconColor("medName")} style={s.icon} />
              <TextInput
                placeholder={tr("medication_name")}
                value={medName} onChangeText={setMedName}
                {...inputProps("medName")}
              />
            </View>

            <Text style={s.label}>{tr("dosage")}</Text>
            <View {...wrapperProps("dosage")}>
              <Ionicons name="flask-outline" size={18} color={iconColor("dosage")} style={s.icon} />
              <TextInput
                placeholder={tr("dosage")}
                value={medDosage} onChangeText={setMedDosage}
                {...inputProps("dosage")}
              />
            </View>
          </View>
        );

      case 4:
        return (
          <View>
            <Text style={s.stepTitle}>{tr("surgeries")}</Text>
            <Text style={s.stepDesc}>{tr("onboarding_surgery_desc")}</Text>

            <Text style={s.label}>{tr("surgery_name")}</Text>
            <View {...wrapperProps("surgeryName")}>
              <Ionicons name="cut-outline" size={18} color={iconColor("surgeryName")} style={s.icon} />
              <TextInput
                placeholder={tr("surgery_name")}
                value={surgeryName} onChangeText={setSurgeryName}
                {...inputProps("surgeryName")}
              />
            </View>

            <Text style={s.label}>{tr("date")}</Text>
            <View {...wrapperProps("surgeryDate")}>
              <Ionicons name="calendar-outline" size={18} color={iconColor("surgeryDate")} style={s.icon} />
              <TextInput
                placeholder="YYYY-MM-DD"
                value={surgeryDate} onChangeText={setSurgeryDate}
                {...inputProps("surgeryDate")}
              />
            </View>
          </View>
        );

      default: return null;
    }
  };

  const meta = STEP_META[step - 1];

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.primary} />

      {/* ── decorative bg circles (same as login) ── */}
      <View style={[s.bgCircle, s.circleTop]} />
      <View style={[s.bgCircle, s.circleBottom]} />

      <Animated.View style={[s.page, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>

        {/* ── Header gradient card ── */}
        <LinearGradient colors={[C.primary, "#047857"]} style={s.topCard}>
          <View style={s.topCardInner}>
            <View style={s.stepIconWrap}>
              <Ionicons name={meta.icon} size={28} color="#fff" />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={s.topLabel}>{tr("medical_history_onboarding")}</Text>
              <Text style={s.topStep}>{tr("step")} {step} {tr("of")} {totalSteps}</Text>
            </View>
          </View>

          {/* progress bar */}
          <View style={s.progressBg}>
            <Animated.View style={[s.progressFill, { width: `${(step / totalSteps) * 100}%` as any }]} />
          </View>
        </LinearGradient>

        {/* ── Form card – mirrors glassCard from login ── */}
        <ScrollView
          style={s.scroll}
          contentContainerStyle={s.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <Animated.View
            style={[s.glassCard, { opacity: stepFade, transform: [{ translateY: stepSlide }] }]}
          >
            {renderStep()}
          </Animated.View>
        </ScrollView>

        {/* ── Footer ── */}
        <View style={s.footer}>
          <TouchableOpacity style={s.skipBtn} onPress={handleSkip} activeOpacity={0.75}>
            <Text style={s.skipTxt}>{tr("skip")}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[s.nextBtn, loading && s.nextBtnDisabled]}
            onPress={handleNext}
            disabled={loading}
            activeOpacity={0.85}
          >
            <LinearGradient colors={[C.primary, "#047857"]} style={s.nextBtnGrad}>
              {loading
                ? <ActivityIndicator color="#fff" size="small" />
                : (
                  <>
                    <Text style={s.nextTxt}>{step === totalSteps ? tr("finish") : tr("next")}</Text>
                    <Ionicons
                      name={step === totalSteps ? "checkmark" : "arrow-forward"}
                      size={18} color="#fff"
                      style={{ marginLeft: 6 }}
                    />
                  </>
                )
              }
            </LinearGradient>
          </TouchableOpacity>
        </View>

      </Animated.View>

      <Toast />
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  // bg decorations (login-style)
  bgCircle: { position: "absolute", borderRadius: 200, opacity: 0.35 },
  circleTop: {
    width: 300, height: 300,
    backgroundColor: "rgba(16,185,129,0.14)",
    top: -100, right: -100,
  },
  circleBottom: {
    width: 350, height: 350,
    backgroundColor: "rgba(2,132,199,0.08)",
    bottom: -150, left: -150,
  },

  page: { flex: 1 },

  // header gradient card
  topCard: {
    paddingTop: 56, paddingBottom: 20,
    paddingHorizontal: 24,
    borderBottomLeftRadius: 28,
    borderBottomRightRadius: 28,
  },
  topCardInner: { flexDirection: "row", alignItems: "center", gap: 16, marginBottom: 20 },
  stepIconWrap: {
    width: 52, height: 52, borderRadius: 16,
    backgroundColor: "rgba(255,255,255,0.2)",
    justifyContent: "center", alignItems: "center",
  },
  topLabel: { fontSize: 13, color: "rgba(255,255,255,0.75)", fontWeight: "600", letterSpacing: 0.3 },
  topStep: { fontSize: 20, color: "#fff", fontWeight: "800", marginTop: 2 },

  progressBg: { height: 5, backgroundColor: "rgba(255,255,255,0.25)", borderRadius: 4 },
  progressFill: { height: 5, backgroundColor: "#fff", borderRadius: 4 },

  // scroll
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 12 },

  // glass card – same as login
  glassCard: {
    backgroundColor: "rgba(255,255,255,0.92)",
    borderRadius: 24,
    padding: 24,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.06,
    shadowRadius: 24,
    elevation: 5,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.6)",
  },

  stepTitle: { fontSize: 20, fontWeight: "800", color: C.text, marginBottom: 6 },
  stepDesc: { fontSize: 14, color: C.textMuted, marginBottom: 22, lineHeight: 20 },

  label: {
    fontSize: 12, fontWeight: "700", color: C.text,
    textTransform: "uppercase", letterSpacing: 0.5,
    marginBottom: 8, marginLeft: 2,
  },
  inputWrapper: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: "#F8FAFC",
    borderWidth: 1, borderColor: C.border,
    borderRadius: 16, height: 56,
    paddingHorizontal: 14, marginBottom: 16,
  },
  inputWrapperFocused: { borderColor: C.primary, backgroundColor: "#fff" },
  icon: { marginRight: 10 },
  inputField: {
    flex: 1, fontSize: 16, color: C.text, height: "100%",
  },
  inputFieldFocused: {},

  // severity chips
  chipRow: { flexDirection: "row", gap: 10, marginBottom: 8 },
  chip: {
    flex: 1, height: 44,
    borderRadius: 14, borderWidth: 1, borderColor: C.border,
    justifyContent: "center", alignItems: "center",
    backgroundColor: "#F8FAFC",
  },
  chipActive: { backgroundColor: "rgba(5,150,105,0.1)", borderColor: C.primary },
  chipTxt: { fontSize: 13, fontWeight: "600", color: C.textMuted },
  chipTxtActive: { color: C.primary },

  // footer
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20, paddingVertical: 16,
    paddingBottom: 36,
    backgroundColor: "#fff",
    borderTopWidth: 1, borderTopColor: "#F1F5F9",
    gap: 12,
  },
  skipBtn: {
    paddingHorizontal: 20, paddingVertical: 14,
    borderRadius: 14, borderWidth: 1, borderColor: C.border,
  },
  skipTxt: { fontSize: 14, fontWeight: "600", color: C.textMuted },
  nextBtn: {
    flex: 1, borderRadius: 16, overflow: "hidden",
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.22, shadowRadius: 10, elevation: 5,
  },
  nextBtnDisabled: { opacity: 0.65 },
  nextBtnGrad: {
    height: 52, flexDirection: "row",
    justifyContent: "center", alignItems: "center",
  },
  nextTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
});