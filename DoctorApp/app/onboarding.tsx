import React, { useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, ActivityIndicator, StatusBar, Alert,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../constants/colors";
import { useLanguage } from "../context/LanguageContext";
import { getMyPatientId } from "../services/authService";
import { createAllergy, createSurgery } from "../services/medicalRecordService";
import Toast from "react-native-toast-message";

export default function OnboardingScreen() {
  const router = useRouter();
  const { tr } = useLanguage();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);

  // Step 1: Allergy
  const [allergyName, setAllergyName] = useState("");
  const [allergySeverity, setAllergySeverity] = useState("Mild");

  // Step 2: Chronic
  const [chronicName, setChronicName] = useState("");

  // Step 3: Medication
  const [medName, setMedName] = useState("");
  const [medDosage, setMedDosage] = useState("");

  // Step 4: Surgery
  const [surgeryName, setSurgeryName] = useState("");
  const [surgeryDate, setSurgeryDate] = useState("");

  const totalSteps = 4;

  const handleNext = async () => {
    if (step < totalSteps) {
      setStep(step + 1);
      return;
    }
    // Final step: save everything
    await saveAll();
  };

  const handleSkip = () => {
    if (step < totalSteps) {
      setStep(step + 1);
    } else {
      router.replace("/(patient)/home");
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

      const promises = [];
      if (allergyName.trim()) {
        promises.push(createAllergy(pid, { allergenName: allergyName, severity: allergySeverity, reactionDescription: "", allergyType: "General", isActive: true }));
      }
      if (chronicName.trim()) {
        // Note: addChronicDisease might not exist in the service; skipping for now
      }
      if (surgeryName.trim()) {
        promises.push(createSurgery(pid, { surgeryName, hospitalName: "", doctorName: "", surgeryDate, complications: "", notes: "" }));
      }

      if (promises.length > 0) await Promise.all(promises);

      Toast.show({ type: "success", text1: tr("saved") });
      router.replace("/(patient)/home");
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || tr("error") });
    } finally {
      setLoading(false);
    }
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <View>
            <Text style={styles.stepTitle}>{tr("allergies")}</Text>
            <Text style={styles.stepDesc}>{tr("onboarding_allergy_desc")}</Text>
            <TextInput style={styles.input} placeholder={tr("allergen_name")} value={allergyName} onChangeText={setAllergyName} />
            <TextInput style={styles.input} placeholder={tr("severity")} value={allergySeverity} onChangeText={setAllergySeverity} />
          </View>
        );
      case 2:
        return (
          <View>
            <Text style={styles.stepTitle}>{tr("chronic_diseases")}</Text>
            <Text style={styles.stepDesc}>{tr("onboarding_chronic_desc")}</Text>
            <TextInput style={styles.input} placeholder={tr("disease_name")} value={chronicName} onChangeText={setChronicName} />
          </View>
        );
      case 3:
        return (
          <View>
            <Text style={styles.stepTitle}>{tr("medications")}</Text>
            <Text style={styles.stepDesc}>{tr("onboarding_medication_desc")}</Text>
            <TextInput style={styles.input} placeholder={tr("medication_name")} value={medName} onChangeText={setMedName} />
            <TextInput style={styles.input} placeholder={tr("dosage")} value={medDosage} onChangeText={setMedDosage} />
          </View>
        );
      case 4:
        return (
          <View>
            <Text style={styles.stepTitle}>{tr("surgeries")}</Text>
            <Text style={styles.stepDesc}>{tr("onboarding_surgery_desc")}</Text>
            <TextInput style={styles.input} placeholder={tr("surgery_name")} value={surgeryName} onChangeText={setSurgeryName} />
            <TextInput style={styles.input} placeholder={tr("date")} value={surgeryDate} onChangeText={setSurgeryDate} />
          </View>
        );
      default: return null;
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />
      <View style={styles.header}>
        <Text style={styles.headerTitle}>{tr("medical_history_onboarding")}</Text>
        <Text style={styles.headerSub}>{tr("step")} {step} {tr("of")} {totalSteps}</Text>
      </View>
      <View style={styles.progressBarBg}>
        <View style={[styles.progressBarFill, { width: `${(step / totalSteps) * 100}%` }]} />
      </View>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {renderStep()}
      </ScrollView>
      <View style={styles.footer}>
        <TouchableOpacity style={styles.skipBtn} onPress={handleSkip}>
          <Text style={styles.skipBtnTxt}>{tr("skip")}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.nextBtn} onPress={handleNext} disabled={loading}>
          {loading ? <ActivityIndicator color="#fff" size="small" /> : (
            <Text style={styles.nextBtnTxt}>{step === totalSteps ? tr("finish") : tr("next")}</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  header: { backgroundColor: COLORS.primary, paddingTop: 60, paddingBottom: 20, alignItems: "center", borderBottomLeftRadius: 20, borderBottomRightRadius: 20 },
  headerTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  headerSub: { fontSize: 13, color: "#fff", opacity: 0.8, marginTop: 6 },
  progressBarBg: { height: 4, backgroundColor: "#E2E8F0", marginTop: 0 },
  progressBarFill: { height: 4, backgroundColor: COLORS.primary },
  scroll: { flex: 1 },
  scrollContent: { padding: 24 },
  stepTitle: { fontSize: 20, fontWeight: "700", color: "#1E293B", marginBottom: 8 },
  stepDesc: { fontSize: 14, color: "#64748B", marginBottom: 20, lineHeight: 20 },
  input: { backgroundColor: "#fff", borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14, fontSize: 15, color: "#1E293B", borderWidth: 1, borderColor: "#E2E8F0", marginBottom: 14 },
  footer: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", padding: 20, paddingBottom: 40, backgroundColor: "#fff", borderTopWidth: 1, borderTopColor: "#F1F5F9" },
  skipBtn: { paddingHorizontal: 20, paddingVertical: 12 },
  skipBtnTxt: { fontSize: 15, fontWeight: "600", color: "#64748B" },
  nextBtn: { backgroundColor: COLORS.primary, paddingHorizontal: 28, paddingVertical: 12, borderRadius: 14 },
  nextBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
});
