import React, { useEffect, useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { getMyPatientId } from "../../services/authService";
import {
  getAllergies, getChronicDiseases, getMedications, getVitals,
  AllergyRecord, ChronicDisease, Medication, VitalReading,
} from "../../services/medicalRecordService";
import Toast from "react-native-toast-message";

export default function MedicalRecordsScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"allergies" | "chronic" | "medications" | "vitals">("allergies");
  const [patientId, setPatientId] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  const [allergies, setAllergies] = useState<AllergyRecord[]>([]);
  const [chronicDiseases, setChronicDiseases] = useState<ChronicDisease[]>([]);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [vitals, setVitals] = useState<VitalReading[]>([]);

  useEffect(() => {
    init();
  }, []);

  const init = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (pid <= 0) {
        Toast.show({ type: "error", text1: "Patient profile not found." });
        return;
      }
      setPatientId(pid);
      await loadAll(pid);
    } finally {
      setLoading(false);
    }
  };

  const loadAll = async (pid: number) => {
    try {
      const [a, c, m, v] = await Promise.all([
        getAllergies(pid),
        getChronicDiseases(pid),
        getMedications(pid),
        getVitals(pid),
      ]);
      setAllergies(a);
      setChronicDiseases(c);
      setMedications(m);
      setVitals(v);
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load records" });
    }
  };

  const tabs = [
    { id: "allergies" as const, icon: "warning-outline", label: "Allergies" },
    { id: "chronic" as const, icon: "fitness-outline", label: "Chronic" },
    { id: "medications" as const, icon: "medical-outline", label: "Meds" },
    { id: "vitals" as const, icon: "pulse-outline", label: "Vitals" },
  ];

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Medical Records</Text>
          <View style={{ width: 22 }} />
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabsWrap}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabsScroll}>
          {tabs.map((tab) => (
            <TouchableOpacity
              key={tab.id}
              style={[styles.tabBtn, activeTab === tab.id && styles.tabBtnActive]}
              onPress={() => setActiveTab(tab.id)}
            >
              <Ionicons name={tab.icon as any} size={16} color={activeTab === tab.id ? COLORS.primary : "#94A3B8"} />
              <Text style={[styles.tabBtnTxt, activeTab === tab.id && styles.tabBtnTxtActive]}>{tab.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Allergies */}
        {activeTab === "allergies" && (
          <View>
            {allergies.length === 0 ? (
              <EmptyState icon="warning-outline" title="No Allergies" subtitle="No allergy records found." />
            ) : (
              allergies.map((item, i) => (
                <View key={i} style={styles.card}>
                  <View style={styles.cardRow}>
                    <Text style={styles.cardTitle}>{item.allergenName}</Text>
                    <View style={[styles.badge, item.severity === "Severe" ? styles.badgeSevere : styles.badgeMild]}>
                      <Text style={styles.badgeTxt}>{item.severity}</Text>
                    </View>
                  </View>
                  <Text style={styles.cardText}>{item.allergyType}</Text>
                  {item.reactionDescription && <Text style={styles.cardMeta}>Reaction: {item.reactionDescription}</Text>}
                </View>
              ))
            )}
          </View>
        )}

        {/* Chronic Diseases */}
        {activeTab === "chronic" && (
          <View>
            {chronicDiseases.length === 0 ? (
              <EmptyState icon="fitness-outline" title="No Chronic Diseases" subtitle="No chronic disease records found." />
            ) : (
              chronicDiseases.map((item, i) => (
                <View key={i} style={styles.card}>
                  <View style={styles.cardRow}>
                    <Text style={styles.cardTitle}>{item.diseaseName}</Text>
                    <View style={[styles.badge, item.isActive ? styles.badgeActive : styles.badgeInactive]}>
                      <Text style={styles.badgeTxt}>{item.isActive ? "Active" : "Inactive"}</Text>
                    </View>
                  </View>
                  <Text style={styles.cardText}>{item.diseaseType}</Text>
                  {item.doctorNotes && <Text style={styles.cardMeta}>Notes: {item.doctorNotes}</Text>}
                </View>
              ))
            )}
          </View>
        )}

        {/* Medications */}
        {activeTab === "medications" && (
          <View>
            {medications.length === 0 ? (
              <EmptyState icon="medical-outline" title="No Medications" subtitle="No medication records found." />
            ) : (
              medications.map((item, i) => (
                <View key={i} style={styles.card}>
                  <View style={styles.cardRow}>
                    <Text style={styles.cardTitle}>{item.medicationName}</Text>
                    <View style={[styles.badge, item.isActive ? styles.badgeActive : styles.badgeInactive]}>
                      <Text style={styles.badgeTxt}>{item.isActive ? "Active" : "Inactive"}</Text>
                    </View>
                  </View>
                  <Text style={styles.cardText}>{item.dosage} • {item.frequency}</Text>
                  {item.instructions && <Text style={styles.cardMeta}>{item.instructions}</Text>}
                </View>
              ))
            )}
          </View>
        )}

        {/* Vitals */}
        {activeTab === "vitals" && (
          <View>
            {vitals.length === 0 ? (
              <EmptyState icon="pulse-outline" title="No Vitals" subtitle="No vital readings found." />
            ) : (
              vitals.map((item, i) => (
                <View key={i} style={styles.card}>
                  <View style={styles.cardRow}>
                    <Text style={styles.cardTitle}>{item.readingType}</Text>
                    <Text style={[styles.vitalValue, !item.isNormal && styles.abnormal]}>
                      {item.value}{item.value2 ? `/${item.value2}` : ""} {item.unit}
                    </Text>
                  </View>
                  <Text style={styles.cardMeta}>{new Date(item.recordedAt).toLocaleDateString()}</Text>
                </View>
              ))
            )}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

function EmptyState({ icon, title, subtitle }: { icon: any; title: string; subtitle: string }) {
  return (
    <View style={styles.emptyState}>
      <View style={styles.emptyCircle}>
        <Ionicons name={icon} size={40} color={COLORS.primary} />
      </View>
      <Text style={styles.emptyStateTitle}>{title}</Text>
      <Text style={styles.emptyStateSub}>{subtitle}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  header: { backgroundColor: COLORS.primary, paddingBottom: 20, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 20, paddingTop: 60, paddingBottom: 20 },
  headerTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  tabsWrap: { backgroundColor: "#fff", paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  tabsScroll: { paddingHorizontal: 20 },
  tabBtn: { flexDirection: "row", alignItems: "center", gap: 6, paddingHorizontal: 16, paddingVertical: 8, borderRadius: 12, marginRight: 10, backgroundColor: "#F8FAFC" },
  tabBtnActive: { backgroundColor: COLORS.primary + "15" },
  tabBtnTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  tabBtnTxtActive: { color: COLORS.primary },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 40 },
  card: { backgroundColor: "#fff", borderRadius: 20, padding: 18, marginBottom: 14, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  cardRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  cardTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  cardText: { fontSize: 14, color: "#475569", marginTop: 4 },
  cardMeta: { fontSize: 12, color: "#94A3B8", marginTop: 6 },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  badgeMild: { backgroundColor: "#FEF9C3" },
  badgeSevere: { backgroundColor: "#FEE2E2" },
  badgeActive: { backgroundColor: "#DCFCE7" },
  badgeInactive: { backgroundColor: "#F1F5F9" },
  badgeTxt: { fontSize: 10, fontWeight: "700", color: "#64748B" },
  vitalValue: { fontSize: 16, fontWeight: "700", color: COLORS.primary },
  abnormal: { color: "#E11D48" },
  emptyState: { alignItems: "center", paddingVertical: 40 },
  emptyCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: COLORS.primary + "10", justifyContent: "center", alignItems: "center", marginBottom: 15 },
  emptyStateTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginBottom: 5 },
  emptyStateSub: { fontSize: 13, color: "#64748B", textAlign: "center" },
});
