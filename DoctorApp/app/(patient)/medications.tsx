import React, { useEffect, useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { SosBar } from "../../components/SosBar";
import { getMyPatientId } from "../../services/authService";
import { getPatientMedications, getMedicationSchedule, type MedicationTracker, type MedicationScheduleItem } from "../../services/medicationService";
import { getAllergies } from "../../services/medicalRecordService";
import Toast from "react-native-toast-message";

export default function MedicationsScreen() {
  const [loading, setLoading] = useState(true);
  const [patientId, setPatientId] = useState<number | null>(null);
  const [medications, setMedications] = useState<MedicationTracker[]>([]);
  const [schedule, setSchedule] = useState<MedicationScheduleItem[]>([]);
  const [sosData, setSosData] = useState<{ bloodType: string; allergies: any[] } | null>(null);
  const [activeTab, setActiveTab] = useState<"schedule" | "active">("schedule");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (!pid) {
        Toast.show({ type: "error", text1: "Patient ID not found" });
        setLoading(false);
        return;
      }
      setPatientId(pid);

      const [meds, sched, allergies] = await Promise.all([
        getPatientMedications(pid).catch(() => []),
        getMedicationSchedule(pid).catch(() => []),
        getAllergies(pid).catch(() => []),
      ]);

      setMedications(meds);
      setSchedule(sched);
      if (allergies.length > 0) {
        setSosData({ bloodType: "", allergies });
      }
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load medications" });
    } finally {
      setLoading(false);
    }
  };

  const getStatusStyle = (status: string) => {
    const s = status?.toLowerCase() || "";
    if (s === "taken") return { bg: "#E0F2F1", text: "#00695C", label: "Taken" };
    if (s === "missed") return { bg: "#FFEBEE", text: "#C62828", label: "Missed" };
    if (s === "skipped") return { bg: "#FFF3E0", text: "#E65100", label: "Skipped" };
    return { bg: "#F5F5F5", text: "#757575", label: "Pending" };
  };

  const getStockStyle = (pills: number, threshold: number) => {
    if (pills <= 3) return { bg: "#FFEBEE", text: "#C62828" };
    if (pills <= threshold) return { bg: "#FFF3E0", text: "#E65100" };
    return { bg: "#E0F2F1", text: "#00695C" };
  };

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
        <Text style={styles.headerTitle}>Medications</Text>
        <Text style={styles.headerSubtitle}>Manage your daily medications</Text>
      </View>

      {/* SOS Bar */}
      {sosData && (
        <SosBar
          bloodType={sosData.bloodType}
          allergies={sosData.allergies}
        />
      )}

      {/* Tabs */}
      <View style={styles.tabRow}>
        <TouchableOpacity
          style={[styles.tabBtn, activeTab === "schedule" && styles.tabBtnActive]}
          onPress={() => setActiveTab("schedule")}
        >
          <Text style={[styles.tabBtnTxt, activeTab === "schedule" && styles.tabBtnTxtActive]}>Today&apos;s Schedule</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabBtn, activeTab === "active" && styles.tabBtnActive]}
          onPress={() => setActiveTab("active")}
        >
          <Text style={[styles.tabBtnTxt, activeTab === "active" && styles.tabBtnTxtActive]}>Active Medications</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {activeTab === "schedule" ? (
          <>
            {schedule.length === 0 ? (
              <View style={styles.emptyCard}>
                <Ionicons name="medkit-outline" size={48} color="#CBD5E1" />
                <Text style={styles.emptyTitle}>No doses scheduled today</Text>
                <Text style={styles.emptyDesc}>Your medication schedule will appear here.</Text>
              </View>
            ) : (
              schedule.map((item, i) => {
                const status = getStatusStyle(item.status);
                const time = new Date(item.scheduledAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                return (
                  <View key={i} style={styles.scheduleCard}>
                    <View style={styles.scheduleHeader}>
                      <View style={styles.timeBadge}>
                        <Text style={styles.timeBadgeText}>{time}</Text>
                      </View>
                      <View style={[styles.statusBadge, { backgroundColor: status.bg }]}>
                        <Text style={[styles.statusBadgeText, { color: status.text }]}>{status.label}</Text>
                      </View>
                    </View>
                    <Text style={styles.medName}>{item.medicationName}</Text>
                    <Text style={styles.medDosage}>{item.dosage}</Text>
                    {item.status?.toLowerCase() === "pending" && (
                      <TouchableOpacity style={styles.markBtn}>
                        <Text style={styles.markBtnTxt}>Mark as Taken</Text>
                      </TouchableOpacity>
                    )}
                  </View>
                );
              })
            )}
          </>
        ) : (
          <>
            {medications.length === 0 ? (
              <View style={styles.emptyCard}>
                <Ionicons name="medkit-outline" size={48} color="#CBD5E1" />
                <Text style={styles.emptyTitle}>No active medications</Text>
                <Text style={styles.emptyDesc}>Your prescribed medications will appear here.</Text>
              </View>
            ) : (
              medications.map((med) => {
                const stock = getStockStyle(med.pillsRemaining || 0, med.refillThreshold);
                const isLow = (med.pillsRemaining || 0) <= med.refillThreshold;
                return (
                  <View key={med.id} style={styles.medCard}>
                    <View style={styles.medHeader}>
                      <Text style={styles.medName}>{med.medicationName}</Text>
                      {med.isChronic && (
                        <View style={styles.chronicBadge}>
                          <Text style={styles.chronicBadgeText}>Chronic</Text>
                        </View>
                      )}
                    </View>
                    <Text style={styles.medMeta}>{med.dosage} • {med.frequency} • {med.form}</Text>
                    {med.doseTimes && <Text style={styles.medTimes}>⏰ {med.doseTimes}</Text>}
                    {med.instructions && <Text style={styles.medInstructions}>{med.instructions}</Text>}
                    <View style={[styles.stockBar, { backgroundColor: stock.bg }]}>
                      <Text style={[styles.stockText, { color: stock.text }]}>
                        {isLow ? "⚠ " : ""}{med.pillsRemaining ?? 0} pills remaining {isLow ? "— time to refill" : ""}
                      </Text>
                    </View>
                  </View>
                );
              })
            )}
          </>
        )}
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  header: { backgroundColor: COLORS.primary, paddingTop: 60, paddingBottom: 20, paddingHorizontal: 20, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 },
  headerTitle: { fontSize: 22, fontWeight: "800", color: "#fff" },
  headerSubtitle: { fontSize: 13, color: "rgba(255,255,255,0.7)", marginTop: 4 },
  tabRow: { flexDirection: "row", paddingHorizontal: 20, paddingVertical: 12, gap: 10 },
  tabBtn: { flex: 1, paddingVertical: 10, borderRadius: 12, backgroundColor: "#fff", alignItems: "center", borderWidth: 1, borderColor: "#E2E8F0" },
  tabBtnActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  tabBtnTxt: { fontSize: 13, fontWeight: "600", color: "#64748B" },
  tabBtnTxtActive: { color: "#fff" },
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 20, paddingTop: 8, paddingBottom: 40 },
  emptyCard: { backgroundColor: "#fff", borderRadius: 16, padding: 40, alignItems: "center", marginTop: 20 },
  emptyTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginTop: 16 },
  emptyDesc: { fontSize: 13, color: "#94A3B8", marginTop: 6, textAlign: "center" },
  scheduleCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  scheduleHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  timeBadge: { backgroundColor: "#E3F2FD", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  timeBadgeText: { fontSize: 12, fontWeight: "700", color: "#1565C0" },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  statusBadgeText: { fontSize: 12, fontWeight: "700" },
  medName: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  medDosage: { fontSize: 13, color: "#64748B", marginTop: 4 },
  markBtn: { backgroundColor: COLORS.primary, paddingVertical: 12, borderRadius: 12, alignItems: "center", marginTop: 12 },
  markBtnTxt: { color: "#fff", fontSize: 14, fontWeight: "700" },
  medCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  medHeader: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 6 },
  chronicBadge: { backgroundColor: "#E0F2F1", paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  chronicBadgeText: { fontSize: 10, fontWeight: "700", color: "#00695C" },
  medMeta: { fontSize: 13, color: "#64748B" },
  medTimes: { fontSize: 12, color: "#475569", marginTop: 4 },
  medInstructions: { fontSize: 12, color: "#475569", marginTop: 4, fontStyle: "italic" },
  stockBar: { marginTop: 10, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10 },
  stockText: { fontSize: 12, fontWeight: "700" },
});
