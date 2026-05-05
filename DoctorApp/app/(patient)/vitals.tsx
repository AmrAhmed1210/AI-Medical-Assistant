import React, { useEffect, useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, TextInput, Switch, Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { SosBar } from "../../components/SosBar";
import { getMyPatientId } from "../../services/authService";
import {
  getPatientVitals, getLatestVital, addVitalReading,
  NORMAL_RANGES, checkVitalNormal, getVitalRangeText,
  type VitalReading, type CreateVitalPayload,
} from "../../services/vitalService";
import { getAllergies, getChronicDiseases, type ChronicDisease } from "../../services/medicalRecordService";
import Toast from "react-native-toast-message";

const VITAL_TYPES = [
  { key: "Blood Pressure", unit: "mmHg", hasValue2: true },
  { key: "Blood Sugar", unit: "mg/dL", hasValue2: false },
  { key: "Heart Rate", unit: "bpm", hasValue2: false },
  { key: "Temperature", unit: "C", hasValue2: false },
  { key: "SpO2", unit: "%", hasValue2: false },
  { key: "Respiratory Rate", unit: "breaths/min", hasValue2: false },
];

export default function VitalsScreen() {
  const [loading, setLoading] = useState(true);
  const [patientId, setPatientId] = useState<number | null>(null);
  const [vitals, setVitals] = useState<VitalReading[]>([]);
  const [chronicDiseases, setChronicDiseases] = useState<ChronicDisease[]>([]);
  const [latestVitals, setLatestVitals] = useState<Record<string, VitalReading | null>>({});
  const [sosData, setSosData] = useState<{ bloodType: string; allergies: any[] } | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [saving, setSaving] = useState(false);

  // Add form state
  const [selectedType, setSelectedType] = useState("Blood Pressure");
  const [value, setValue] = useState("");
  const [value2, setValue2] = useState("");
  const [notes, setNotes] = useState("");
  const [isNormal, setIsNormal] = useState(true);

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

      const [allVitals, allergies, diseases] = await Promise.all([
        getPatientVitals(pid).catch(() => []),
        getAllergies(pid).catch(() => []),
        getChronicDiseases(pid).catch(() => []),
      ]);

      setVitals(allVitals);
      setChronicDiseases(diseases);
      if (allergies.length > 0) {
        setSosData({ bloodType: "", allergies });
      }

      // Load latest for each type
      const latestPromises = VITAL_TYPES.map(async (t) => {
        const v = await getLatestVital(pid, t.key).catch(() => null);
        return { type: t.key, reading: v };
      });
      const latestResults = await Promise.all(latestPromises);
      const latestMap: Record<string, VitalReading | null> = {};
      latestResults.forEach((r) => { latestMap[r.type] = r.reading; });
      setLatestVitals(latestMap);
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load vitals" });
    } finally {
      setLoading(false);
    }
  };

  const currentTypeInfo = VITAL_TYPES.find((t) => t.key === selectedType)!;

  const validateAndSetNormal = () => {
    const v = Number(value);
    const v2 = currentTypeInfo.hasValue2 ? Number(value2) : undefined;
    if (!Number.isNaN(v) && v > 0) {
      setIsNormal(checkVitalNormal(selectedType, v, v2));
    }
  };

  const handleSave = async () => {
    const numValue = Number(value);
    if (Number.isNaN(numValue) || numValue <= 0) {
      Toast.show({ type: "error", text1: "Please enter a valid value" });
      return;
    }
    if (currentTypeInfo.hasValue2 && (!value2 || Number(value2) <= 0)) {
      Toast.show({ type: "error", text1: "Please enter both systolic and diastolic values" });
      return;
    }
    const activeDisease = chronicDiseases.find((d) => d.isActive);
    const monitorId = activeDisease ? activeDisease.id : undefined;

    const payload: CreateVitalPayload = {
      chronicDiseaseMonitorId: monitorId,
      readingType: selectedType,
      value: numValue,
      value2: currentTypeInfo.hasValue2 && value2 ? Number(value2) : undefined,
      unit: currentTypeInfo.unit,
      isNormal: isNormal,
      notes: notes || undefined,
      sugarReadingContext: selectedType === "Blood Sugar" ? "random" : undefined,
    };

    try {
      setSaving(true);
      await addVitalReading(patientId!, payload);
      Toast.show({ type: "success", text1: "Vital recorded successfully" });
      setValue("");
      setValue2("");
      setNotes("");
      setShowAddForm(false);
      await loadData();
    } catch (e: any) {
      const status = e?.status;
      if (status === 500) {
        Toast.show({ type: "error", text1: "Server error. Please ensure you have a chronic disease record, or contact support." });
      } else {
        Toast.show({ type: "error", text1: e.message || "Failed to save vital" });
      }
    } finally {
      setSaving(false);
    }
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

      <View style={styles.header}>
        <Text style={styles.headerTitle}>Vitals Tracker</Text>
        <Text style={styles.headerSubtitle}>Monitor your health readings</Text>
      </View>

      {sosData && <SosBar bloodType={sosData.bloodType} allergies={sosData.allergies} />}

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Quick Add Button */}
        <TouchableOpacity style={styles.addBtn} onPress={() => setShowAddForm(!showAddForm)}>
          <Ionicons name={showAddForm ? "close-circle" : "add-circle"} size={20} color="#fff" />
          <Text style={styles.addBtnTxt}>{showAddForm ? "Cancel" : "Log New Reading"}</Text>
        </TouchableOpacity>

        {/* Add Form */}
        {showAddForm && (
          <View style={styles.formCard}>
            <Text style={styles.formTitle}>New Vital Reading</Text>

            {/* Type Selector */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.typeScroll}>
              {VITAL_TYPES.map((t) => (
                <TouchableOpacity
                  key={t.key}
                  style={[styles.typeChip, selectedType === t.key && styles.typeChipActive]}
                  onPress={() => { setSelectedType(t.key); setValue(""); setValue2(""); }}
                >
                  <Text style={[styles.typeChipTxt, selectedType === t.key && styles.typeChipTxtActive]}>{t.key}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            {/* Value Inputs */}
            <View style={styles.inputRow}>
              <View style={styles.inputWrap}>
                <Text style={styles.inputLabel}>Value {currentTypeInfo.hasValue2 && "(Systolic)"}</Text>
                <TextInput
                  style={styles.input}
                  keyboardType="numeric"
                  value={value}
                  onChangeText={(t) => { setValue(t); validateAndSetNormal(); }}
                  placeholder={getVitalRangeText(selectedType + (currentTypeInfo.hasValue2 ? " Systolic" : ""))}
                />
              </View>
              {currentTypeInfo.hasValue2 && (
                <View style={styles.inputWrap}>
                  <Text style={styles.inputLabel}>Diastolic</Text>
                  <TextInput
                    style={styles.input}
                    keyboardType="numeric"
                    value={value2}
                    onChangeText={(t) => { setValue2(t); validateAndSetNormal(); }}
                    placeholder={getVitalRangeText("Blood Pressure Diastolic")}
                  />
                </View>
              )}
            </View>

            {!isNormal && (
              <View style={styles.abnormalBanner}>
                <Text style={styles.abnormalText}>⚠ Outside normal range</Text>
              </View>
            )}

            <View style={styles.inputWrap}>
              <Text style={styles.inputLabel}>Notes (optional)</Text>
              <TextInput style={styles.input} value={notes} onChangeText={setNotes} placeholder="e.g., after meal, fasting..." />
            </View>

            <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={saving}>
              {saving ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.saveBtnTxt}>Save Reading</Text>}
            </TouchableOpacity>
          </View>
        )}

        {/* Latest Vitals Summary */}
        <Text style={styles.sectionTitle}>Latest Readings</Text>
        <View style={styles.latestGrid}>
          {VITAL_TYPES.map((t) => {
            const reading = latestVitals[t.key];
            const isAbnormal = reading && !reading.isNormal;
            return (
              <View key={t.key} style={[styles.latestCard, isAbnormal && styles.latestCardAbnormal]}>
                <Text style={styles.latestType}>{t.key}</Text>
                {reading ? (
                  <>
                    <Text style={[styles.latestValue, isAbnormal && styles.latestValueAbnormal]}>
                      {reading.value}{reading.value2 != null ? ` / ${reading.value2}` : ""} {reading.unit}
                    </Text>
                    <Text style={styles.latestDate}>{new Date(reading.recordedAt).toLocaleDateString()}</Text>
                  </>
                ) : (
                  <Text style={styles.latestEmpty}>—</Text>
                )}
              </View>
            );
          })}
        </View>

        {/* History List */}
        <Text style={styles.sectionTitle}>History</Text>
        {vitals.length === 0 ? (
          <View style={styles.emptyCard}>
            <Ionicons name="pulse-outline" size={48} color="#CBD5E1" />
            <Text style={styles.emptyTitle}>No readings yet</Text>
            <Text style={styles.emptyDesc}>Tap &quot;Log New Reading&quot; to record your first vital.</Text>
          </View>
        ) : (
          vitals.map((v, i) => {
            const isAbnormal = !v.isNormal;
            return (
              <View key={i} style={[styles.historyCard, isAbnormal && styles.historyCardAbnormal]}>
                <View style={styles.historyHeader}>
                  <Text style={styles.historyType}>{v.readingType}</Text>
                  <View style={[styles.historyBadge, isAbnormal ? styles.historyBadgeAbnormal : styles.historyBadgeNormal]}>
                    <Text style={[styles.historyBadgeTxt, isAbnormal ? styles.historyBadgeTxtAbnormal : styles.historyBadgeTxtNormal]}>
                      {isAbnormal ? "Abnormal" : "Normal"}
                    </Text>
                  </View>
                </View>
                <Text style={[styles.historyValue, isAbnormal && styles.historyValueAbnormal]}>
                  {v.value}{v.value2 != null ? ` / ${v.value2}` : ""} {v.unit}
                </Text>
                {v.notes && <Text style={styles.historyNotes}>{v.notes}</Text>}
                <Text style={styles.historyDate}>{new Date(v.recordedAt).toLocaleString()}</Text>
              </View>
            );
          })
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
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 40 },
  addBtn: { backgroundColor: COLORS.primary, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, paddingVertical: 14, borderRadius: 16, marginBottom: 16 },
  addBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  formCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 16, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  formTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginBottom: 12 },
  typeScroll: { flexDirection: "row", marginBottom: 12 },
  typeChip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, backgroundColor: "#F1F5F9", marginRight: 8, borderWidth: 1, borderColor: "#E2E8F0" },
  typeChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  typeChipTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  typeChipTxtActive: { color: "#fff" },
  inputRow: { flexDirection: "row", gap: 12 },
  inputWrap: { flex: 1, marginBottom: 12 },
  inputLabel: { fontSize: 12, fontWeight: "600", color: "#64748B", marginBottom: 6 },
  input: { backgroundColor: "#F8FAFC", borderRadius: 12, paddingHorizontal: 15, paddingVertical: 12, fontSize: 15, color: "#1E293B", borderWidth: 1, borderColor: "#E2E8F0" },
  abnormalBanner: { backgroundColor: "#FFEBEE", borderRadius: 10, padding: 10, marginBottom: 12 },
  abnormalText: { color: "#C62828", fontSize: 13, fontWeight: "700" },
  saveBtn: { backgroundColor: COLORS.primary, paddingVertical: 14, borderRadius: 12, alignItems: "center" },
  saveBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  sectionTitle: { fontSize: 18, fontWeight: "800", color: "#1E293B", marginBottom: 12, marginTop: 8 },
  latestGrid: { flexDirection: "row", flexWrap: "wrap", justifyContent: "space-between", marginBottom: 16 },
  latestCard: { width: "48%", backgroundColor: "#fff", borderRadius: 16, padding: 14, marginBottom: 12, borderWidth: 1, borderColor: "#F0F0F0", shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.03, shadowRadius: 10, elevation: 2 },
  latestCardAbnormal: { borderColor: "#FFCDD2", backgroundColor: "#FFEBEE" },
  latestType: { fontSize: 11, fontWeight: "700", color: "#64748B", marginBottom: 6, textTransform: "uppercase" },
  latestValue: { fontSize: 20, fontWeight: "800", color: "#1E293B" },
  latestValueAbnormal: { color: "#C62828" },
  latestDate: { fontSize: 11, color: "#94A3B8", marginTop: 4 },
  latestEmpty: { fontSize: 18, color: "#CBD5E1", fontWeight: "700" },
  emptyCard: { backgroundColor: "#fff", borderRadius: 16, padding: 40, alignItems: "center", marginTop: 8 },
  emptyTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginTop: 16 },
  emptyDesc: { fontSize: 13, color: "#94A3B8", marginTop: 6, textAlign: "center" },
  historyCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 10, borderWidth: 1, borderColor: "#F0F0F0", shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.03, shadowRadius: 10, elevation: 2 },
  historyCardAbnormal: { borderColor: "#FFCDD2" },
  historyHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  historyType: { fontSize: 14, fontWeight: "700", color: "#1E293B" },
  historyBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  historyBadgeNormal: { backgroundColor: "#E0F2F1" },
  historyBadgeAbnormal: { backgroundColor: "#FFEBEE" },
  historyBadgeTxt: { fontSize: 11, fontWeight: "700" },
  historyBadgeTxtNormal: { color: "#00695C" },
  historyBadgeTxtAbnormal: { color: "#C62828" },
  historyValue: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  historyValueAbnormal: { color: "#C62828" },
  historyNotes: { fontSize: 12, color: "#64748B", marginTop: 4, fontStyle: "italic" },
  historyDate: { fontSize: 11, color: "#94A3B8", marginTop: 6 },
});
