import React, { useEffect, useState, useCallback } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, Modal, TextInput, Switch,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { SosBar } from "../../components/SosBar";
import { getMyPatientId } from "../../services/authService";
import {
  getPatientMedications, getMedicationSchedule, createPatientMedication, markMedicationTaken,
  type MedicationTracker, type MedicationScheduleItem, type CreateMedicationPayload,
} from "../../services/medicationService";
import { getAllergies } from "../../services/medicalRecordService";
import Toast from "react-native-toast-message";

const DAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
const FORMS = ["Pill", "Syrup", "Injection", "Inhaler", "Cream", "Drops", "Patch", "Powder"];

export default function MedicationsScreen() {
  const [loading, setLoading] = useState(true);
  const [patientId, setPatientId] = useState<number | null>(null);
  const [medications, setMedications] = useState<MedicationTracker[]>([]);
  const [schedule, setSchedule] = useState<MedicationScheduleItem[]>([]);
  const [sosData, setSosData] = useState<{ bloodType: string; allergies: any[] } | null>(null);
  const [activeTab, setActiveTab] = useState<"schedule" | "active">("schedule");
  const [showAddModal, setShowAddModal] = useState(false);
  const [saving, setSaving] = useState(false);

  // Add form state
  const [medName, setMedName] = useState("");
  const [dosage, setDosage] = useState("");
  const [form, setForm] = useState("Pill");
  const [selectedDays, setSelectedDays] = useState<string[]>([...DAYS]);
  const [timesPerDay, setTimesPerDay] = useState(1);
  const [doseTimes, setDoseTimes] = useState("08:00");
  const [startDate, setStartDate] = useState(new Date().toISOString().split("T")[0]);
  const [endDate, setEndDate] = useState("");
  const [pills, setPills] = useState("");
  const [instructions, setInstructions] = useState("");
  const [isChronic, setIsChronic] = useState(false);

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (!pid) { Toast.show({ type: "error", text1: "Patient ID not found" }); setLoading(false); return; }
      setPatientId(pid);
      const [meds, sched, allergies] = await Promise.all([
        getPatientMedications(pid).catch(() => []),
        getMedicationSchedule(pid).catch(() => []),
        getAllergies(pid).catch(() => []),
      ]);
      setMedications(meds);
      setSchedule(sched);
      if (allergies.length > 0) setSosData({ bloodType: "", allergies });
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load medications" });
    } finally { setLoading(false); }
  };

  const toggleDay = (day: string) => {
    setSelectedDays(prev => prev.includes(day) ? prev.filter(d => d !== day) : [...prev, day]);
  };

  const handleSave = async () => {
    if (!patientId || !medName.trim() || !dosage.trim()) {
      Toast.show({ type: "error", text1: "Please fill medication name and dosage" });
      return;
    }
    if (selectedDays.length === 0) {
      Toast.show({ type: "error", text1: "Please select at least one day" });
      return;
    }
    const payload: CreateMedicationPayload = {
      medicationName: medName.trim(),
      dosage: dosage.trim(),
      form,
      frequency: `${timesPerDay}x daily`,
      timesPerDay,
      doseTimes: doseTimes.trim(),
      daysOfWeek: selectedDays.join(","),
      startDate,
      endDate: endDate || undefined,
      pillsRemaining: pills ? Number(pills) : undefined,
      refillThreshold: 5,
      isChronic,
      instructions: instructions.trim() || undefined,
    };
    try {
      setSaving(true);
      await createPatientMedication(patientId, payload);
      Toast.show({ type: "success", text1: "Medication added!" });
      resetForm();
      setShowAddModal(false);
      await loadData();
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to add medication" });
    } finally { setSaving(false); }
  };

  const resetForm = () => {
    setMedName(""); setDosage(""); setForm("Pill"); setSelectedDays([...DAYS]);
    setTimesPerDay(1); setDoseTimes("08:00"); setStartDate(new Date().toISOString().split("T")[0]);
    setEndDate(""); setPills(""); setInstructions(""); setIsChronic(false);
  };

  const handleMarkTaken = async (logId?: number) => {
    if (!logId) return;
    try {
      await markMedicationTaken(logId);
      Toast.show({ type: "success", text1: "Marked as taken!" });
      await loadData();
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to mark" });
    }
  };

  const getStatusStyle = (status: string) => {
    const s = status?.toLowerCase() || "";
    if (s === "taken") return { bg: "#E0F2F1", text: "#00695C", label: "Taken", icon: "checkmark-circle" as const };
    if (s === "missed") return { bg: "#FFEBEE", text: "#C62828", label: "Missed", icon: "close-circle" as const };
    if (s === "skipped") return { bg: "#FFF3E0", text: "#E65100", label: "Skipped", icon: "remove-circle" as const };
    return { bg: "#E3F2FD", text: "#1565C0", label: "Pending", icon: "time" as const };
  };

  const getStockStyle = (pills: number, threshold: number) => {
    if (pills <= 3) return { bg: "#FFEBEE", text: "#C62828" };
    if (pills <= threshold) return { bg: "#FFF3E0", text: "#E65100" };
    return { bg: "#E0F2F1", text: "#00695C" };
  };

  const renderDayChips = (daysStr?: string) => {
    const days = daysStr?.split(",").map(d => d.trim()).filter(Boolean) ?? [];
    const shortNames: Record<string, string> = { Saturday: "Sat", Sunday: "Sun", Monday: "Mon", Tuesday: "Tue", Wednesday: "Wed", Thursday: "Thu", Friday: "Fri" };
    return (
      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 4, marginTop: 6 }}>
        {DAYS.map(d => {
          const active = days.includes(d);
          return (
            <View key={d} style={[styles.dayChip, active && styles.dayChipActive]}>
              <Text style={[styles.dayChipTxt, active && styles.dayChipTxtActive]}>{shortNames[d]}</Text>
            </View>
          );
        })}
      </View>
    );
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
        <View style={styles.headerTop}>
          <Text style={styles.headerTitle}>My Medications</Text>
          <TouchableOpacity style={styles.addBtn} onPress={() => setShowAddModal(true)}>
            <Ionicons name="add" size={22} color="#fff" />
          </TouchableOpacity>
        </View>
        <Text style={styles.headerSubtitle}>Track doses, schedule & reminders</Text>
      </View>

      {sosData && <SosBar bloodType={sosData.bloodType} allergies={sosData.allergies} />}

      <View style={styles.tabRow}>
        <TouchableOpacity style={[styles.tabBtn, activeTab === "schedule" && styles.tabBtnActive]} onPress={() => setActiveTab("schedule")}>
          <Text style={[styles.tabBtnTxt, activeTab === "schedule" && styles.tabBtnTxtActive]}>Today&apos;s Doses</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.tabBtn, activeTab === "active" && styles.tabBtnActive]} onPress={() => setActiveTab("active")}>
          <Text style={[styles.tabBtnTxt, activeTab === "active" && styles.tabBtnTxtActive]}>My Prescriptions</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {activeTab === "schedule" ? (
          <>
            {schedule.length === 0 ? (
              <View style={styles.emptyCard}>
                <Ionicons name="medkit-outline" size={48} color="#CBD5E1" />
                <Text style={styles.emptyTitle}>No doses for today</Text>
                <Text style={styles.emptyDesc}>Add a medication to see your schedule.</Text>
                <TouchableOpacity style={styles.primaryBtnSmall} onPress={() => setShowAddModal(true)}>
                  <Text style={styles.primaryBtnSmallTxt}>Add Medication</Text>
                </TouchableOpacity>
              </View>
            ) : (
              schedule.map((item, i) => {
                const status = getStatusStyle(item.status);
                const time = new Date(item.scheduledAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                const isPending = item.status?.toLowerCase() === "pending";
                return (
                  <View key={i} style={[styles.scheduleCard, isPending && { borderLeftWidth: 4, borderLeftColor: COLORS.primary }]}>
                    <View style={styles.scheduleHeader}>
                      <View style={styles.timeBadge}>
                        <Ionicons name="time-outline" size={14} color="#1565C0" />
                        <Text style={styles.timeBadgeText}>{time}</Text>
                      </View>
                      <View style={[styles.statusBadge, { backgroundColor: status.bg }]}>
                        <Ionicons name={status.icon} size={12} color={status.text} />
                        <Text style={[styles.statusBadgeText, { color: status.text }]}>{status.label}</Text>
                      </View>
                    </View>
                    <Text style={styles.medName}>{item.medicationName}</Text>
                    <Text style={styles.medDosage}>{item.dosage}</Text>
                    {isPending && (
                      <TouchableOpacity style={styles.markBtn} onPress={() => handleMarkTaken(item.logId)}>
                        <Ionicons name="checkmark-circle" size={18} color="#fff" />
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
                <Text style={styles.emptyTitle}>No medications</Text>
                <Text style={styles.emptyDesc}>Add your prescriptions to track them.</Text>
                <TouchableOpacity style={styles.primaryBtnSmall} onPress={() => setShowAddModal(true)}>
                  <Text style={styles.primaryBtnSmallTxt}>Add Medication</Text>
                </TouchableOpacity>
              </View>
            ) : (
              medications.map((med) => {
                const stock = getStockStyle(med.pillsRemaining || 0, med.refillThreshold);
                const isLow = (med.pillsRemaining || 0) <= med.refillThreshold;
                return (
                  <View key={med.id} style={styles.medCard}>
                    <View style={styles.medHeader}>
                      <Text style={styles.medName}>{med.medicationName}</Text>
                      <View style={{ flexDirection: "row", gap: 6 }}>
                        {med.isChronic && (
                          <View style={styles.chronicBadge}>
                            <Text style={styles.chronicBadgeText}>Chronic</Text>
                          </View>
                        )}
                        <View style={[styles.stockBadge, { backgroundColor: stock.bg }]}>
                          <Text style={[styles.stockBadgeText, { color: stock.text }]}>{med.pillsRemaining ?? 0}</Text>
                        </View>
                      </View>
                    </View>
                    <Text style={styles.medMeta}>{med.dosage} • {med.frequency} • {med.form}</Text>
                    {med.doseTimes && <Text style={styles.medTimes}>⏰ {med.doseTimes}</Text>}
                    {renderDayChips(med.daysOfWeek)}
                    {med.instructions && <Text style={styles.medInstructions}>{med.instructions}</Text>}
                    {isLow && (
                      <View style={[styles.stockAlert, { backgroundColor: stock.bg }]}>
                        <Ionicons name="warning-outline" size={14} color={stock.text} />
                        <Text style={[styles.stockAlertText, { color: stock.text }]}>Low stock — time to refill</Text>
                      </View>
                    )}
                  </View>
                );
              })
            )}
          </>
        )}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Add Medication Modal */}
      <Modal visible={showAddModal} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            <View style={styles.sheetHandle} />
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Add Medication</Text>
              <TouchableOpacity onPress={() => setShowAddModal(false)}>
                <Ionicons name="close" size={24} color="#64748B" />
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40 }}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Medication Name *</Text>
                <TextInput style={styles.textInput} value={medName} onChangeText={setMedName} placeholder="e.g. Panadol" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Dosage *</Text>
                <TextInput style={styles.textInput} value={dosage} onChangeText={setDosage} placeholder="e.g. 500mg" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Form</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                  <View style={{ flexDirection: "row", gap: 8 }}>
                    {FORMS.map(f => (
                      <TouchableOpacity key={f} style={[styles.formChip, form === f && styles.formChipActive]} onPress={() => setForm(f)}>
                        <Text style={[styles.formChipTxt, form === f && styles.formChipTxtActive]}>{f}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                </ScrollView>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Days of Week</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  {DAYS.map(d => (
                    <TouchableOpacity key={d} style={[styles.daySelectChip, selectedDays.includes(d) && styles.daySelectChipActive]} onPress={() => toggleDay(d)}>
                      <Text style={[styles.daySelectChipTxt, selectedDays.includes(d) && styles.daySelectChipTxtActive]}>{d.slice(0, 3)}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              <View style={styles.rowInputs}>
                <View style={[styles.inputGroup, { flex: 1 }]}>
                  <Text style={styles.inputLabel}>Times/Day</Text>
                  <View style={styles.counterRow}>
                    <TouchableOpacity style={styles.counterBtn} onPress={() => setTimesPerDay(Math.max(1, timesPerDay - 1))}>
                      <Ionicons name="remove" size={16} color={COLORS.primary} />
                    </TouchableOpacity>
                    <Text style={styles.counterVal}>{timesPerDay}</Text>
                    <TouchableOpacity style={styles.counterBtn} onPress={() => setTimesPerDay(Math.min(6, timesPerDay + 1))}>
                      <Ionicons name="add" size={16} color={COLORS.primary} />
                    </TouchableOpacity>
                  </View>
                </View>
                <View style={[styles.inputGroup, { flex: 2 }]}>
                  <Text style={styles.inputLabel}>Dose Times (comma separated)</Text>
                  <TextInput style={styles.textInput} value={doseTimes} onChangeText={setDoseTimes} placeholder="08:00,14:00,20:00" />
                </View>
              </View>

              <View style={styles.rowInputs}>
                <View style={[styles.inputGroup, { flex: 1 }]}>
                  <Text style={styles.inputLabel}>Start Date</Text>
                  <TextInput style={styles.textInput} value={startDate} onChangeText={setStartDate} />
                </View>
                <View style={[styles.inputGroup, { flex: 1 }]}>
                  <Text style={styles.inputLabel}>End Date (optional)</Text>
                  <TextInput style={styles.textInput} value={endDate} onChangeText={setEndDate} placeholder="YYYY-MM-DD" />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Pills Remaining</Text>
                <TextInput style={styles.textInput} value={pills} onChangeText={setPills} keyboardType="number-pad" placeholder="30" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Instructions</Text>
                <TextInput style={[styles.textInput, { height: 60 }]} value={instructions} onChangeText={setInstructions} placeholder="After meals, etc." multiline />
              </View>

              <View style={styles.switchRow}>
                <Text style={styles.inputLabel}>Chronic (long-term)</Text>
                <Switch value={isChronic} onValueChange={setIsChronic} trackColor={{ false: "#E2E8F0", true: COLORS.primary }} />
              </View>

              <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={saving}>
                {saving ? <ActivityIndicator color="#fff" /> : <Text style={styles.saveBtnTxt}>Save Medication</Text>}
              </TouchableOpacity>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  header: { backgroundColor: COLORS.primary, paddingTop: 60, paddingBottom: 20, paddingHorizontal: 20, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 },
  headerTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  headerTitle: { fontSize: 22, fontWeight: "800", color: "#fff" },
  headerSubtitle: { fontSize: 13, color: "rgba(255,255,255,0.7)", marginTop: 4 },
  addBtn: { width: 40, height: 40, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.2)", justifyContent: "center", alignItems: "center" },
  tabRow: { flexDirection: "row", paddingHorizontal: 20, paddingVertical: 12, gap: 10 },
  tabBtn: { flex: 1, paddingVertical: 10, borderRadius: 12, backgroundColor: "#fff", alignItems: "center", borderWidth: 1, borderColor: "#E2E8F0" },
  tabBtnActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  tabBtnTxt: { fontSize: 13, fontWeight: "600", color: "#64748B" },
  tabBtnTxtActive: { color: "#fff" },
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 20, paddingTop: 8, paddingBottom: 40 },
  emptyCard: { backgroundColor: "#fff", borderRadius: 20, padding: 40, alignItems: "center", marginTop: 20, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  emptyTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginTop: 16 },
  emptyDesc: { fontSize: 13, color: "#94A3B8", marginTop: 6, textAlign: "center" },
  primaryBtnSmall: { backgroundColor: COLORS.primary, paddingHorizontal: 25, paddingVertical: 12, borderRadius: 20, marginTop: 16 },
  primaryBtnSmallTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },
  scheduleCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  scheduleHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  timeBadge: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "#E3F2FD", paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  timeBadgeText: { fontSize: 12, fontWeight: "700", color: "#1565C0" },
  statusBadge: { flexDirection: "row", alignItems: "center", gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  statusBadgeText: { fontSize: 12, fontWeight: "700" },
  medName: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  medDosage: { fontSize: 13, color: "#64748B", marginTop: 4 },
  markBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, backgroundColor: COLORS.primary, paddingVertical: 12, borderRadius: 12, marginTop: 12 },
  markBtnTxt: { color: "#fff", fontSize: 14, fontWeight: "700" },
  medCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  medHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 6 },
  chronicBadge: { backgroundColor: "#E0F2F1", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  chronicBadgeText: { fontSize: 10, fontWeight: "700", color: "#00695C" },
  stockBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  stockBadgeText: { fontSize: 10, fontWeight: "700" },
  medMeta: { fontSize: 13, color: "#64748B" },
  medTimes: { fontSize: 12, color: "#475569", marginTop: 4 },
  medInstructions: { fontSize: 12, color: "#64748B", marginTop: 6, fontStyle: "italic" },
  dayChip: { paddingHorizontal: 6, paddingVertical: 3, borderRadius: 6, backgroundColor: "#F1F5F9" },
  dayChipActive: { backgroundColor: COLORS.primary + "18" },
  dayChipTxt: { fontSize: 10, fontWeight: "600", color: "#94A3B8" },
  dayChipTxtActive: { color: COLORS.primary },
  stockAlert: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 10, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8 },
  stockAlertText: { fontSize: 12, fontWeight: "700" },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.4)", justifyContent: "flex-end" },
  modalSheet: { backgroundColor: "#fff", borderTopLeftRadius: 32, borderTopRightRadius: 32, padding: 24, maxHeight: "90%" },
  sheetHandle: { width: 40, height: 4, backgroundColor: "#E2E8F0", borderRadius: 2, alignSelf: "center", marginBottom: 16 },
  modalHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 20 },
  modalTitle: { fontSize: 20, fontWeight: "800", color: "#1E293B" },
  inputGroup: { marginBottom: 16 },
  inputLabel: { fontSize: 12, fontWeight: "700", color: "#64748B", marginBottom: 6 },
  textInput: { backgroundColor: "#F8FAFC", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 14, color: "#1E293B", borderWidth: 1, borderColor: "#E2E8F0" },
  rowInputs: { flexDirection: "row", gap: 12 },
  formChip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10, backgroundColor: "#F1F5F9", borderWidth: 1, borderColor: "#E2E8F0" },
  formChipActive: { backgroundColor: COLORS.primary + "15", borderColor: COLORS.primary },
  formChipTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  formChipTxtActive: { color: COLORS.primary, fontWeight: "700" },
  daySelectChip: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10, backgroundColor: "#F1F5F9", borderWidth: 1, borderColor: "#E2E8F0" },
  daySelectChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  daySelectChipTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  daySelectChipTxtActive: { color: "#fff", fontWeight: "700" },
  counterRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  counterBtn: { width: 36, height: 36, borderRadius: 10, backgroundColor: "#F1F5F9", justifyContent: "center", alignItems: "center" },
  counterVal: { fontSize: 16, fontWeight: "700", color: "#1E293B", minWidth: 20, textAlign: "center" },
  switchRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 20 },
  saveBtn: { backgroundColor: COLORS.primary, paddingVertical: 16, borderRadius: 16, alignItems: "center", marginTop: 8 },
  saveBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "800" },
});
