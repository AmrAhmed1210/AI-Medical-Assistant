import React, { useEffect, useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, Modal, TextInput, Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { SosBar } from "../../components/SosBar";
import { getMyPatientId } from "../../services/authService";
import {
  getPatientMedications, getMedicationSchedule, createPatientMedication, markMedicationTaken,
  updateMedication, deleteMedication,
  type MedicationTracker, type MedicationScheduleItem, type CreateMedicationPayload,
} from "../../services/medicationService";
import { getAllergies } from "../../services/medicalRecordService";
import {
  requestNotificationPermissions, scheduleMedicationReminders,
} from "../../services/medicationReminderService";
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
  const [editingId, setEditingId] = useState<number | null>(null);
  const [takingId, setTakingId] = useState<number | null>(null);

  // Form state
  const [medName, setMedName] = useState("");
  const [dosage, setDosage] = useState("");
  const [form, setForm] = useState("Pill");
  const [selectedDays, setSelectedDays] = useState<string[]>([...DAYS]);
  const [dayPreset, setDayPreset] = useState<"daily" | "weekdays" | "weekends" | "custom">("daily");
  const [doseTimes, setDoseTimes] = useState("08:00");
  const [doseMode, setDoseMode] = useState<"fixed" | "interval">("fixed");
  const [intervalHours, setIntervalHours] = useState("8");
  const [timesPerDay, setTimesPerDay] = useState("1");
  const [startDate, setStartDate] = useState(new Date().toISOString().split("T")[0]);
  const [durationMode, setDurationMode] = useState<"chronic" | "days" | "until">("days");
  const [durationDays, setDurationDays] = useState("7");
  const [endDate, setEndDate] = useState("");
  const [pills, setPills] = useState("");
  const [instructions, setInstructions] = useState("");

  useEffect(() => { loadData(); requestNotificationPermissions(); }, []);

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
      setSchedule(sched.sort((a, b) => new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime()));
      if (allergies.length > 0) setSosData({ bloodType: "", allergies });
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load medications" });
    } finally { setLoading(false); }
  };

  const toggleDay = (day: string) => {
    setSelectedDays(prev => prev.includes(day) ? prev.filter(d => d !== day) : [...prev, day]);
  };

  const applyDayPreset = (preset: "daily" | "weekdays" | "weekends" | "custom") => {
    setDayPreset(preset);
    if (preset === "daily") setSelectedDays([...DAYS]);
    else if (preset === "weekdays") setSelectedDays(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]);
    else if (preset === "weekends") setSelectedDays(["Friday", "Saturday"]);
  };

  const computedEndDate = (): string | undefined => {
    if (durationMode === "chronic") return undefined;
    if (durationMode === "until") return endDate || undefined;
    if (durationMode === "days" && durationDays) {
      const d = new Date(startDate);
      d.setDate(d.getDate() + Number(durationDays));
      return d.toISOString().split("T")[0];
    }
    return undefined;
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
    const finalEnd = computedEndDate();
    // Compute doseTimes from interval or fixed mode
    let computedDoseTimes = doseTimes.trim();
    let computedTimesPerDay = doseTimes.split(",").filter(Boolean).length;
    if (doseMode === "interval") {
      const hours = Number(intervalHours);
      const count = Number(timesPerDay);
      if (isNaN(hours) || hours < 1 || isNaN(count) || count < 1) {
        Toast.show({ type: "error", text1: "Please enter valid interval and count" });
        return;
      }
      const times: string[] = [];
      let currentHour = 8;
      for (let i = 0; i < count; i++) {
        times.push(`${String(currentHour).padStart(2, "0")}:00`);
        currentHour = (currentHour + hours) % 24;
      }
      computedDoseTimes = times.join(",");
      computedTimesPerDay = count;
    }
    const payload: CreateMedicationPayload = {
      medicationName: medName.trim(), dosage: dosage.trim(), form,
      frequency: `${computedTimesPerDay}x daily`,
      timesPerDay: computedTimesPerDay,
      doseTimes: computedDoseTimes,
      daysOfWeek: selectedDays.join(","),
      startDate,
      endDate: finalEnd,
      pillsRemaining: pills ? Number(pills) : undefined,
      refillThreshold: 5,
      isChronic: durationMode === "chronic",
      instructions: instructions.trim() || undefined,
    };
    try {
      setSaving(true);
      if (editingId) {
        await updateMedication(editingId, payload);
        Toast.show({ type: "success", text1: "Medication updated!" });
      } else {
        const created = await createPatientMedication(patientId, payload);
        await scheduleMedicationReminders(
          created.id, created.medicationName, created.dosage,
          created.doseTimes, created.daysOfWeek, created.startDate, created.endDate
        );
        Toast.show({ type: "success", text1: "Medication added!" });
      }
      resetForm(); setShowAddModal(false); await loadData();
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to save medication" });
    } finally { setSaving(false); }
  };

  const resetForm = () => {
    setMedName(""); setDosage(""); setForm("Pill"); setSelectedDays([...DAYS]); setDayPreset("daily");
    setDoseTimes("08:00"); setDoseMode("fixed"); setIntervalHours("8"); setTimesPerDay("1");
    setStartDate(new Date().toISOString().split("T")[0]);
    setDurationMode("days"); setDurationDays("7"); setEndDate("");
    setPills(""); setInstructions(""); setEditingId(null);
  };

  const handleEdit = (med: MedicationTracker) => {
    setEditingId(med.id);
    setMedName(med.medicationName);
    setDosage(med.dosage);
    setForm(med.form);
    setDoseTimes(med.doseTimes);
    setStartDate(med.startDate);
    setPills(med.pillsRemaining?.toString() ?? "");
    setInstructions(med.instructions ?? "");

    // Days preset
    const days = med.daysOfWeek?.split(",").map(d => d.trim()) ?? [];
    if (days.length === DAYS.length) setDayPreset("daily");
    else if (days.length === 5 && days.includes("Sunday") && days.includes("Monday") && days.includes("Tuesday") && days.includes("Wednesday") && days.includes("Thursday")) setDayPreset("weekdays");
    else if (days.length === 2 && days.includes("Friday") && days.includes("Saturday")) setDayPreset("weekends");
    else setDayPreset("custom");
    setSelectedDays(days.length > 0 ? days : [...DAYS]);

    // Dose mode detection
    const times = med.doseTimes?.split(",").map(t => t.trim()).filter(Boolean) ?? [];
    if (times.length > 1) {
      // Check if times are evenly spaced (interval mode)
      const hours = times.map(t => Number(t.split(":")[0])).sort((a, b) => a - b);
      const diffs: number[] = [];
      for (let i = 1; i < hours.length; i++) diffs.push(hours[i] - hours[i - 1]);
      const allSame = diffs.length > 0 && diffs.every(d => d === diffs[0]);
      if (allSame && diffs[0] > 0) {
        setDoseMode("interval");
        setIntervalHours(String(diffs[0]));
        setTimesPerDay(String(times.length));
      } else {
        setDoseMode("fixed");
        setDoseTimes(med.doseTimes);
        setTimesPerDay(String(times.length));
      }
    } else {
      setDoseMode("fixed");
      setDoseTimes(med.doseTimes || "08:00");
      setTimesPerDay("1");
    }

    // Duration mode
    if (med.isChronic) { setDurationMode("chronic"); setEndDate(""); setDurationDays("7"); }
    else if (med.endDate) { setDurationMode("until"); setEndDate(med.endDate); setDurationDays("7"); }
    else { setDurationMode("days"); setDurationDays("7"); setEndDate(""); }

    setShowAddModal(true);
  };

  const handleDelete = (medId: number, medName: string) => {
    Alert.alert(
      "Delete Medication",
      `Are you sure you want to delete "${medName}"?`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete", style: "destructive",
          onPress: async () => {
            try {
              await deleteMedication(medId);
              Toast.show({ type: "success", text1: "Medication deleted" });
              await loadData();
            } catch (e: any) {
              Toast.show({ type: "error", text1: e.message || "Failed to delete" });
            }
          }
        },
      ]
    );
  };

  const handleMarkTaken = async (logId?: number) => {
    if (!logId) return;
    try {
      setTakingId(logId);
      // Optimistic update
      setSchedule(prev => prev.map(s => s.logId === logId ? { ...s, status: "taken" } : s));
      await markMedicationTaken(logId);
      Toast.show({ type: "success", text1: "Marked as taken!" });
      await loadData();
    } catch (e: any) {
      // Revert on error
      await loadData();
      Toast.show({ type: "error", text1: e.message || "Failed to mark" });
    } finally {
      setTakingId(null);
    }
  };

  const getStatusStyle = (status: string) => {
    const s = status?.toLowerCase() || "";
    if (s === "taken") return { bg: "#E0F2F1", text: "#00695C", label: "Taken", icon: "checkmark-circle" as const, dot: "#10B981" };
    if (s === "missed") return { bg: "#FFEBEE", text: "#C62828", label: "Missed", icon: "close-circle" as const, dot: "#EF4444" };
    if (s === "skipped") return { bg: "#FFF3E0", text: "#E65100", label: "Skipped", icon: "remove-circle" as const, dot: "#F59E0B" };
    return { bg: "#EFF6FF", text: "#2563EB", label: "Pending", icon: "time" as const, dot: COLORS.primary };
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
          <Ionicons name="time-outline" size={14} color={activeTab === "schedule" ? "#fff" : "#64748B"} />
          <Text style={[styles.tabBtnTxt, activeTab === "schedule" && styles.tabBtnTxtActive]}>Today&apos;s Doses</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.tabBtn, activeTab === "active" && styles.tabBtnActive]} onPress={() => setActiveTab("active")}>
          <Ionicons name="medkit-outline" size={14} color={activeTab === "active" ? "#fff" : "#64748B"} />
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
              <View style={{ paddingTop: 8 }}>
                {/* Target Progress Card */}
                {(() => {
                  const todayItems = schedule.map(item => {
                    const scheduledTime = new Date(item.scheduledAt);
                    const isPast = new Date() > scheduledTime;
                    const status = item.status?.toLowerCase() === "pending" && isPast ? "missed" : item.status?.toLowerCase() || "pending";
                    return { ...item, effectiveStatus: status, scheduledTime };
                  });

                  const total = todayItems.length;
                  const taken = todayItems.filter(i => i.effectiveStatus === "taken").length;
                  const progress = total > 0 ? taken / total : 0;
                  
                  const pendingItems = todayItems.filter(i => i.effectiveStatus !== "taken");
                  const takenItems = todayItems.filter(i => i.effectiveStatus === "taken");

                  return (
                    <>
                      <View style={styles.targetCard}>
                        <View style={styles.targetInfo}>
                          <Text style={styles.targetTitle}>Daily Target</Text>
                          <Text style={styles.targetDesc}>
                            {taken === total && total > 0 
                              ? "Awesome! You've taken all your meds today." 
                              : `You have taken ${taken} out of ${total} doses.`}
                          </Text>
                        </View>
                        <View style={styles.progressCircle}>
                          {/* A simple CSS pie chart or just a circular indicator could go here, 
                              but let's use a nice wide progress bar instead for React Native compat */}
                          <Text style={styles.progressText}>{Math.round(progress * 100)}%</Text>
                        </View>
                      </View>
                      
                      <View style={styles.progressBarBg}>
                        <View style={[styles.progressBarFill, { width: `${progress * 100}%` }]} />
                      </View>

                      {pendingItems.length === 0 && total > 0 && (
                        <View style={styles.allDoneMsg}>
                          <Ionicons name="checkmark-circle" size={40} color="#10B981" />
                          <Text style={styles.allDoneTxt}>All done for today!</Text>
                        </View>
                      )}

                      {/* Timeline */}
                      {pendingItems.map((item, i) => {
                        const status = getStatusStyle(item.effectiveStatus);
                        const time = item.scheduledTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                        const isPending = item.effectiveStatus === "pending";
                        const isMissed = item.effectiveStatus === "missed";
                        const isLast = i === pendingItems.length - 1;
                        return (
                          <View key={i} style={styles.timelineRow}>
                            <View style={styles.timeCol}>
                              <View style={[styles.timelineDot, { backgroundColor: status.dot }]} />
                              {!isLast && <View style={styles.timelineLine} />}
                              <Text style={styles.timelineTime}>{time}</Text>
                            </View>

                            <View style={[styles.timelineCard, isPending && styles.timelineCardPending, isMissed && styles.timelineCardMissed]}>
                              <View style={styles.timelineCardHeader}>
                                <Text style={styles.timelineMedName}>{item.medicationName}</Text>
                                <View style={[styles.timelineStatusBadge, { backgroundColor: status.bg }]}>
                                  <Ionicons name={status.icon} size={10} color={status.text} />
                                  <Text style={[styles.timelineStatusText, { color: status.text }]}>{status.label}</Text>
                                </View>
                              </View>
                              <Text style={styles.timelineDosage}>{item.dosage}</Text>
                              {(isPending || isMissed) && (
                                <TouchableOpacity
                                  style={[styles.timelineTakeBtn, takingId === item.logId && { opacity: 0.7 }]}
                                  onPress={() => handleMarkTaken(item.logId)}
                                  disabled={takingId === item.logId}
                                >
                                  {takingId === item.logId ? (
                                    <ActivityIndicator size="small" color="#fff" />
                                  ) : (
                                    <>
                                      <Ionicons name="checkmark" size={16} color="#fff" />
                                      <Text style={styles.timelineTakeBtnTxt}>Mark as Taken</Text>
                                    </>
                                  )}
                                </TouchableOpacity>
                              )}
                            </View>
                          </View>
                        );
                      })}
                      
                      {takenItems.length > 0 && (
                        <View style={styles.takenSection}>
                          <Text style={styles.takenSectionTitle}>Completed Today</Text>
                          {takenItems.map((item, i) => (
                             <View key={`taken-${i}`} style={styles.takenCard}>
                                <Ionicons name="checkmark-circle" size={20} color="#10B981" />
                                <View style={{ flex: 1, marginLeft: 10 }}>
                                  <Text style={styles.takenMedName}>{item.medicationName}</Text>
                                  <Text style={styles.takenMedTime}>{item.dosage} • {item.scheduledTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</Text>
                                </View>
                             </View>
                          ))}
                        </View>
                      )}
                    </>
                  );
                })()}

              </View>
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
                      <View style={{ flexDirection: "row", gap: 6, alignItems: "center" }}>
                        {med.isChronic && (
                          <View style={styles.chronicBadge}>
                            <Text style={styles.chronicBadgeText}>Chronic</Text>
                          </View>
                        )}
                        <View style={[styles.stockBadge, { backgroundColor: stock.bg }]}>
                          <Text style={[styles.stockBadgeText, { color: stock.text }]}>{med.pillsRemaining ?? 0}</Text>
                        </View>
                        <TouchableOpacity style={styles.actionBtn} onPress={() => handleEdit(med)}>
                          <Ionicons name="create-outline" size={16} color="#64748B" />
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(med.id, med.medicationName)}>
                          <Ionicons name="trash-outline" size={16} color="#EF4444" />
                        </TouchableOpacity>
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
              <Text style={styles.modalTitle}>{editingId ? "Edit Medication" : "Add Medication"}</Text>
              <View style={{ flexDirection: "row", gap: 12, alignItems: "center" }}>
                {editingId && (
                  <TouchableOpacity onPress={() => { resetForm(); setShowAddModal(false); }}>
                    <Ionicons name="trash-outline" size={22} color="#EF4444" />
                  </TouchableOpacity>
                )}
                <TouchableOpacity onPress={() => { resetForm(); setShowAddModal(false); }}>
                  <Ionicons name="close" size={24} color="#64748B" />
                </TouchableOpacity>
              </View>
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
                <Text style={styles.inputLabel}>When to take</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  {(["daily", "weekdays", "weekends", "custom"] as const).map(p => (
                    <TouchableOpacity key={p} style={[styles.presetChip, dayPreset === p && styles.presetChipActive]} onPress={() => applyDayPreset(p)}>
                      <Text style={[styles.presetChipTxt, dayPreset === p && styles.presetChipTxtActive]}>
                        {p === "daily" ? "Everyday" : p === "weekdays" ? "Weekdays" : p === "weekends" ? "Weekends" : "Custom"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {dayPreset === "custom" && (
                  <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
                    {DAYS.map(d => (
                      <TouchableOpacity key={d} style={[styles.daySelectChip, selectedDays.includes(d) && styles.daySelectChipActive]} onPress={() => toggleDay(d)}>
                        <Text style={[styles.daySelectChipTxt, selectedDays.includes(d) && styles.daySelectChipTxtActive]}>{d.slice(0, 3)}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Dose Schedule</Text>
                <View style={{ flexDirection: "row", gap: 8 }}>
                  {(["fixed", "interval"] as const).map(m => (
                    <TouchableOpacity key={m} style={[styles.presetChip, doseMode === m && styles.presetChipActive]} onPress={() => setDoseMode(m)}>
                      <Text style={[styles.presetChipTxt, doseMode === m && styles.presetChipTxtActive]}>
                        {m === "fixed" ? "Fixed times" : "Every X hours"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {doseMode === "fixed" ? (
                  <View style={{ marginTop: 10 }}>
                    <Text style={[styles.inputLabel, { marginBottom: 6 }]}>Times (comma separated)</Text>
                    <TextInput style={styles.textInput} value={doseTimes} onChangeText={setDoseTimes} placeholder="08:00, 14:00, 20:00" />
                  </View>
                ) : (
                  <View style={{ marginTop: 10, flexDirection: "row", gap: 12 }}>
                    <View style={{ flex: 1 }}>
                      <Text style={[styles.inputLabel, { marginBottom: 6 }]}>Every (hours)</Text>
                      <TextInput style={styles.textInput} value={intervalHours} onChangeText={setIntervalHours} keyboardType="number-pad" placeholder="8" />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={[styles.inputLabel, { marginBottom: 6 }]}>Times per day</Text>
                      <TextInput style={styles.textInput} value={timesPerDay} onChangeText={setTimesPerDay} keyboardType="number-pad" placeholder="3" />
                    </View>
                  </View>
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Duration</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  {(["days", "until", "chronic"] as const).map(mode => (
                    <TouchableOpacity key={mode} style={[styles.presetChip, durationMode === mode && styles.presetChipActive]} onPress={() => setDurationMode(mode)}>
                      <Text style={[styles.presetChipTxt, durationMode === mode && styles.presetChipTxtActive]}>
                        {mode === "days" ? "For X days" : mode === "until" ? "Until date" : "Chronic"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {durationMode === "days" && (
                  <View style={{ marginTop: 10 }}>
                    <TextInput style={styles.textInput} value={durationDays} onChangeText={setDurationDays} keyboardType="number-pad" placeholder="Number of days e.g. 7" />
                  </View>
                )}
                {durationMode === "until" && (
                  <View style={{ marginTop: 10 }}>
                    <TextInput style={styles.textInput} value={endDate} onChangeText={setEndDate} placeholder="End date YYYY-MM-DD" />
                  </View>
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Start Date</Text>
                <TextInput style={styles.textInput} value={startDate} onChangeText={setStartDate} placeholder="YYYY-MM-DD" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Pills Remaining</Text>
                <TextInput style={styles.textInput} value={pills} onChangeText={setPills} keyboardType="number-pad" placeholder="30" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Instructions</Text>
                <TextInput style={[styles.textInput, { height: 60 }]} value={instructions} onChangeText={setInstructions} placeholder="After meals, with water, etc." multiline />
              </View>

              <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={saving}>
                {saving ? <ActivityIndicator color="#fff" /> : <Text style={styles.saveBtnTxt}>{editingId ? "Update Medication" : "Save Medication"}</Text>}
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
  tabBtn: { flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, paddingVertical: 10, borderRadius: 12, backgroundColor: "#fff", borderWidth: 1, borderColor: "#E2E8F0" },
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

  summaryRow: { flexDirection: "row", justifyContent: "center", alignItems: "center", backgroundColor: "#fff", borderRadius: 16, paddingVertical: 14, marginBottom: 16, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 8, elevation: 2 },
  summaryItem: { flex: 1, alignItems: "center" },
  summaryNum: { fontSize: 20, fontWeight: "800", color: "#1E293B" },
  summaryLabel: { fontSize: 11, fontWeight: "600", color: "#94A3B8", marginTop: 2 },
  summaryDivider: { width: 1, height: 30, backgroundColor: "#E2E8F0" },

  timelineRow: { flexDirection: "row", marginBottom: 16 },
  timeCol: { width: 56, alignItems: "center", position: "relative", paddingTop: 4 },
  timelineDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 2, borderColor: "#fff" },
  timelineLine: { position: "absolute", top: 16, bottom: -16, width: 2, backgroundColor: "#E2E8F0", left: 27 },
  timelineTime: { fontSize: 11, fontWeight: "700", color: "#64748B", marginTop: 4 },
  timelineCard: { flex: 1, backgroundColor: "#fff", borderRadius: 16, padding: 14, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 8, elevation: 2 },
  timelineCardPending: { borderLeftWidth: 3, borderLeftColor: COLORS.primary },
  timelineCardMissed: { borderLeftWidth: 3, borderLeftColor: "#EF4444" },
  timelineCardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 4 },
  timelineMedName: { fontSize: 15, fontWeight: "700", color: "#1E293B", flex: 1 },
  timelineStatusBadge: { flexDirection: "row", alignItems: "center", gap: 3, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  timelineStatusText: { fontSize: 10, fontWeight: "700" },
  timelineDosage: { fontSize: 12, color: "#64748B", marginTop: 2 },
  timelineTakeBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, backgroundColor: COLORS.primary, paddingVertical: 10, borderRadius: 10, marginTop: 10 },
  timelineTakeBtnTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },

  targetCard: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 8, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 8, elevation: 2 },
  targetInfo: { flex: 1, paddingRight: 12 },
  targetTitle: { fontSize: 16, fontWeight: "800", color: "#1E293B", marginBottom: 4 },
  targetDesc: { fontSize: 12, color: "#64748B", lineHeight: 18 },
  progressCircle: { width: 50, height: 50, borderRadius: 25, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: COLORS.primary },
  progressText: { fontSize: 13, fontWeight: "800", color: COLORS.primary },
  progressBarBg: { height: 8, backgroundColor: "#E2E8F0", borderRadius: 4, marginBottom: 20 },
  progressBarFill: { height: "100%", backgroundColor: COLORS.primary, borderRadius: 4 },
  allDoneMsg: { alignItems: "center", paddingVertical: 20 },
  allDoneTxt: { fontSize: 15, fontWeight: "700", color: "#10B981", marginTop: 8 },
  takenSection: { marginTop: 20, paddingTop: 20, borderTopWidth: 1, borderTopColor: "#E2E8F0" },
  takenSectionTitle: { fontSize: 15, fontWeight: "700", color: "#1E293B", marginBottom: 12 },
  takenCard: { flexDirection: "row", alignItems: "center", backgroundColor: "#F8FAFC", padding: 12, borderRadius: 12, marginBottom: 8, borderWidth: 1, borderColor: "#E2E8F0" },
  takenMedName: { fontSize: 14, fontWeight: "600", color: "#1E293B" },
  takenMedTime: { fontSize: 12, color: "#64748B", marginTop: 2 },

  medCard: { backgroundColor: "#fff", borderRadius: 16, padding: 16, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  medHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 6 },
  medName: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  medMeta: { fontSize: 13, color: "#64748B" },
  medTimes: { fontSize: 12, color: "#475569", marginTop: 4 },
  medInstructions: { fontSize: 12, color: "#64748B", marginTop: 6, fontStyle: "italic" },
  chronicBadge: { backgroundColor: "#E0F2F1", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  chronicBadgeText: { fontSize: 10, fontWeight: "700", color: "#00695C" },
  stockBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  stockBadgeText: { fontSize: 10, fontWeight: "700" },
  dayChip: { paddingHorizontal: 6, paddingVertical: 3, borderRadius: 6, backgroundColor: "#F1F5F9" },
  dayChipActive: { backgroundColor: COLORS.primary + "18" },
  dayChipTxt: { fontSize: 10, fontWeight: "600", color: "#94A3B8" },
  dayChipTxtActive: { color: COLORS.primary },
  stockAlert: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 10, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8 },
  stockAlertText: { fontSize: 12, fontWeight: "700" },
  actionBtn: { width: 32, height: 32, borderRadius: 8, backgroundColor: "#F1F5F9", justifyContent: "center", alignItems: "center" },

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
  presetChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 10, backgroundColor: "#F1F5F9", borderWidth: 1, borderColor: "#E2E8F0" },
  presetChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  presetChipTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  presetChipTxtActive: { color: "#fff", fontWeight: "700" },
  daySelectChip: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10, backgroundColor: "#F1F5F9", borderWidth: 1, borderColor: "#E2E8F0" },
  daySelectChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  daySelectChipTxt: { fontSize: 12, fontWeight: "600", color: "#64748B" },
  daySelectChipTxtActive: { color: "#fff", fontWeight: "700" },
  saveBtn: { backgroundColor: COLORS.primary, paddingVertical: 16, borderRadius: 16, alignItems: "center", marginTop: 8 },
  saveBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "800" },
});
