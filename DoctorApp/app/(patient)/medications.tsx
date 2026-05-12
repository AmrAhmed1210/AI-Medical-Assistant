import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, Modal, TextInput, Alert, Dimensions, Animated, Platform, LayoutAnimation
} from "react-native";
import { 
  Pill, Timer, Calendar, Activity, ChevronRight, 
  Plus, CheckCircle2, AlertTriangle, Trash2, Edit3, 
  User, Search, Bell, X, ShieldCheck, Sparkles, Clock, ArrowRight, Info, ChevronDown, ChevronUp
} from "lucide-react-native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
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
  requestNotificationPermissions, scheduleMedicationReminders, cancelMedicationReminders,
} from "../../services/medicationReminderService";
import Toast from "react-native-toast-message";
import { useLanguage } from "../../context/LanguageContext";
import { onNewMedication } from "../../services/signalr";

const { width, height } = Dimensions.get("window");

const DAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
const FORMS = ["Pill", "Syrup", "Injection", "Inhaler", "Cream", "Drops", "Patch", "Powder"];

export default function MedicationsScreen() {
  const { tr, isRTL } = useLanguage();
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
  const [expandedMedId, setExpandedMedId] = useState<number | null>(null);

  const scrollY = useRef(new Animated.Value(0)).current;

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

  useEffect(() => { 
    loadData(); 
    requestNotificationPermissions(); 
    
    const unsubMed = onNewMedication(() => {
      loadData();
    });

    return () => {
      unsubMed();
    };
  }, []);

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

  const toggleExpand = (id: number) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpandedMedId(expandedMedId === id ? null : id);
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

    const days = med.daysOfWeek?.split(",").map(d => d.trim()) ?? [];
    if (days.length === DAYS.length) setDayPreset("daily");
    else if (days.length === 5 && days.includes("Sunday") && days.includes("Monday") && days.includes("Tuesday") && days.includes("Wednesday") && days.includes("Thursday")) setDayPreset("weekdays");
    else if (days.length === 2 && days.includes("Friday") && days.includes("Saturday")) setDayPreset("weekends");
    else setDayPreset("custom");
    setSelectedDays(days.length > 0 ? days : [...DAYS]);

    const times = med.doseTimes?.split(",").map(t => t.trim()).filter(Boolean) ?? [];
    if (times.length > 1) {
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
              await cancelMedicationReminders(medId);
              // Optimistically update schedule to remove items for the deleted medication
              setSchedule(prev => prev.filter(s => s.medicationTrackerId !== medId));
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
      setSchedule(prev => prev.map(s => s.logId === logId ? { ...s, status: "taken" } : s));
      await markMedicationTaken(logId);
      Toast.show({ type: "success", text1: "Marked as taken!" });
      await loadData();
    } catch (e: any) {
      await loadData();
      Toast.show({ type: "error", text1: e.message || "Failed to mark" });
    } finally {
      setTakingId(null);
    }
  };

  const getStatusStyle = (status: string) => {
    const s = status?.toLowerCase() || "";
    if (s === "taken") return { bg: "#ECFDF5", text: "#059669", label: tr("taken"), icon: "checkmark-circle" as const, dot: "#10B981" };
    if (s === "missed") return { bg: "#FEF2F2", text: "#DC2626", label: tr("missed"), icon: "close-circle" as const, dot: "#EF4444" };
    if (s === "accumulated") return { bg: "#FFF7ED", text: "#EA580C", label: tr("accumulated"), icon: "alert-circle" as const, dot: "#F97316" };
    if (s === "skipped") return { bg: "#FFFBEB", text: "#D97706", label: tr("skipped"), icon: "remove-circle" as const, dot: "#F59E0B" };
    return { bg: "#EFF6FF", text: "#2563EB", label: tr("pending"), icon: "time" as const, dot: "#3B82F6" };
  };

  const getStockStyle = (pills: number, threshold: number) => {
    if (pills <= 3) return { bg: "#FEF2F2", text: "#DC2626" };
    if (pills <= threshold) return { bg: "#FFFBEB", text: "#D97706" };
    return { bg: "#ECFDF5", text: "#059669" };
  };

  const headerHeight = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [260, 180],
    extrapolate: 'clamp',
  });

  const tabsTranslateY = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [0, -65],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [1, 0.98],
    extrapolate: 'clamp',
  });

  const renderDayChips = (daysStr?: string) => {
    const days = daysStr?.split(",").map(d => d.trim()).filter(Boolean) ?? [];
    const shortNames: Record<string, string> = { Saturday: "Sat", Sunday: "Sun", Monday: "Mon", Tuesday: "Tue", Wednesday: "Wed", Thursday: "Thu", Friday: "Fri" };
    return (
      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
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

  if (loading && !medications.length) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#059669" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* ANIMATED LUXURY HEADER - INCREASED Z-INDEX FOR TOUCHES */}
      <Animated.View style={[styles.magicHeader, { height: headerHeight, opacity: headerOpacity, zIndex: 1000 }]}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={StyleSheet.absoluteFill}>
          <View style={styles.headerTop}>
            <View>
              <Text style={styles.headerGreet}>Pharmacy Concierge</Text>
              <View style={styles.nameRow}>
                <Text style={styles.headerTitle}>{tr("my_medications")}</Text>
                <View style={styles.liveBadge}><Text style={styles.liveBadgeText}>LIVE</Text></View>
              </View>
            </View>
            <TouchableOpacity 
              style={styles.premiumAddBtn} 
              onPress={() => { setShowAddModal(true); }}
              activeOpacity={0.7}
            >
              <LinearGradient colors={["#fff", "#F8FAFC"]} style={styles.addBtnGradient}>
                <Plus size={24} color="#059669" />
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {/* Decorative Mesh Blobs */}
          <View style={[styles.meshBlob, { top: -60, right: -40, width: 220, height: 220, backgroundColor: 'rgba(16, 185, 129, 0.2)' }]} />
          <View style={[styles.meshBlob, { bottom: -40, left: -20, width: 180, height: 180, backgroundColor: 'rgba(52, 211, 153, 0.15)' }]} />
        </LinearGradient>
      </Animated.View>

      {sosData && <SosBar bloodType={sosData.bloodType} allergies={sosData.allergies} />}

      <Animated.View style={[styles.pillTabsContainer, { transform: [{ translateY: tabsTranslateY }] }]}>
        <View style={styles.pillTabsBackground}>
          <TouchableOpacity 
            style={[styles.pillTab, activeTab === "schedule" && styles.pillTabActive]} 
            onPress={() => setActiveTab("schedule")}
          >
            <Clock size={16} color={activeTab === "schedule" ? "#059669" : "#94A3B8"} />
            <Text style={[styles.pillTabText, activeTab === "schedule" && styles.pillTabTextActive]}>{tr("todays_doses")}</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.pillTab, activeTab === "active" && styles.pillTabActive]} 
            onPress={() => setActiveTab("active")}
          >
            <Pill size={16} color={activeTab === "active" ? "#059669" : "#94A3B8"} />
            <Text style={[styles.pillTabText, activeTab === "active" && styles.pillTabTextActive]}>{tr("my_prescriptions")}</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>

      <Animated.ScrollView 
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
        style={styles.scroll} 
        contentContainerStyle={{ paddingTop: 280, paddingBottom: 120 }} 
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.contentOverlap}>
          {activeTab === "schedule" ? (
            <>
              {schedule.length === 0 ? (
                <View style={styles.emptyCard}>
                  <View style={styles.emptyIconBox}>
                    <Pill size={48} color="#CBD5E1" />
                  </View>
                  <Text style={styles.emptyTitle}>{tr("no_doses_today")}</Text>
                  <Text style={styles.emptyDesc}>{tr("add_medication_hint")}</Text>
                  <TouchableOpacity style={styles.primaryBtnSmall} onPress={() => setShowAddModal(true)}>
                    <LinearGradient colors={["#059669", "#047857"]} style={styles.btnGradient}>
                      <Text style={styles.primaryBtnSmallTxt}>{tr("add_medication")}</Text>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              ) : (
                <View>
                  {(() => {
                    const todayItems = schedule.map((item, idx, arr) => {
                      const scheduledTime = new Date(item.scheduledAt);
                      const isPast = new Date() > scheduledTime;
                      let status = item.status?.toLowerCase() || "pending";
                      
                      if (status === "pending" && isPast) {
                        // Check if there's a later dose for the same medication that is also in the past
                        const laterPastDose = arr.find(other => 
                          other.medicationTrackerId === item.medicationTrackerId && 
                          new Date(other.scheduledAt) > scheduledTime && 
                          new Date(other.scheduledAt) <= new Date()
                        );
                        
                        if (laterPastDose) {
                          status = "accumulated";
                        } else {
                          status = "missed";
                        }
                      }
                      
                      return { ...item, effectiveStatus: status, scheduledTime };
                    });

                    const total = todayItems.length;
                    const taken = todayItems.filter(i => i.effectiveStatus === "taken").length;
                    const progress = total > 0 ? taken / total : 0;
                    
                    const pendingItems = todayItems.filter(i => i.effectiveStatus !== "taken");
                    const takenItems = todayItems.filter(i => i.effectiveStatus === "taken");

                    return (
                      <>
                        <View style={styles.tipWrapper}>
                          <LinearGradient
                            colors={["rgba(255, 255, 255, 0.95)", "rgba(240, 253, 244, 0.9)"]}
                            style={styles.magicTargetCard}
                          >
                            <View style={styles.targetIconBox}>
                              <View style={styles.innerIconBox}>
                                <ShieldCheck size={22} color="#059669" />
                              </View>
                            </View>
                            <View style={{ flex: 1 }}>
                              <Text style={styles.targetTitleText}>{tr("daily_target")}</Text>
                              <Text style={styles.targetDescText}>
                                {taken === total && total > 0 
                                  ? tr("all_done_today")
                                  : `${tr("taken_of")} ${taken}/${total} ${tr("doses")}`}
                              </Text>
                            </View>
                            <View style={styles.progressCircle}>
                              <Text style={styles.progressTextValue}>{Math.round(progress * 100)}%</Text>
                            </View>
                          </LinearGradient>
                        </View>
                        
                        <View style={styles.progressBarContainer}>
                          <View style={[styles.progressBarValue, { width: `${progress * 100}%` }]} />
                        </View>

                        {pendingItems.length === 0 && total > 0 && (
                          <View style={styles.successMessage}>
                            <Sparkles size={32} color="#10B981" />
                            <Text style={styles.successText}>{tr("all_done_today")}</Text>
                          </View>
                        )}

                        {/* Timeline */}
                        <View style={styles.timelineSection}>
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
                                    <View style={{ flex: 1 }}>
                                      <Text style={styles.timelineMedName}>{item.medicationName}</Text>
                                      <Text style={styles.timelineDosageText}>{item.dosage}</Text>
                                    </View>
                                    <View style={[styles.timelineStatusBadge, { backgroundColor: status.bg }]}>
                                      <Text style={[styles.timelineStatusText, { color: status.text }]}>{status.label}</Text>
                                    </View>
                                  </View>
                                  
                                  <View style={styles.timelineCardFooter}>
                                    <View style={styles.scheduleInfoRow}>
                                       <Clock size={12} color="#64748B" />
                                       <Text style={styles.scheduleInfoText}>{time}</Text>
                                    </View>
                                    {(isPending || isMissed) && (
                                      <TouchableOpacity
                                        style={[styles.timelineTakeBtn, takingId === item.logId && { opacity: 0.7 }]}
                                        onPress={() => handleMarkTaken(item.logId)}
                                        disabled={takingId === item.logId}
                                      >
                                        <LinearGradient colors={["#064E3B", "#059669"]} style={styles.takeBtnGradient}>
                                          {takingId === item.logId ? (
                                            <ActivityIndicator size="small" color="#fff" />
                                          ) : (
                                            <Text style={styles.timelineTakeBtnTxt}>{tr("mark_taken")}</Text>
                                          )}
                                        </LinearGradient>
                                      </TouchableOpacity>
                                    )}
                                  </View>
                                </View>
                              </View>
                            )
                          })}
                        </View>
                        
                        {takenItems.length > 0 && (
                          <View style={styles.takenSection}>
                            <View style={styles.sectionHeaderRow}>
                              <Text style={styles.takenSectionTitle}>{tr("completed_today")}</Text>
                              <View style={styles.countBadge}><Text style={styles.countBadgeText}>{takenItems.length}</Text></View>
                            </View>
                            {takenItems.map((item, i) => (
                               <View key={`taken-${i}`} style={styles.takenCard}>
                                  <View style={styles.takenCheckIcon}>
                                    <CheckCircle2 size={20} color="#059669" />
                                  </View>
                                  <View style={{ flex: 1, marginLeft: 12 }}>
                                    <Text style={styles.takenMedName}>{item.medicationName}</Text>
                                    <Text style={styles.takenMedTime}>{item.dosage} • {item.scheduledTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</Text>
                                  </View>
                               </View>
                            ))}
                          </View>
                        )}
                      </>
                    )
                  })()}

                </View>
              )}
            </>
          ) : (
            <>
              {medications.length === 0 ? (
                <View style={styles.emptyCard}>
                  <View style={styles.emptyIconBox}>
                    <Ionicons name="medkit-outline" size={48} color="#CBD5E1" />
                  </View>
                  <Text style={styles.emptyTitle}>{tr("no_medications")}</Text>
                  <Text style={styles.emptyDesc}>{tr("add_medication_hint")}</Text>
                  <TouchableOpacity style={styles.primaryBtnSmall} onPress={() => setShowAddModal(true)}>
                    <LinearGradient colors={["#059669", "#047857"]} style={styles.btnGradient}>
                      <Text style={styles.primaryBtnSmallTxt}>{tr("add_medication")}</Text>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={styles.medsList}>
                  {medications.map((med) => {
                    const stock = getStockStyle(med.pillsRemaining || 0, med.refillThreshold);
                    const isLow = (med.pillsRemaining || 0) <= med.refillThreshold;
                    const isExpanded = expandedMedId === med.id;
                    return (
                      <TouchableOpacity 
                        key={med.id} 
                        style={[styles.medCard, isExpanded && styles.medCardExpanded]}
                        onPress={() => toggleExpand(med.id)}
                        activeOpacity={0.9}
                      >
                        <View style={styles.medHeaderMain}>
                          <View style={styles.medIconBox}>
                             <Pill size={22} color="#059669" />
                          </View>
                          <View style={{ flex: 1, marginLeft: 12 }}>
                            <Text style={styles.medName}>{med.medicationName}</Text>
                            <Text style={styles.medCompactSub}>{med.dosage} • {med.frequency}</Text>
                          </View>
                          <View style={styles.headerRightInfo}>
                             <View style={[styles.stockBadgeCompact, { backgroundColor: stock.bg }]}>
                               <Text style={[styles.stockBadgeText, { color: stock.text }]}>{med.pillsRemaining ?? 0} Left</Text>
                             </View>
                             {isExpanded ? <ChevronUp size={20} color="#94A3B8" /> : <ChevronDown size={20} color="#94A3B8" />}
                          </View>
                        </View>

                        {isExpanded && (
                          <View style={styles.expandedContent}>
                            <View style={styles.divider} />
                            <View style={styles.medBodyGrid}>
                               <View style={styles.gridItem}>
                                  <Text style={styles.gridLabel}>FORM</Text>
                                  <Text style={styles.gridValue}>{med.form}</Text>
                               </View>
                               <View style={styles.gridItem}>
                                  <Text style={styles.gridLabel}>TYPE</Text>
                                  <Text style={styles.gridValue}>{med.isChronic ? "Chronic" : "Regular"}</Text>
                               </View>
                               <View style={styles.gridItem}>
                                  <Text style={styles.gridLabel}>REFILL AT</Text>
                                  <Text style={styles.gridValue}>{med.refillThreshold}</Text>
                                </View>
                            </View>

                            {med.doseTimes && (
                              <View style={styles.scheduleRow}>
                                <Clock size={14} color="#64748B" />
                                <Text style={styles.scheduleText}>{med.doseTimes.split(',').join(' • ')}</Text>
                              </View>
                            )}

                            {renderDayChips(med.daysOfWeek)}

                            {med.instructions && (
                              <View style={styles.noteBox}>
                                <Info size={14} color="#64748B" />
                                <Text style={styles.noteText}>{med.instructions}</Text>
                              </View>
                            )}

                            <View style={styles.cardActionsRow}>
                              <TouchableOpacity style={styles.actionBtnOutline} onPress={() => handleEdit(med)}>
                                <Edit3 size={16} color="#64748B" />
                                <Text style={styles.actionBtnText}>Edit</Text>
                              </TouchableOpacity>
                              <TouchableOpacity style={styles.actionBtnOutline} onPress={() => handleDelete(med.id, med.medicationName)}>
                                <Trash2 size={16} color="#EF4444" />
                                <Text style={[styles.actionBtnText, { color: "#EF4444" }]}>Delete</Text>
                              </TouchableOpacity>
                            </View>

                            {isLow && (
                              <View style={[styles.alertBar, { backgroundColor: stock.bg, marginTop: 15 }]}>
                                <AlertTriangle size={14} color={stock.text} />
                                <Text style={[styles.alertText, { color: stock.text }]}>Low stock! Consider refilling soon.</Text>
                              </View>
                            )}
                          </View>
                        )}
                      </TouchableOpacity>
                    )
                  })}
                </View>
              )}
            </>
          )}
        </View>
      </Animated.ScrollView>

      {/* Add Medication Modal */}
      <Modal visible={showAddModal} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            <View style={styles.sheetHandle} />
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{editingId ? "Edit Medication" : "Add Medication"}</Text>
              <TouchableOpacity style={styles.modalCloseBtn} onPress={() => { resetForm(); setShowAddModal(false); }}>
                <X size={24} color="#64748B" />
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 60 }}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Medication Name *</Text>
                <TextInput style={styles.textInput} value={medName} onChangeText={setMedName} placeholder="e.g. Panadol" placeholderTextColor="#CBD5E1" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Dosage *</Text>
                <TextInput style={styles.textInput} value={dosage} onChangeText={setDosage} placeholder="e.g. 500mg" placeholderTextColor="#CBD5E1" />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Form</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10 }}>
                  {FORMS.map(f => (
                    <TouchableOpacity key={f} style={[styles.formChip, form === f && styles.formChipActive]} onPress={() => setForm(f)}>
                      <Text style={[styles.formChipTxt, form === f && styles.formChipTxtActive]}>{f}</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>When to take</Text>
                <View style={styles.presetGrid}>
                  {(["daily", "weekdays", "weekends", "custom"] as const).map(p => (
                    <TouchableOpacity key={p} style={[styles.presetChip, dayPreset === p && styles.presetChipActive]} onPress={() => applyDayPreset(p)}>
                      <Text style={[styles.presetChipTxt, dayPreset === p && styles.presetChipTxtActive]}>
                        {p === "daily" ? "Everyday" : p === "weekdays" ? "Weekdays" : p === "weekends" ? "Weekends" : "Custom"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {dayPreset === "custom" && (
                  <View style={styles.daySelectorRow}>
                    {DAYS.map(d => (
                      <TouchableOpacity key={d} style={[styles.daySelectCircle, selectedDays.includes(d) && styles.daySelectCircleActive]} onPress={() => toggleDay(d)}>
                        <Text style={[styles.daySelectCircleTxt, selectedDays.includes(d) && styles.daySelectCircleTxtActive]}>{d.slice(0, 1)}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Schedule Mode</Text>
                <View style={styles.presetGrid}>
                  {(["fixed", "interval"] as const).map(m => (
                    <TouchableOpacity key={m} style={[styles.presetChip, doseMode === m && styles.presetChipActive]} onPress={() => setDoseMode(m)}>
                      <Text style={[styles.presetChipTxt, doseMode === m && styles.presetChipTxtActive]}>
                        {m === "fixed" ? "Fixed Times" : "Interval (Every X hrs)"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {doseMode === "fixed" ? (
                  <View style={{ marginTop: 15 }}>
                    <Text style={styles.smallInputLabel}>Dose Times (e.g. 08:00, 20:00)</Text>
                    <TextInput style={styles.textInput} value={doseTimes} onChangeText={setDoseTimes} placeholder="08:00, 14:00, 20:00" placeholderTextColor="#CBD5E1" />
                  </View>
                ) : (
                  <View style={styles.intervalRow}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.smallInputLabel}>Interval (hrs)</Text>
                      <TextInput style={styles.textInput} value={intervalHours} onChangeText={setIntervalHours} keyboardType="number-pad" placeholder="8" placeholderTextColor="#CBD5E1" />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.smallInputLabel}>Times/Day</Text>
                      <TextInput style={styles.textInput} value={timesPerDay} onChangeText={setTimesPerDay} keyboardType="number-pad" placeholder="3" placeholderTextColor="#CBD5E1" />
                    </View>
                  </View>
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Duration</Text>
                <View style={styles.presetGrid}>
                  {(["days", "until", "chronic"] as const).map(mode => (
                    <TouchableOpacity key={mode} style={[styles.presetChip, durationMode === mode && styles.presetChipActive]} onPress={() => setDurationMode(mode)}>
                      <Text style={[styles.presetChipTxt, durationMode === mode && styles.presetChipTxtActive]}>
                        {mode === "days" ? "Days Count" : mode === "until" ? "End Date" : "Chronic"}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {durationMode === "days" && (
                  <TextInput style={[styles.textInput, { marginTop: 15 }]} value={durationDays} onChangeText={setDurationDays} keyboardType="number-pad" placeholder="Number of days" placeholderTextColor="#CBD5E1" />
                )}
                {durationMode === "until" && (
                  <TextInput style={[styles.textInput, { marginTop: 15 }]} value={endDate} onChangeText={setEndDate} placeholder="YYYY-MM-DD" placeholderTextColor="#CBD5E1" />
                )}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Instructions</Text>
                <TextInput style={[styles.textInput, { height: 80, textAlignVertical: 'top' }]} value={instructions} onChangeText={setInstructions} placeholder="e.g. Take after meal" placeholderTextColor="#CBD5E1" multiline />
              </View>

              <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={saving}>
                <LinearGradient colors={["#059669", "#047857"]} style={styles.saveBtnGradient}>
                  {saving ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.saveBtnTxt}>{editingId ? "Update Prescription" : "Save Prescription"}</Text>}
                </LinearGradient>
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
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#fff" },
  magicHeader: { position: 'absolute', top: 0, left: 0, right: 0, overflow: 'hidden' },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 25, paddingTop: 60, zIndex: 10 },
  headerGreet: { fontSize: 12, color: "rgba(255,255,255,0.7)", fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1 },
  nameRow: { flexDirection: 'row', alignItems: 'center', marginTop: 4 },
  headerTitle: { fontSize: 24, fontWeight: '900', color: '#fff', letterSpacing: -0.5 },
  liveBadge: { backgroundColor: '#FDE047', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8, marginLeft: 10 },
  liveBadgeText: { fontSize: 10, fontWeight: '900', color: '#064E3B' },
  premiumAddBtn: { width: 48, height: 48, borderRadius: 14, overflow: 'hidden', elevation: 10, shadowColor: '#000', shadowOpacity: 0.2 },
  addBtnGradient: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  meshBlob: { position: 'absolute', borderRadius: 150 },

  pillTabsContainer: { paddingHorizontal: 25, position: 'absolute', top: 205, left: 0, right: 0, zIndex: 2000 },
  pillTabsBackground: { flexDirection: 'row', backgroundColor: '#fff', borderRadius: 25, padding: 6, elevation: 15, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 20 },
  pillTab: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, height: 48, borderRadius: 20 },
  pillTabActive: { backgroundColor: '#F0FDF4' },
  pillTabText: { fontSize: 13, fontWeight: '700', color: '#94A3B8' },
  pillTabTextActive: { color: '#059669' },

  scroll: { flex: 1 },
  contentOverlap: { minHeight: height },

  emptyCard: { backgroundColor: '#fff', borderRadius: 35, padding: 40, alignItems: 'center', marginHorizontal: 25, marginTop: 40, borderWidth: 1, borderColor: '#F1F5F9', elevation: 8, shadowOpacity: 0.05 },
  emptyIconBox: { width: 80, height: 80, borderRadius: 30, backgroundColor: '#F8FAFC', justifyContent: 'center', alignItems: 'center', marginBottom: 20 },
  emptyTitle: { fontSize: 20, fontWeight: '900', color: '#1E293B' },
  emptyDesc: { fontSize: 15, color: '#94A3B8', marginTop: 10, textAlign: 'center', lineHeight: 24 },
  primaryBtnSmall: { borderRadius: 22, overflow: 'hidden', marginTop: 30, elevation: 8, shadowColor: '#059669', shadowOpacity: 0.3 },
  btnGradient: { paddingHorizontal: 35, paddingVertical: 18, justifyContent: 'center', alignItems: 'center' },
  primaryBtnSmallTxt: { color: "#fff", fontSize: 15, fontWeight: "900" },

  tipWrapper: { paddingHorizontal: 25, marginTop: 25 },
  magicTargetCard: { flexDirection: 'row', alignItems: 'center', padding: 22, borderRadius: 32, elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.15, shadowRadius: 25, borderWidth: 1, borderColor: 'rgba(255,255,255,0.9)' },
  targetIconBox: { width: 56, height: 56, borderRadius: 20, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center' },
  innerIconBox: { width: 42, height: 42, borderRadius: 16, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', elevation: 4 },
  targetTitleText: { fontSize: 15, fontWeight: '900', color: '#064E3B', marginBottom: 4 },
  targetDescText: { fontSize: 13, color: '#059669', lineHeight: 18, fontWeight: '700' },
  progressCircle: { width: 64, height: 64, borderRadius: 32, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', borderWidth: 5, borderColor: '#10B981', elevation: 5 },
  progressTextValue: { fontSize: 16, fontWeight: '900', color: '#064E3B' },
  progressBarContainer: { marginHorizontal: 35, height: 10, backgroundColor: '#F1F5F9', borderRadius: 5, marginTop: 20, marginBottom: 30, overflow: 'hidden' },
  progressBarValue: { height: '100%', backgroundColor: '#10B981', borderRadius: 5 },
  successMessage: { alignItems: 'center', paddingVertical: 30, backgroundColor: '#ECFDF5', borderRadius: 40, marginHorizontal: 25, marginBottom: 30, borderWidth: 1, borderColor: '#D1FAE5' },
  successText: { fontSize: 18, fontWeight: "900", color: "#065F46", marginTop: 15 },

  timelineSection: { paddingHorizontal: 25 },
  timelineRow: { flexDirection: "row", marginBottom: 20 },
  timeCol: { width: 70, alignItems: "center", position: "relative" },
  timelineDot: { width: 16, height: 16, borderRadius: 8, backgroundColor: '#059669', borderWidth: 4, borderColor: '#fff', zIndex: 10, elevation: 8 },
  timelineLine: { position: "absolute", top: 16, bottom: -30, width: 2, backgroundColor: '#E2E8F0', left: 34 },
  timelineTime: { fontSize: 12, fontWeight: "900", color: "#1E293B", marginTop: 10 },
  timelineCard: { flex: 1, backgroundColor: '#fff', borderRadius: 30, padding: 20, borderWidth: 1, borderColor: '#F1F5F9', elevation: 6, shadowColor: '#000', shadowOpacity: 0.05 },
  timelineCardPending: { borderLeftWidth: 6, borderLeftColor: '#3B82F6' },
  timelineCardMissed: { borderLeftWidth: 6, borderLeftColor: "#EF4444" },
  timelineCardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  timelineMedName: { fontSize: 17, fontWeight: '900', color: '#1E293B', marginBottom: 2 },
  timelineDosageText: { fontSize: 13, color: '#64748B', fontWeight: '600' },
  timelineStatusBadge: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 10, alignSelf: 'flex-start' },
  timelineStatusText: { fontSize: 10, fontWeight: "900", textTransform: 'uppercase' },
  timelineCardFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 15 },
  scheduleInfoRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  scheduleInfoText: { fontSize: 12, color: '#64748B', fontWeight: '700' },
  timelineTakeBtn: { borderRadius: 14, overflow: 'hidden', elevation: 4, shadowColor: '#064E3B', shadowOpacity: 0.2 },
  takeBtnGradient: { paddingHorizontal: 20, paddingVertical: 10, justifyContent: 'center', alignItems: 'center' },
  timelineTakeBtnTxt: { color: "#fff", fontSize: 13, fontWeight: "800" },

  takenSection: { marginTop: 40, paddingHorizontal: 25 },
  sectionHeaderRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 20 },
  takenSectionTitle: { fontSize: 16, fontWeight: '900', color: '#1E293B' },
  countBadge: { backgroundColor: '#059669', paddingHorizontal: 10, paddingVertical: 3, borderRadius: 10 },
  countBadgeText: { color: '#fff', fontSize: 11, fontWeight: '900' },
  takenCard: { flexDirection: "row", alignItems: "center", backgroundColor: "#fff", padding: 18, borderRadius: 28, marginBottom: 15, borderWidth: 1, borderColor: '#F1F5F9', elevation: 4 },
  takenCheckIcon: { width: 40, height: 40, borderRadius: 14, backgroundColor: '#ECFDF5', justifyContent: 'center', alignItems: 'center' },
  takenMedName: { fontSize: 16, fontWeight: "800", color: "#1E293B" },
  takenMedTime: { fontSize: 13, color: "#94A3B8", marginTop: 4, fontWeight: '700' },

  medsList: { paddingHorizontal: 25, gap: 12, marginTop: 10 },
  medCard: { backgroundColor: '#fff', borderRadius: 24, padding: 18, borderWidth: 1, borderColor: '#F1F5F9', elevation: 4, shadowColor: '#000', shadowOpacity: 0.05 },
  medCardExpanded: { borderColor: '#10B981', elevation: 8, shadowColor: '#059669' },
  medHeaderMain: { flexDirection: "row", alignItems: "center" },
  medIconBox: { width: 44, height: 44, borderRadius: 15, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center' },
  medName: { fontSize: 16, fontWeight: '900', color: '#1E293B' },
  medCompactSub: { fontSize: 12, color: '#64748B', fontWeight: '600', marginTop: 2 },
  headerRightInfo: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  stockBadgeCompact: { minWidth: 50, height: 24, borderRadius: 8, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 8 },
  stockBadgeText: { fontSize: 10, fontWeight: '900' },
  
  expandedContent: { marginTop: 15 },
  divider: { height: 1, backgroundColor: '#F1F5F9', marginBottom: 15 },
  medBodyGrid: { flexDirection: 'row', justifyContent: 'space-between', backgroundColor: '#F8FAFC', padding: 12, borderRadius: 18, marginBottom: 15 },
  gridItem: { flex: 1, alignItems: 'center' },
  gridLabel: { fontSize: 8, fontWeight: '900', color: '#94A3B8', marginBottom: 4, letterSpacing: 0.5 },
  gridValue: { fontSize: 12, fontWeight: '800', color: '#1E293B' },
  
  scheduleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 15, paddingLeft: 5 },
  scheduleText: { fontSize: 13, color: '#1E293B', fontWeight: '800' },
  dayChip: { paddingHorizontal: 8, paddingVertical: 5, borderRadius: 8, backgroundColor: "#F8FAFC", borderWidth: 1, borderColor: '#F1F5F9' },
  dayChipActive: { backgroundColor: '#F0FDF4', borderColor: '#10B981' },
  dayChipTxt: { fontSize: 9, fontWeight: "800", color: "#94A3B8" },
  dayChipTxtActive: { color: "#059669" },
  
  noteBox: { flexDirection: 'row', gap: 10, marginTop: 15, backgroundColor: '#F8FAFC', padding: 12, borderRadius: 18, borderLeftWidth: 3, borderLeftColor: '#CBD5E1' },
  noteText: { fontSize: 12, color: "#475569", fontWeight: '600', fontStyle: 'italic', flex: 1 },
  
  cardActionsRow: { flexDirection: 'row', gap: 10, marginTop: 20 },
  actionBtnOutline: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 44, borderRadius: 12, backgroundColor: '#fff', borderWidth: 1, borderColor: '#E2E8F0' },
  actionBtnText: { fontSize: 13, fontWeight: '700', color: '#64748B' },
  
  alertBar: { flexDirection: "row", alignItems: "center", gap: 10, padding: 10, borderRadius: 12 },
  alertText: { fontSize: 11, fontWeight: '800' },

  modalOverlay: { flex: 1, backgroundColor: "rgba(6, 78, 59, 0.7)", justifyContent: "flex-end" },
  modalSheet: { backgroundColor: "#fff", borderTopLeftRadius: 50, borderTopRightRadius: 50, padding: 25, maxHeight: "94%", elevation: 25 },
  sheetHandle: { width: 50, height: 6, backgroundColor: '#E2E8F0', borderRadius: 3, alignSelf: "center", marginBottom: 25 },
  modalHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 30 },
  modalTitle: { fontSize: 22, fontWeight: '900', color: '#1E293B' },
  modalCloseBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#F8FAFC', justifyContent: 'center', alignItems: 'center' },
  inputGroup: { marginBottom: 25 },
  inputLabel: { fontSize: 15, fontWeight: "800", color: "#1E293B", marginBottom: 12, marginLeft: 5 },
  smallInputLabel: { fontSize: 12, fontWeight: "700", color: "#64748B", marginBottom: 8, marginLeft: 5 },
  textInput: { backgroundColor: "#F8FAFC", borderRadius: 20, paddingHorizontal: 20, paddingVertical: 16, fontSize: 16, color: "#1E293B", borderWidth: 1.5, borderColor: "#F1F5F9", fontWeight: '600' },
  formChip: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 16, backgroundColor: "#F8FAFC", borderWidth: 1.5, borderColor: "#F1F5F9" },
  formChipActive: { backgroundColor: '#F0FDF4', borderColor: '#059669' },
  formChipTxt: { fontSize: 14, fontWeight: "700", color: "#64748B" },
  formChipTxtActive: { color: "#059669", fontWeight: '900' },
  presetGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  presetChip: { flex: 1, minWidth: '45%', paddingVertical: 14, borderRadius: 18, backgroundColor: "#F8FAFC", borderWidth: 1.5, borderColor: "#F1F5F9", alignItems: 'center' },
  presetChipActive: { backgroundColor: '#F0FDF4', borderColor: '#059669' },
  presetChipTxt: { fontSize: 13, fontWeight: "700", color: "#64748B" },
  presetChipTxtActive: { color: "#059669", fontWeight: '900' },
  daySelectorRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 15 },
  daySelectCircle: { width: 40, height: 40, borderRadius: 20, backgroundColor: '#F8FAFC', justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: '#F1F5F9' },
  daySelectCircleActive: { backgroundColor: '#059669', borderColor: '#059669' },
  daySelectCircleTxt: { fontSize: 14, fontWeight: '800', color: '#64748B' },
  daySelectCircleTxtActive: { color: '#fff' },
  intervalRow: { flexDirection: 'row', gap: 15, marginTop: 15 },
  saveBtn: { borderRadius: 22, overflow: 'hidden', marginTop: 15, elevation: 8, shadowColor: '#059669', shadowOpacity: 0.3 },
  saveBtnGradient: { paddingVertical: 18, justifyContent: 'center', alignItems: 'center' },
  saveBtnTxt: { color: "#fff", fontSize: 16, fontWeight: "900" },
});
