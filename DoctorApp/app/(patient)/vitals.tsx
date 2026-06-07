import React, { useState, useEffect, useRef } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, TextInput, Switch, Alert, Dimensions, Animated, Platform
} from "react-native";
import { useRouter } from "expo-router";
import { useLanguage } from "../../context/LanguageContext";
import { useTheme } from "../../context/ThemeContext";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import {
  Activity, Droplets, Heart as HeartIcon, Thermometer, Wind, Beaker, Plus, Info, Timer, Sparkles, AlertCircle, Trash2, Edit
} from "lucide-react-native";
import { COLORS } from "../../constants/colors";
import { SosBar } from "../../components/SosBar";
import PatientBackgroundBubbles from "@/components/PatientBackgroundBubbles";
import { getMyPatientId } from "../../services/authService";
import {
  getPatientVitals, getLatestVital, addVitalReading,
  NORMAL_RANGES, checkVitalNormal, getVitalRangeText,
  updateVitalReading, deleteVitalReading,
  type VitalReading, type CreateVitalPayload,
} from "../../services/vitalService";
import { assessVitalReading } from "../../services/healthSafety";
import { addNotification } from "../../services/notificationService";
import { getAllergies, getChronicDiseases, type ChronicDisease } from "../../services/medicalRecordService";
import Toast from "react-native-toast-message";

const { width } = Dimensions.get("window");

const VITAL_TYPES = [
  { key: "Blood Pressure", unit: "mmHg", hasValue2: true, icon: Activity, color: "#0EA5E9", bg: "#F0F9FF" },
  { key: "Blood Sugar", unit: "mg/dL", hasValue2: false, icon: Droplets, color: "#F59E0B", bg: "#FFFBEB" },
  { key: "Heart Rate", unit: "bpm", hasValue2: false, icon: HeartIcon, color: "#EF4444", bg: "#FEF2F2" },
  { key: "Temperature", unit: "C", hasValue2: false, icon: Thermometer, color: "#0284C7", bg: "#E0F2FE" },
  { key: "SpO2", unit: "%", hasValue2: false, icon: Wind, color: "#10B981", bg: "#ECFDF5" },
  { key: "Respiratory Rate", unit: "breaths/min", hasValue2: false, icon: Beaker, color: "#0D9488", bg: "#F0FDFA" },
];

export default function VitalsScreen() {
  const router = useRouter();
  const { theme, isDark, colors } = useTheme();
  const { tr, isRTL } = useLanguage();
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
  const [editingId, setEditingId] = useState<number | null>(null);
  const [aiAdvice, setAiAdvice] = useState<any>(null);
  const [isAnalyzingAdvice, setIsAnalyzingAdvice] = useState(false);
  const [adviceLang, setAdviceLang] = useState<"en" | "ar">(isRTL ? "ar" : "en");

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
      // Trigger AI Advice for latest context if available
      if (allVitals.length > 0) {
        triggerAiAdvice(allVitals[0]);
      }
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load vitals" });
    } finally {
      setLoading(false);
    }
  };

  const currentTypeInfo = VITAL_TYPES.find((t) => t.key === selectedType)!;

  const validateAndSetNormal = (nextValue = value, nextValue2 = value2, nextType = selectedType) => {
    const typeInfo = VITAL_TYPES.find((t) => t.key === nextType) ?? currentTypeInfo;
    const v = Number(nextValue);
    const v2 = typeInfo.hasValue2 ? Number(nextValue2) : undefined;
    if (!Number.isNaN(v) && v > 0) {
      setIsNormal(assessVitalReading(nextType, v, v2).isNormal);
    } else {
      setIsNormal(false);
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

    const assessment = assessVitalReading(
      selectedType,
      numValue,
      currentTypeInfo.hasValue2 && value2 ? Number(value2) : undefined
    );
    const warningNote = assessment.isNormal ? "" : `[Health warning] ${assessment.message}`;
    const payload: CreateVitalPayload = {
      chronicDiseaseMonitorId: monitorId,
      readingType: selectedType,
      value: numValue,
      value2: currentTypeInfo.hasValue2 && value2 ? Number(value2) : undefined,
      unit: currentTypeInfo.unit,
      isNormal: assessment.isNormal,
      notes: [notes.trim(), warningNote].filter(Boolean).join("\n") || undefined,
      sugarReadingContext: selectedType === "Blood Sugar" ? "random" : undefined,
    };

    try {
      setSaving(true);
      if (editingId) {
        await updateVitalReading(editingId, payload);
        Toast.show({ type: "success", text1: "Vital updated successfully" });
      } else {
        await addVitalReading(patientId!, payload);
        if (assessment.isNormal) {
          Toast.show({ type: "success", text1: "Vital recorded successfully" });
        } else {
          Toast.show({ type: "error", text1: assessment.title, text2: "Saved with warning" });
          Alert.alert(assessment.title, assessment.message);
          await addNotification({
            id: `vital_${Date.now()}`,
            type: "update",
            icon: "alert-circle",
            title: assessment.title,
            message: assessment.message,
            timestamp: Date.now(),
          });
        }

        // Trigger AI Advice after save
        triggerAiAdvice(payload);
      }
      setValue("");
      setValue2("");
      setNotes("");
      setEditingId(null);
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

  const triggerAiAdvice = (reading: any) => {
    const val = reading.value;
    const val2 = reading.value2;
    const assessment = assessVitalReading(reading.readingType, val, val2);

    const typeAr: Record<string, string> = {
      "Blood Pressure": "ضغط الدم", "Blood Sugar": "سكر الدم",
      "Heart Rate": "معدل ضربات القلب", "Temperature": "درجة الحرارة",
      "SpO2": "الأكسجين", "Respiratory Rate": "معدل التنفس"
    };
    const nameAr = typeAr[reading.readingType] || reading.readingType;
    const readingStr = `${val}${val2 != null ? `/${val2}` : ""} ${reading.unit || ""}`;

    const advice_en = assessment.isNormal
      ? `Your ${reading.readingType} reading (${readingStr}) is within the normal range (${assessment.rangeText}). Keep up the good health!`
      : `${assessment.title}: ${reading.readingType} reading ${readingStr} is outside normal range (${assessment.rangeText}). ${assessment.message}`;

    const titleAr: Record<string, string> = {
      "Low blood pressure warning": "تحذير: انخفاض ضغط الدم",
      "High blood pressure warning": "تحذير: ارتفاع ضغط الدم",
      "Invalid reading": "قراءة غير صالحة",
    };
    const advice_ar = assessment.isNormal
      ? `قراءة ${nameAr} (${readingStr}) ضمن المعدل الطبيعي (${assessment.rangeText}). حافظ على صحتك!`
      : `${titleAr[assessment.title] || "تنبيه صحي"}: قراءة ${nameAr} ${readingStr} خارج المعدل الطبيعي (${assessment.rangeText}). يرجى مراجعة الطبيب.`;

    setAiAdvice({ advice_en, advice_ar });
  };

  const handleDelete = async (id: number) => {
    Alert.alert(
      "Delete Reading",
      "Are you sure you want to delete this record?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              await deleteVitalReading(id);
              Toast.show({ type: "success", text1: "Reading deleted" });
              await loadData();
            } catch (e: any) {
              Toast.show({ type: "error", text1: "Failed to delete" });
            }
          }
        }
      ]
    );
  };

  const handleEditInit = (v: VitalReading) => {
    setSelectedType(v.readingType);
    setValue(v.value.toString());
    setValue2(v.value2?.toString() || "");
    setNotes(v.notes || "");
    setIsNormal(v.isNormal);
    setEditingId(v.id);
    setShowAddForm(true);
    // Scroll to top for visibility if needed, but simple show form is okay
  };

  const scrollY = useRef(new Animated.Value(0)).current;

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#059669" />
      </View>
    );
  }

  const headerHeight = scrollY.interpolate({
    inputRange: [0, 150],
    outputRange: [220, 120],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 150],
    outputRange: [1, 0.9],
    extrapolate: 'clamp',
  });

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} translucent backgroundColor="transparent" />

      {/* Background Bubbles */}
      <PatientBackgroundBubbles isDark={isDark} scrollY={scrollY} />

      {/* ANIMATED LUXURY HEADER */}
      <Animated.View style={[styles.magicHeader, { height: headerHeight, opacity: headerOpacity }]}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={StyleSheet.absoluteFill}>
          <View style={styles.headerTop}>
            <TouchableOpacity style={styles.glassBtn} onPress={() => router.back()}>
              <Ionicons name="chevron-back" size={24} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Vitals Dashboard</Text>
            <TouchableOpacity style={styles.glassBtn} onPress={loadData}>
              <Ionicons name="refresh" size={20} color="#fff" />
            </TouchableOpacity>
          </View>

          <View style={styles.heroContent}>
            <View style={styles.heroTextRow}>
              <View>
                <Text style={styles.heroLabel}>Health Summary</Text>
                <Text style={styles.heroMain}>Biometric Data</Text>
              </View>
              <View style={styles.premiumTag}>
                <Sparkles size={12} color="#FDE047" />
                <Text style={styles.premiumText}>Live Analysis</Text>
              </View>
            </View>
          </View>

          {/* Decorative Blobs */}
          <View style={[styles.liquidBlob, { top: -40, right: -60, width: 220, height: 220, backgroundColor: '#10B981', opacity: 0.2 }]} />
          <View style={[styles.liquidBlob, { bottom: -20, left: -50, width: 180, height: 180, backgroundColor: '#34D399', opacity: 0.15 }]} />
        </LinearGradient>
      </Animated.View>

      {sosData && <View style={styles.sosContainer}><SosBar bloodType={sosData.bloodType} allergies={sosData.allergies} /></View>}

      <Animated.ScrollView
        showsVerticalScrollIndicator={false}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
        contentContainerStyle={{ paddingTop: 200, paddingBottom: 100 }}
      >
        <View style={[styles.contentOverlap, { backgroundColor: colors.background }]}>

          {/* AI ADVICE CARD */}
          {(aiAdvice || isAnalyzingAdvice) && (
            <View style={styles.aiAdviceWrapper}>
              <LinearGradient colors={isDark ? ["#1E293B", "#121B2E"] : ["#F5F3FF", "#EDE9FE"]} style={[styles.aiAdviceCard, { borderColor: colors.border }]}>
                <View style={styles.aiAdviceHeader}>
                  <View style={styles.aiSparkleBg}>
                    <Sparkles size={18} color="#7C3AED" />
                  </View>
                  <Text style={[styles.aiAdviceTitle, { color: colors.text }]}>{isRTL ? "نصيحة ذكية من AI" : "Smart AI Insights"}</Text>
                  <View style={styles.cardLangToggle}>
                    <TouchableOpacity onPress={() => setAdviceLang("en")} style={[styles.cardLangBtn, adviceLang === "en" && styles.cardLangBtnActive]}>
                      <Text style={[styles.cardLangText, adviceLang === "en" && styles.cardLangTextActive]}>EN</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => setAdviceLang("ar")} style={[styles.cardLangBtn, adviceLang === "ar" && styles.cardLangBtnActive]}>
                      <Text style={[styles.cardLangText, adviceLang === "ar" && styles.cardLangTextActive]}>عربي</Text>
                    </TouchableOpacity>
                  </View>
                  {isAnalyzingAdvice && <ActivityIndicator size="small" color="#7C3AED" />}
                </View>
                {aiAdvice && (
                  <Text style={[styles.aiAdviceText, { color: colors.textMuted }, adviceLang === 'ar' && { textAlign: 'right' }]}>
                    {adviceLang === 'ar' ? aiAdvice.advice_ar : aiAdvice.advice_en}
                  </Text>
                )}
              </LinearGradient>
            </View>
          )}

          {/* Quick Add Toggle */}
          <View style={styles.sectionHeaderRow}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Latest Health Metrics</Text>
            <TouchableOpacity style={styles.addBtnSmall} onPress={() => setShowAddForm(!showAddForm)}>
              <LinearGradient colors={["#059669", "#047857"]} style={styles.addBtnSmallGradient}>
                <Plus size={18} color="#fff" />
                <Text style={styles.addBtnSmallText}>{showAddForm ? tr("cancel") : "Add New"}</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {/* Add Form with Glass Style & Premium Feel */}
          {showAddForm && (
            <View style={[styles.formCard, { backgroundColor: colors.surface, borderColor: isDark ? '#1E293B' : '#E0F2FE' }]}>
              <View style={[styles.formIndicator, { backgroundColor: isDark ? '#334155' : '#E2E8F0' }]} />
              
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <Text style={[styles.formTitle, { color: colors.text }]}>{editingId ? (isRTL ? "تعديل القياس" : "Edit Reading") : (isRTL ? "تسجيل قياس جديد" : "New Vital Reading")}</Text>
                {editingId && (
                  <TouchableOpacity onPress={() => { setEditingId(null); setValue(""); setValue2(""); setNotes(""); setShowAddForm(false); }} style={styles.cancelEditBtn}>
                    <Ionicons name="close" size={20} color={colors.textMuted} />
                  </TouchableOpacity>
                )}
              </View>

              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.typeScroll} contentContainerStyle={{ paddingRight: 20, gap: 10 }}>
                {VITAL_TYPES.map((t) => (
                  <TouchableOpacity
                    key={t.key}
                    style={[
                      styles.typeChip, 
                      { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: isDark ? "#1E293B" : "#F1F5F9" }, 
                      selectedType === t.key && styles.typeChipActive
                    ]}
                    onPress={() => { 
                      import('expo-haptics').then(Haptics => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light));
                      setSelectedType(t.key); 
                      setValue(""); 
                      setValue2(""); 
                      setIsNormal(true);
                    }}
                  >
                    <View style={[styles.typeChipIconWrapper, selectedType === t.key && { backgroundColor: 'rgba(255,255,255,0.2)' }]}>
                      <t.icon size={16} color={selectedType === t.key ? "#fff" : t.color} />
                    </View>
                    <Text style={[styles.typeChipTxt, selectedType === t.key && styles.typeChipTxtActive]}>{isRTL ? t.key : t.key}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>

              <View style={styles.inputRow}>
                <View style={styles.inputWrap}>
                  <Text style={styles.inputLabel}>{isRTL ? "القيمة" : "Value"} {currentTypeInfo.hasValue2 && (isRTL ? "(الانقباضي)" : "(Systolic)")}</Text>
                  <View style={[styles.inputPremiumContainer, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: isDark ? '#1E293B' : '#F1F5F9' }]}>
                    <TextInput
                      style={[styles.premiumInputInner, { color: colors.text }]}
                      keyboardType="numeric"
                      value={value}
                      onChangeText={(t) => { setValue(t); validateAndSetNormal(t, value2); }}
                      placeholder="0"
                      placeholderTextColor="#CBD5E1"
                    />
                    <Text style={styles.inputUnitTag}>{currentTypeInfo.unit}</Text>
                  </View>
                </View>
                {currentTypeInfo.hasValue2 && (
                  <View style={styles.inputWrap}>
                    <Text style={styles.inputLabel}>{isRTL ? "الانبساطي" : "Diastolic"}</Text>
                    <View style={[styles.inputPremiumContainer, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: isDark ? '#1E293B' : '#F1F5F9' }]}>
                      <TextInput
                        style={[styles.premiumInputInner, { color: colors.text }]}
                        keyboardType="numeric"
                        value={value2}
                        onChangeText={(t) => { setValue2(t); validateAndSetNormal(value, t); }}
                        placeholder="0"
                        placeholderTextColor="#CBD5E1"
                      />
                      <Text style={styles.inputUnitTag}>{currentTypeInfo.unit}</Text>
                    </View>
                  </View>
                )}
              </View>

              {!isNormal && value !== "" && (
                <View style={styles.abnormalBanner}>
                  <AlertCircle size={18} color="#B91C1C" />
                  <Text style={styles.abnormalText}>
                    {assessVitalReading(selectedType, Number(value), currentTypeInfo.hasValue2 ? Number(value2) : undefined).message}
                  </Text>
                </View>
              )}

              <View style={styles.inputWrap}>
                <Text style={styles.inputLabel}>{isRTL ? "ملاحظات (اختياري)" : "Notes (optional)"}</Text>
                <TextInput 
                  style={[styles.premiumInputFull, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", color: colors.text, borderColor: isDark ? '#1E293B' : '#F1F5F9' }]} 
                  value={notes} 
                  onChangeText={setNotes} 
                  placeholder={isRTL ? "مثال: بعد الأكل..." : "e.g., after meal..."} 
                  placeholderTextColor="#CBD5E1" 
                  multiline
                />
              </View>

              <TouchableOpacity 
                style={[styles.saveBtn, saving && { opacity: 0.7 }]} 
                onPress={() => {
                  import('expo-haptics').then(Haptics => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium));
                  handleSave();
                }} 
                disabled={saving} 
                activeOpacity={0.8}
              >
                <LinearGradient colors={["#059669", "#047857"]} style={styles.saveBtnGradient}>
                  {saving ? (
                    <ActivityIndicator color="#fff" size="small" />
                  ) : (
                    <>
                      <Sparkles size={18} color="#fff" style={{ marginRight: 8 }} />
                      <Text style={styles.saveBtnTxt}>{editingId ? (isRTL ? "تحديث القياس" : "Update Record") : (isRTL ? "حفظ القياس" : "Save Record")}</Text>
                    </>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            </View>
          )}

          <View style={styles.liveIndicatorRow}>
            <View style={styles.liveIndicator}>
              <View style={styles.liveDot} />
              <Text style={styles.liveText}>Live Sync Active</Text>
            </View>
          </View>

          <View style={styles.latestGrid}>
            {VITAL_TYPES.map((t) => {
              const reading = latestVitals[t.key];
              const isAbnormal = reading && !reading.isNormal;
              return (
                <View key={t.key} style={styles.latestCardWrap}>
                  <View style={[styles.latestCard, { backgroundColor: colors.surface, borderColor: colors.border }, isAbnormal && styles.latestCardAbnormal]}>
                    <View style={styles.cardHeaderRow}>
                      <View style={[styles.iconCircle, { backgroundColor: t.bg }]}>
                        <t.icon size={16} color={t.color} />
                      </View>
                      {isAbnormal && <View style={styles.alertDot} />}
                    </View>

                    <Text style={styles.latestType}>{t.key}</Text>

                    {reading ? (
                      <View style={styles.readingContent}>
                        <View style={styles.valueRow}>
                          <Text style={[styles.latestValue, { color: colors.text }, isAbnormal && styles.latestValueAbnormal]}>
                            {reading.value}{reading.value2 != null ? `/${reading.value2}` : ""}
                          </Text>
                          <Text style={styles.latestUnit}>{reading.unit}</Text>
                        </View>
                        <View style={styles.readingMeta}>
                          <Timer size={10} color="#94A3B8" />
                          <Text style={styles.latestDate}>{new Date(reading.recordedAt).toLocaleDateString()}</Text>
                        </View>
                      </View>
                    ) : (
                      <TouchableOpacity style={styles.addPlaceholder} onPress={() => { setSelectedType(t.key); setShowAddForm(true); }}>
                        <Text style={styles.addPlaceholderText}>Record</Text>
                        <Plus size={10} color="#94A3B8" />
                      </TouchableOpacity>
            )}
                  </View>
                </View>
              );
            })}
          </View>

          {/* History Section */}
          <View style={styles.sectionHeaderRow}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Medical Timeline</Text>
            <View style={styles.historyBadgeCount}><Text style={styles.historyBadgeCountText}>{vitals.length}</Text></View>
          </View>

          {vitals.length === 0 ? (
            <View style={styles.emptyCard}>
              <Activity size={48} color="#E2E8F0" />
              <Text style={styles.emptyTitle}>No History Yet</Text>
              <Text style={styles.emptyDesc}>Your biometric timeline will appear here once you log your first measurement.</Text>
            </View>
          ) : (
            <View style={styles.timeline}>
              {vitals.map((v, i) => {
                const isAbnormal = !v.isNormal;
                const typeInfo = VITAL_TYPES.find(t => t.key === v.readingType);
                return (
                  <View key={i} style={styles.timelineItem}>
                    <View style={styles.timelineLeft}>
                      <View style={[styles.timelineDot, { backgroundColor: typeInfo?.color || '#CBD5E1' }]} />
                      {i !== vitals.length - 1 && <View style={styles.timelineLine} />}
                    </View>
                    <View style={[styles.historyCard, { backgroundColor: colors.surface, borderColor: colors.border }, isAbnormal && styles.historyCardAbnormal]}>
                      <View style={styles.historyTopRow}>
                        <Text style={[styles.historyType, { color: colors.text }]}>{v.readingType}</Text>
                        <View style={[styles.statusBadge, isAbnormal ? styles.statusBadgeError : styles.statusBadgeSuccess]}>
                          <Text style={[styles.statusText, isAbnormal ? styles.statusTextError : styles.statusTextSuccess]}>
                            {isAbnormal ? "High Alert" : "Normal"}
                          </Text>
                        </View>
                        <View style={{ flexDirection: 'row', gap: 12 }}>
                          <TouchableOpacity onPress={() => handleEditInit(v)}>
                            <Edit size={16} color="#64748B" />
                          </TouchableOpacity>
                          <TouchableOpacity onPress={() => handleDelete(v.id)}>
                            <Trash2 size={16} color="#EF4444" />
                          </TouchableOpacity>
                        </View>
                      </View>
                      <View style={styles.historyValueRow}>
                        <Text style={[styles.historyValue, { color: colors.text }, isAbnormal && styles.historyValueAbnormal]}>
                          {v.value}{v.value2 != null ? ` / ${v.value2}` : ""}
                        </Text>
                        <Text style={styles.historyUnit}>{v.unit}</Text>
                      </View>
                      {v.notes && (
                        <View style={styles.historyNotesBox}>
                          <Info size={12} color="#64748B" />
                          <Text style={styles.historyNotes}>{v.notes}</Text>
                        </View>
                      )}
                      <Text style={[styles.historyDate, { color: colors.textMuted }]}>{new Date(v.recordedAt).toLocaleString()}</Text>
                    </View>
                  </View>
                );
              })}
            </View>
          )}
        </View>
      </Animated.ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  bgBubble: { position: 'absolute', borderRadius: 300, filter: 'blur(40px)' },
  bubbleTopLeft: { width: 350, height: 350, top: -100, left: -100 },
  bubbleBottomRight: { width: 400, height: 400, bottom: -150, right: -150 },
  bubbleCenter: { width: 250, height: 250, top: '40%', left: '20%' },
  magicHeader: { position: 'absolute', top: 0, left: 0, right: 0, zIndex: 100, overflow: 'hidden' },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingTop: 60, zIndex: 10 },
  glassBtn: { width: 44, height: 44, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  headerTitle: { fontSize: 16, fontWeight: '700', color: '#fff', letterSpacing: 0.5 },
  heroContent: { paddingHorizontal: 25, marginTop: 30, zIndex: 10 },
  heroTextRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' },
  heroLabel: { fontSize: 12, color: "rgba(255,255,255,0.7)", fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1 },
  heroMain: { fontSize: 28, fontWeight: "900", color: "#fff", marginTop: 4, letterSpacing: -0.5 },
  premiumTag: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  premiumText: { fontSize: 10, fontWeight: '800', color: '#fff', textTransform: 'uppercase' },
  liquidBlob: { position: 'absolute', borderRadius: 150 },

  contentOverlap: { backgroundColor: '#F8FAFC', borderTopLeftRadius: 40, borderTopRightRadius: 40, minHeight: 600, paddingTop: 30 },
  sectionHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 25, marginBottom: 20, marginTop: 10 },
  sectionTitle: { fontSize: 18, fontWeight: '800', color: '#1E293B', letterSpacing: -0.5 },
  addBtnSmall: { borderRadius: 15, overflow: 'hidden', elevation: 5 },
  addBtnSmallGradient: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 16, paddingVertical: 10 },
  addBtnSmallText: { color: '#fff', fontSize: 12, fontWeight: '800' },

  formCard: { backgroundColor: '#fff', borderRadius: 32, padding: 25, marginHorizontal: 20, marginBottom: 30, elevation: 12, shadowColor: '#0EA5E9', shadowOpacity: 0.1, shadowRadius: 20, borderWidth: 2, borderColor: '#BAE6FD' },
  formIndicator: { width: 40, height: 5, backgroundColor: '#F1F5F9', borderRadius: 5, alignSelf: 'center', marginBottom: 20 },
  formTitle: { fontSize: 18, fontWeight: '900', color: '#1E293B', marginBottom: 0 },
  cancelEditBtn: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#F1F5F9', justifyContent: 'center', alignItems: 'center' },
  typeScroll: { marginBottom: 25 },
  typeChip: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 20, backgroundColor: '#F8FAFC', marginRight: 12, borderWidth: 1, borderColor: '#F1F5F9' },
  typeChipActive: { backgroundColor: '#0EA5E9', borderColor: '#0EA5E9' },
  typeChipIconWrapper: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  typeChipTxt: { fontSize: 13, fontWeight: '700', color: '#64748B', marginRight: 6 },
  typeChipTxtActive: { color: "#fff" },
  inputRow: { flexDirection: "row", gap: 15 },
  inputWrap: { flex: 1, marginBottom: 20 },
  inputLabel: { fontSize: 13, fontWeight: "800", color: "#64748B", marginBottom: 10, marginLeft: 4, textTransform: 'uppercase', letterSpacing: 0.5 },
  
  inputPremiumContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F8FAFC', borderRadius: 18, paddingHorizontal: 16, height: 60, borderWidth: 1.5, borderColor: '#F1F5F9' },
  premiumInputInner: { flex: 1, fontSize: 18, fontWeight: '800', color: '#1E293B' },
  inputUnitTag: { fontSize: 13, fontWeight: '800', color: '#94A3B8', marginLeft: 8 },
  premiumInputFull: { backgroundColor: '#F8FAFC', borderRadius: 18, paddingHorizontal: 18, height: 100, fontSize: 15, color: '#1E293B', borderWidth: 1.5, borderColor: '#F1F5F9', fontWeight: '600', textAlignVertical: 'top', paddingTop: 16 },
  
  abnormalBanner: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: 15, padding: 12, marginBottom: 20, borderWidth: 1, borderColor: "#FCD34D" },
  abnormalText: { color: "#D97706", fontSize: 12, fontWeight: "800" },
  saveBtn: { borderRadius: 20, overflow: 'hidden', elevation: 8, shadowColor: '#059669', shadowOpacity: 0.3, shadowRadius: 10 },
  saveBtnGradient: { flexDirection: 'row', height: 60, justifyContent: 'center', alignItems: 'center' },
  saveBtnTxt: { color: "#fff", fontSize: 16, fontWeight: "800", letterSpacing: 0.5 },

  liveIndicatorRow: { paddingHorizontal: 25, marginBottom: 15 },
  liveIndicator: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: 'rgba(16, 185, 129, 0.1)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 15, alignSelf: 'flex-start' },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#10B981' },
  liveText: { fontSize: 10, fontWeight: '800', color: '#10B981', textTransform: 'uppercase' },

  latestGrid: { flexDirection: "row", flexWrap: "wrap", paddingHorizontal: 20, gap: 12, marginBottom: 30 },
  latestCardWrap: { width: '48%', marginBottom: 5 },
  latestCard: { borderRadius: 28, padding: 18, borderWidth: 2, elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15 },
  latestCardAbnormal: { borderColor: "#FCD34D", backgroundColor: "rgba(245, 158, 11, 0.05)" },
  cardHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  iconCircle: { width: 36, height: 36, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  alertDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#F59E0B' },
  latestType: { fontSize: 10, fontWeight: '800', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 },
  readingContent: { gap: 6 },
  valueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 4 },
  latestValue: { fontSize: 20, fontWeight: '900' },
  latestValueAbnormal: { color: "#D97706" },
  latestUnit: { fontSize: 11, color: '#64748B', fontWeight: '700' },
  readingMeta: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4 },
  latestDate: { fontSize: 10, color: "#94A3B8", fontWeight: '600' },
  addPlaceholder: { height: 45, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'transparent', paddingHorizontal: 12, borderRadius: 12, borderStyle: 'dashed', borderWidth: 1, borderColor: '#CBD5E1' },
  addPlaceholderText: { fontSize: 12, color: '#94A3B8', fontWeight: '700' },

  historyBadgeCount: { backgroundColor: '#059669', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 10 },
  historyBadgeCountText: { color: '#fff', fontSize: 11, fontWeight: '800' },
  emptyCard: { borderRadius: 35, padding: 40, alignItems: 'center', marginHorizontal: 20, borderWidth: 2, borderColor: '#BAE6FD' },
  emptyTitle: { fontSize: 18, fontWeight: "900", color: "#1E293B", marginTop: 20 },
  emptyDesc: { fontSize: 14, color: "#94A3B8", marginTop: 10, textAlign: "center", lineHeight: 22 },

  timeline: { paddingHorizontal: 20, paddingBottom: 40 },
  timelineItem: { flexDirection: 'row', gap: 15 },
  timelineLeft: { alignItems: 'center', width: 20 },
  timelineDot: { width: 12, height: 12, borderRadius: 6, zIndex: 10, marginTop: 25 },
  timelineLine: { flex: 1, width: 2, backgroundColor: '#E2E8F0', marginTop: -10 },
  historyCard: { flex: 1, borderRadius: 24, padding: 18, marginBottom: 20, borderWidth: 2, elevation: 5, shadowColor: '#000', shadowOpacity: 0.04 },
  historyCardAbnormal: { borderColor: "#FCD34D" },
  historyTopRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  historyType: { fontSize: 15, fontWeight: '800' },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  statusBadgeSuccess: { backgroundColor: "rgba(16, 185, 129, 0.1)" },
  statusBadgeError: { backgroundColor: "rgba(245, 158, 11, 0.1)" },
  statusText: { fontSize: 10, fontWeight: "800", textTransform: 'uppercase' },
  statusTextSuccess: { color: "#059669" },
  statusTextError: { color: "#D97706" },
  historyValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6 },
  historyValue: { fontSize: 20, fontWeight: '900' },
  historyValueAbnormal: { color: "#D97706" },
  historyUnit: { fontSize: 13, color: '#64748B', fontWeight: '700' },
  historyNotesBox: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#F8FAFC', padding: 10, borderRadius: 12, marginTop: 12 },
  historyNotes: { fontSize: 12, color: "#64748B", fontWeight: '500' },
  historyDate: { fontSize: 11, color: '#94A3B8', marginTop: 12, fontWeight: '600' },
  sosContainer: { marginTop: -10, marginBottom: 10 },
  aiAdviceWrapper: { paddingHorizontal: 25, marginBottom: 20 },
  aiAdviceCard: { borderRadius: 24, padding: 18, borderWidth: 2, borderColor: '#BAE6FD', elevation: 4, shadowOpacity: 0.05, backgroundColor: '#fff' },
  aiAdviceHeader: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 12 },
  aiSparkleBg: { width: 36, height: 36, borderRadius: 12, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', elevation: 2 },
  aiAdviceTitle: { fontSize: 15, fontWeight: '900', color: '#1E293B', flex: 1, letterSpacing: -0.3 },
  aiAdviceText: { fontSize: 13, color: '#475569', lineHeight: 20, fontWeight: '600' },
  cardLangToggle: { flexDirection: 'row', backgroundColor: '#F1F5F9', borderRadius: 10, padding: 2, marginRight: 10 },
  cardLangBtn: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
  cardLangBtnActive: { backgroundColor: '#fff', elevation: 2 },
  cardLangText: { fontSize: 10, fontWeight: '800', color: '#64748B' },
  cardLangTextActive: { color: '#0EA5E9' },
});
