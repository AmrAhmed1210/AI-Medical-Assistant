import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar, ActivityIndicator, Alert, Image, Dimensions, Modal
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import {
  Heart, Droplets, Calendar, Pill, Search,
  ChevronRight, Star, Bell, LayoutGrid, Stethoscope,
  ArrowRight, HeartPulse, Thermometer, Activity, User, Sparkles, Clock, CheckCircle2
} from "lucide-react-native";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter, useFocusEffect } from "expo-router";
import { BASE_URL } from "../../constants/api";
import { Ionicons } from "@expo/vector-icons";
import { useLanguage } from "../../context/LanguageContext";
import { useTheme } from "../../context/ThemeContext";
import { useEffect, useState, useCallback, useRef } from "react";
import { Animated as RNAnimated, Platform } from "react-native";
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeInDown,
  FadeInRight,
  useAnimatedStyle,
  withSpring,
  useSharedValue,
  withRepeat,
  withSequence,
  withDelay
} from 'react-native-reanimated';
import { getAllDoctors, getDoctorById, getReviewsByDoctor, Doctor } from "../../services/doctorService";
import { getMyProfile, Profile } from "../../services/profileService";
import { getMyAppointments, Appointment } from "../../services/appointmentService";
import { getMedicationSchedule, type MedicationScheduleItem } from "../../services/medicationService";
import { getLatestVital, getPatientVitals, type VitalReading } from "../../services/vitalService";
import { getMyPatientId } from "../../services/authService";
import { NotificationBell } from "../../components/NotificationBell";
import DoctorCard from "../../components/DoctorCard";
import { addNotification } from "../../services/notificationService";
import { startSignalRConnection, onDoctorUpdated, onScheduleReady, onScheduleUpdated, onNewConsultation, onNewMedication } from "../../services/signalr";
import { scheduleAppointmentReminders } from "../../services/appointmentReminders";
import { analyzePatientHistory } from "../../services/aiService";
import { updateAiDiagnosis, getVitals, getSurgeries, getMedications, getAllergies, getChronicDiseases, getPatientDocuments } from "../../services/medicalRecordService";

const { width } = Dimensions.get("window");

const DEFAULT_AVATAR = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png";

function resolvePhotoUrl(url?: string | null): string {
  if (!url || !url.trim()) return DEFAULT_AVATAR;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  const separator = url.startsWith('/') ? '' : '/';
  return `${BASE_URL}${separator}${url}`;
}

export default function HomeScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
  const { theme, isDark, colors } = useTheme();
  const [userName, setUserName] = useState("");
  const [popularDocs, setPopularDocs] = useState<Doctor[]>([]);
  const [nextBooking, setNextBooking] = useState<Appointment | null>(null);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [nextDose, setNextDose] = useState<MedicationScheduleItem | null>(null);
  const [lastBP, setLastBP] = useState<VitalReading | null>(null);
  const [lastSugar, setLastSugar] = useState<VitalReading | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [showVitalReminder, setShowVitalReminder] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reportLang, setReportLang] = useState<'en' | 'ar'>(isRTL ? 'ar' : 'en');
  const [showAiModal, setShowAiModal] = useState(false);
  const scrollY = useRef(new RNAnimated.Value(0)).current;

  // Animation values
  const pulseScale = useSharedValue(1);

  useEffect(() => {
    pulseScale.value = withRepeat(
      withSequence(
        withSpring(1.1, { damping: 2 }),
        withSpring(1, { damping: 2 })
      ),
      -1,
      true
    );
  }, []);

  const triggerHaptic = (type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type);
  };

  const CATEGORIES = [
    { icon: Heart, label: tr("spec_cardiology"), specialty: "Cardiology" },
    { icon: Search, label: tr("spec_eye"), specialty: "Eye" },
    { icon: Activity, label: tr("spec_ortho"), specialty: "Orthopedics" },
    { icon: Stethoscope, label: tr("spec_neurology"), specialty: "Neurology" },
    { icon: Thermometer, label: tr("spec_dermatology"), specialty: "Dermatology" },
  ];

  useEffect(() => {
    fetchPopularDoctors();
    fetchNextBooking();
    fetchProfile();
    fetchHealthData();
    checkDailyReminder();

    // Real-time updates
    const unsubMed = onNewMedication(() => {
      fetchHealthData();
    });

    const unsubDoctor = onDoctorUpdated(() => {
      fetchPopularDoctors();
    });

    return () => {
      unsubMed();
      unsubDoctor();
    };
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchHealthData();
      fetchNextBooking();
      fetchPopularDoctors(); // Added to refresh doctor cards on focus
      checkDailyReminder();
    }, [])
  );

  const fetchProfile = async () => {
    try {
      const data = await getMyProfile();
      setProfile(data);
      if (data.name) setUserName(data.name);
    } catch { }
  };

  const fetchHealthData = async () => {
    try {
      setLoadingHealth(true);
      const pid = await getMyPatientId();
      if (!pid) { setLoadingHealth(false); return; }
      const [schedule, bp, sugar] = await Promise.all([
        getMedicationSchedule(pid).catch(() => []),
        getLatestVital(pid, "Blood Pressure").catch(() => null),
        getLatestVital(pid, "Blood Sugar").catch(() => null),
      ]);
      const pending = schedule
        .filter((s) => s.status?.toLowerCase() === "pending")
        .sort((a, b) => new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime());
      setNextDose(pending[0] || null);
      setLastBP(bp);
      setLastSugar(sugar);
    } finally {
      setLoadingHealth(false);
    }
  };

  const checkDailyReminder = async () => {
    try {
      const pid = await getMyPatientId();
      if (!pid) return;

      const vitals = await getPatientVitals(pid).catch(() => []);
      const today = new Date().toISOString().split('T')[0];

      const recordedToday = vitals.some(v => v.recordedAt.split('T')[0] === today);

      if (!recordedToday) {
        setShowVitalReminder(true);
        // Only send notification if not already notified today to avoid spamming
        const lastNotified = await AsyncStorage.getItem("last_vital_notification_date");
        if (lastNotified !== today) {
          await addNotification({
            id: `reminder_${Date.now()}`,
            type: 'message',
            icon: '📊',
            title: 'Daily Vital Check-in',
            message: 'You haven\'t recorded your health measurements for today yet.',
            timestamp: Date.now()
          });
          await AsyncStorage.setItem("last_vital_notification_date", today);
        }
      } else {
        setShowVitalReminder(false);
      }
    } catch (e) {
      console.log("Reminder check failed", e);
    }
  };

  const fetchPopularDoctors = async () => {
    try {
      setLoadingDocs(true);
      const data = await getAllDoctors();
      setPopularDocs(data.slice(0, 10));
    } finally {
      setLoadingDocs(false);
    }
  };

  const fetchNextBooking = async () => {
    try {
      const appts = await getMyAppointments();
      const now = new Date();
      console.log("Total appointments from API:", appts?.length);

      // Filter future active bookings robustly
      const futureAppts = (appts || [])
        .filter(a => {
          const status = a.status?.toLowerCase() || "";
          const isRelevant = status === "pending" || status === "confirmed";

          const apptDate = new Date(a.date);
          let [h, m] = (a.time || "0:0").split(':').map(val => parseInt(val, 10));
          if (a.time?.toLowerCase().includes("pm") && h < 12) h += 12;
          if (a.time?.toLowerCase().includes("am") && h === 12) h = 0;

          if (!isNaN(apptDate.getTime())) {
            apptDate.setHours(h, isNaN(m) ? 0 : m, 0, 0);
            const isFuture = apptDate > now;
            console.log(`Checking Appt: Dr.${a.doctorName}, Date:${a.date}, Time:${a.time}, Status:${status}, isFuture:${isFuture}`);
            return isRelevant && isFuture;
          }
          return false;
        })
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

      const active = futureAppts[0] || null;
      console.log("FINAL Next Appointment Detected:", active ? active.doctorName : "NONE");
      setNextBooking(active);

      if (active && typeof scheduleAppointmentReminders === 'function') {
        scheduleAppointmentReminders(futureAppts.slice(0, 3)).catch((err: any) => console.error(err));
      }
    } catch (e) {
      console.log("Fetch next booking failed", e);
      setNextBooking(null);
    }
  };

  const runAiAnalysis = async () => {
    try {
      setIsAnalyzing(true);
      const pid = await getMyPatientId();
      if (!pid) return;

      const [vitals, surgeries, meds, allergies, chronic, docs] = await Promise.all([
        getVitals(pid),
        getSurgeries(pid),
        getMedications(pid),
        getAllergies(pid),
        getChronicDiseases(pid),
        getPatientDocuments(pid).catch(() => []),
      ]);

      console.log("DEBUG: Patient docs fetched in Home:", JSON.stringify(docs));
      const analysis = await analyzePatientHistory({
        vitals: vitals.map(v => ({ type: v.readingType, value: v.value, recordedAt: v.recordedAt })),
        surgeries: surgeries.map(s => s.surgeryName),
        medications: meds.map(m => m.medicationName),
        allergies: allergies.map(a => a.allergenName),
        chronic_diseases: chronic.map(c => ({ diseaseName: c.diseaseName })),
        documents_analysis: docs.map(d => ({ 
          title: d.title ?? (d as any).Title ?? "", 
          ai_analysis: d.description ?? (d as any).Description ?? "" 
        }))
      });

      await updateAiDiagnosis(pid, JSON.stringify(analysis));
      await fetchProfile(); // Refresh profile to get the new diagnosis

      Alert.alert(isRTL ? "تم التحديث" : "Updated", isRTL ? "تم تحليل حالتك الصحية بنجاح" : "Your health analysis has been updated.");
    } catch (error) {
      console.error("AI Analysis error:", error);
      Alert.alert("Analysis Failed", "Could not complete AI analysis at this time.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good Morning";
    if (hour < 18) return "Good Afternoon";
    return "Good Evening";
  };

  const headerTranslateY = scrollY.interpolate({
    inputRange: [0, 150],
    outputRange: [0, -40],
    extrapolate: 'clamp',
  });

  const searchScale = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [1, 0.9],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [1, 0],
    extrapolate: 'clamp',
  });

  const blob1Move = scrollY.interpolate({
    inputRange: [0, 200],
    outputRange: [0, -30],
    extrapolate: 'clamp',
  });

  return (
    <View style={[styles.main, { backgroundColor: colors.background }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} />

      {/* Background Bubbles */}
      <View style={[styles.bgBubble, styles.bubbleTopLeft, { backgroundColor: isDark ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.08)' }]} />
      <View style={[styles.bgBubble, styles.bubbleBottomRight, { backgroundColor: isDark ? 'rgba(14, 165, 233, 0.15)' : 'rgba(14, 165, 233, 0.08)' }]} />
      <View style={[styles.bgBubble, styles.bubbleCenter, { backgroundColor: isDark ? 'rgba(139, 92, 246, 0.1)' : 'rgba(139, 92, 246, 0.05)' }]} />

      <RNAnimated.ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
        onScroll={RNAnimated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
        decelerationRate="fast"
        bounces={true}
      >
        <RNAnimated.View style={[styles.magicHeaderContainer, { transform: [{ translateY: headerTranslateY }] }]}>
          <LinearGradient colors={["#064E3B", "#059669"]} style={styles.magicHeader}>
            <RNAnimated.View style={[styles.headerTop, { opacity: headerOpacity }]}>
              <View>
                <Text style={styles.greetText}>{getGreeting()}</Text>
                <View style={styles.nameRow}>
                  <Text style={styles.userName}>{profile?.name?.split(" ")[0] || "Patient"}</Text>
                  <Sparkles size={16} color="#FDE047" style={{ marginLeft: 4 }} />
                </View>
              </View>
              <View style={styles.headerRight}>
                <TouchableOpacity style={styles.headerIconBtn} onPress={() => router.push("/(patient)/doctors")}>
                  <Search size={22} color="#fff" />
                </TouchableOpacity>
                <NotificationBell light />
                <TouchableOpacity style={styles.avatarWrap} onPress={() => router.push("/(patient)/profile")}>
                  <Image
                    source={{ uri: resolvePhotoUrl(profile?.photoUrl) }}
                    style={styles.avatarImg}
                    defaultSource={{ uri: DEFAULT_AVATAR }}
                  />
                </TouchableOpacity>
              </View>
            </RNAnimated.View>

            <RNAnimated.View style={{ transform: [{ scale: searchScale }] }}>
              <TouchableOpacity
                style={styles.findDoctorBtn}
                activeOpacity={0.8}
                onPress={() => router.push("/(patient)/doctors")}
              >
                <View style={styles.findDoctorContent}>
                  <Stethoscope size={20} color="#059669" />
                  <Text style={styles.findDoctorText}>Find a Specialist Doctor</Text>
                </View>
                <View style={styles.findDoctorArrow}>
                  <ArrowRight size={18} color="#059669" />
                </View>
              </TouchableOpacity>
            </RNAnimated.View>

            <RNAnimated.View style={[styles.headerDecor1, { transform: [{ translateY: blob1Move }] }]} />
            <RNAnimated.View style={[styles.headerDecor2, { transform: [{ translateY: RNAnimated.multiply(blob1Move, 1.5) }] }]} />
          </LinearGradient>
        </RNAnimated.View>

        {/* FLOATING GLASS TIP CARD */}
        <View style={styles.tipWrapper}>
          <LinearGradient
            colors={isDark ? ["rgba(30, 41, 59, 0.95)", "rgba(15, 23, 42, 0.9)"] : ["rgba(255, 255, 255, 0.95)", "rgba(240, 253, 244, 0.9)"]}
            style={[styles.magicTipCard, { borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(5, 150, 105, 0.12)' }]}
          >
            <View style={[styles.tipIconBox, { backgroundColor: isDark ? 'rgba(16, 185, 129, 0.2)' : '#ECFDF5' }]}>
              <View style={[styles.innerIconBox, { backgroundColor: isDark ? '#059669' : '#fff' }]}>
                <Ionicons name="bulb" size={22} color={isDark ? "#fff" : "#059669"} />
              </View>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[styles.tipTitle, { color: isDark ? '#6EE7B7' : '#064E3B' }]}>Daily Wellness</Text>
              <Text style={[styles.tipDesc, { color: isDark ? '#34D399' : '#047857' }]}>"Small steps lead to great health. Stay active and hydrated."</Text>
            </View>
            <TouchableOpacity style={styles.tipArrow}>
              <ChevronRight size={18} color="#059669" />
            </TouchableOpacity>
          </LinearGradient>
        </View>

        {/* AI HEALTH INSIGHTS CARD - REFINED BILINGUAL */}
        <View style={styles.aiInsightContainer}>
          <View style={[styles.aiWhiteCardRefined, { backgroundColor: colors.surface, borderColor: isDark ? '#0284C7' : '#BAE6FD' }]}>
            <View style={styles.aiHeaderRefined}>
              <View style={[styles.aiIconBoxRefined, { backgroundColor: isDark ? 'rgba(14, 165, 233, 0.2)' : '#F0F9FF' }]}>
                <Sparkles size={20} color="#0EA5E9" />
              </View>
              <Text style={[styles.aiTitleRefined, { color: colors.text }]}>{isRTL ? "تحليل الذكاء الاصطناعي" : "AI Health Insights"}</Text>

              <View style={styles.miniToggleBox}>
                <TouchableOpacity onPress={() => setReportLang('en')} style={[styles.miniToggleBtn, reportLang === 'en' && styles.miniToggleBtnActive]}>
                  <Text style={[styles.miniToggleText, reportLang === 'en' && styles.miniToggleTextActive]}>EN</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setReportLang('ar')} style={[styles.miniToggleBtn, reportLang === 'ar' && styles.miniToggleBtnActive]}>
                  <Text style={[styles.miniToggleText, reportLang === 'ar' && styles.miniToggleTextActive]}>AR</Text>
                </TouchableOpacity>
              </View>

              {isAnalyzing ? (
                <ActivityIndicator size="small" color="#0EA5E9" style={{ marginLeft: 10 }} />
              ) : (
                <TouchableOpacity onPress={runAiAnalysis} style={{ marginLeft: 10 }}>
                  <Clock size={18} color="#0EA5E9" />
                </TouchableOpacity>
              )}
            </View>

            {profile?.aiDiagnosisSummary ? (
              <View style={styles.aiContentRefined}>
                <Text style={[styles.aiTextRefined, { color: colors.textMuted }, reportLang === 'ar' && { textAlign: 'right' }]} numberOfLines={3}>
                  {(() => {
                    try {
                      const parsed = JSON.parse(profile.aiDiagnosisSummary);
                      return reportLang === 'ar' ? parsed.analysis_ar : parsed.analysis_en;
                    } catch { return profile.aiDiagnosisSummary; }
                  })()}
                </Text>
                <TouchableOpacity
                  style={styles.fullReportBtnRefined}
                  onPress={() => setShowAiModal(true)}
                >
                  <Text style={styles.fullReportBtnText}>{isRTL ? "عرض التقرير الكامل" : "Read Full Analysis"}</Text>
                  <ArrowRight size={14} color="#0EA5E9" />
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.aiEmptyStateRefined}>
                <Text style={[styles.aiEmptyTextRefined, { color: colors.textMuted }]}>{isRTL ? "لا يوجد تحليل حالياً. اضغط تحديث." : "No AI analysis yet. Tap refresh."}</Text>
              </View>
            )}
          </View>
        </View>

        {nextBooking && (
          <View style={styles.reminderContainer}>
            <TouchableOpacity
              onPress={() => router.push("/(patient)/profile?tab=activity")}
              activeOpacity={0.9}
            >
              <LinearGradient
                colors={isDark ? ["#064E3B", "#121B2E"] : ["#ECFDF5", "#F0FDF4"]}
                style={[styles.reminderCardCalm, { borderColor: isDark ? '#047857' : '#DCFCE7' }]}
              >
                <View style={styles.reminderHeaderCompact}>
                  <View style={[styles.reminderIconBoxCalm, { backgroundColor: isDark ? 'rgba(5, 150, 105, 0.2)' : '#fff' }]}>
                    <Calendar size={18} color="#059669" />
                  </View>
                  <View style={{ flex: 1, marginLeft: 15 }}>
                    <Text style={[styles.reminderTitleCalm, { color: isDark ? '#34D399' : '#059669' }]}>Upcoming Appointment</Text>
                    <Text style={[styles.reminderDoctorCalm, { color: isDark ? '#fff' : '#064E3B' }]}>Dr. {nextBooking.doctorName} • {nextBooking.time}</Text>
                  </View>
                  <ChevronRight size={20} color="#059669" />
                </View>

                {/* Decorative element */}
                <View style={[styles.cardCircle, { backgroundColor: 'rgba(5, 150, 105, 0.05)' }]} />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        )}

        {/* DAILY HEALTH TRACKER DASHBOARD */}
        <Animated.View
          entering={FadeInDown.delay(200).duration(800)}
          style={styles.healthDashboardContainer}
        >
          <LinearGradient
            colors={isDark ? ["#121B2E", "#0F172A"] : ["#FFFFFF", "#F0F9FF"]}
            style={[styles.healthDashboardCard, { borderColor: colors.border }]}
          >
            <View style={styles.dashboardHeader}>
              <View>
                <Text style={[styles.dashboardTitle, { color: colors.text }]}>{isRTL ? "متابعة صحتك اليوم" : "Today's Health Tracker"}</Text>
                <Text style={[styles.dashboardSubTitle, { color: colors.textMuted }]}>{isRTL ? "سجل قياساتك للحصول على تحليل دقيق" : "Track your vitals & medications"}</Text>
              </View>
              <View style={[styles.healthScoreCircle, { backgroundColor: isDark ? 'rgba(14,165,233,0.15)' : '#E0F2FE' }]}>
                <Activity size={20} color="#0EA5E9" />
              </View>
            </View>

            <View style={[styles.dashboardDivider, { backgroundColor: colors.border }]} />

            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.healthTasksRow}>
              <TouchableOpacity
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border }]}
                onPress={() => {
                  triggerHaptic();
                  router.push("/(patient)/vitals");
                }}
              >
                <View style={[styles.taskIconCircle, { backgroundColor: isDark ? 'rgba(14,165,233,0.1)' : '#F0F9FF' }]}>
                  <Activity size={20} color="#0EA5E9" />
                </View>
                <Text style={[styles.taskLabel, { color: colors.text }]}>{isRTL ? "القياسات" : "Vitals"}</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border }]}
                onPress={() => {
                  triggerHaptic();
                  router.push("/(patient)/medications");
                }}
              >
                <View style={[styles.taskIconCircle, { backgroundColor: isDark ? 'rgba(20, 184, 166, 0.1)' : '#F0FDFA' }]}>
                  <Pill size={20} color="#14B8A6" />
                </View>
                <Text style={[styles.taskLabel, { color: colors.text }]}>{isRTL ? "الأدوية" : "Meds"}</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border }]}
                onPress={() => {
                  triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
                  router.push("/(patient)/ai-profile-assistant");
                }}
              >
                <View style={[styles.taskIconCircle, { backgroundColor: isDark ? 'rgba(16,185,129,0.1)' : '#ECFDF5' }]}>
                  <Sparkles size={20} color="#10B981" />
                </View>
                <Text style={[styles.taskLabel, { color: colors.text }]}>{isRTL ? "مساعد AI" : "AI Assist"}</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border }]}
                onPress={() => {
                  triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
                  setShowAiModal(true);
                }}
              >
                <View style={[styles.taskIconCircle, { backgroundColor: isDark ? 'rgba(14,165,233,0.1)' : '#F0F9FF' }]}>
                  <Activity size={20} color="#0EA5E9" />
                </View>
                <Text style={[styles.taskLabel, { color: colors.text }]}>{isRTL ? "التقرير" : "Report"}</Text>
              </TouchableOpacity>
            </ScrollView>

            {showVitalReminder && (
              <TouchableOpacity
                style={styles.recordNowBtn}
                onPress={() => {
                  triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);
                  router.push("/(patient)/vitals");
                }}
              >
                <LinearGradient colors={["#0EA5E9", "#0284C7"]} style={styles.recordNowGradient}>
                  <Text style={styles.recordNowText}>{isRTL ? "سجل قياساتك الآن" : "Record Vitals Now"}</Text>
                  <ArrowRight size={16} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            )}
          </LinearGradient>
        </Animated.View>

        {/* MEDICATION DUE REMINDER CARD */}
        {nextDose && (
          <View style={styles.medTaskContainer}>
            <LinearGradient colors={isDark ? ["#042F2E", "#134E4A"] : ["#F0FDFA", "#CCFBF1"]} style={[styles.medTaskCard, { borderColor: isDark ? '#0D9488' : '#99F6E4' }]}>
              <View style={[styles.medTaskIconBox, { backgroundColor: isDark ? 'rgba(20, 184, 166, 0.2)' : '#fff' }]}>
                <Pill size={20} color="#14B8A6" />
              </View>
              <View style={{ flex: 1, marginLeft: 15 }}>
                <Text style={[styles.medTaskTitle, { color: isDark ? '#5EEAD4' : '#0F766E' }]}>Medication Due</Text>
                <Text style={[styles.medTaskName, { color: isDark ? '#fff' : '#134E4A' }]}>{nextDose.medicationName} • {nextDose.dosage}</Text>
              </View>
              <TouchableOpacity
                style={styles.medTaskBtn}
                onPress={() => router.push("/(patient)/medications")}
              >
                <Text style={styles.medTaskBtnText}>Take</Text>
                <ArrowRight size={14} color="#fff" />
              </TouchableOpacity>
            </LinearGradient>
          </View>
        )}

        <View style={styles.metricsArea}>
          <View style={styles.metricsRow}>
            <TouchableOpacity
              style={{ width: '48%' }}
              onPress={() => router.push("/(patient)/medications" as any)}
            >
              <SmallMetric icon={Pill} color="#14B8A6" bg={isDark ? "rgba(20, 184, 166, 0.2)" : "#F0FDFA"} label="Your Medications" val={nextDose ? nextDose.medicationName : "Up to date"} />
            </TouchableOpacity>

            <TouchableOpacity
              style={{ width: '48%' }}
              onPress={() => router.push("/(patient)/vitals" as any)}
            >
              <SmallMetric icon={HeartPulse} color="#0EA5E9" bg={isDark ? "rgba(14, 165, 233, 0.2)" : "#F0F9FF"} label="Record Vitals" val={lastBP ? `${lastBP.value}/${lastBP.value2}` : "No readings"} />
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Medical History & Records</Text>
        </View>
        <View style={styles.quickAccessRow}>
          <TouchableOpacity
            style={styles.fullHistoryBtn}
            onPress={() => router.push("/(patient)/profile?tab=history")}
            activeOpacity={0.8}
          >
            <LinearGradient colors={isDark ? ["#064E3B", "#121B2E"] : ["#ECFDF5", "#fff"]} style={[styles.fullHistoryGradient, { borderColor: colors.border }]}>
              <View style={styles.historyIconCircle}>
                <Clock size={24} color="#059669" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.fullHistoryTitle, { color: colors.text }]}>View Full Medical History</Text>
                <Text style={[styles.fullHistorySub, { color: colors.textMuted }]}>Allergies, Surgeries, Records & Folders</Text>
              </View>
              <ChevronRight size={20} color="#059669" />
            </LinearGradient>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomMetricsRow}>
          <SmallMetric icon={Activity} color="#0EA5E9" bg={isDark ? "rgba(14, 165, 233, 0.2)" : "#F0F9FF"} label="Vitals" val="Latest" onPress={() => router.push("/(patient)/vitals")} />
          <SmallMetric icon={Pill} color="#14B8A6" bg={isDark ? "rgba(20, 184, 166, 0.2)" : "#F0FDFA"} label="Meds" val="Schedule" onPress={() => router.push("/(patient)/medications")} />
        </View>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Medical Specialists</Text>
        </View>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.catScroll}>
          {CATEGORIES.map((cat, i) => (
            <TouchableOpacity
              key={i}
              style={styles.catCard}
              onPress={() => router.push({ pathname: "/(patient)/doctors", params: { specialty: cat.specialty } } as any)}
            >
              <LinearGradient colors={isDark ? ["#1E293B", "#121B2E"] : ["#fff", "#F1F5F9"]} style={[styles.catIconBox, { borderColor: colors.border }]}>
                <cat.icon size={24} color={colors.primary} />
              </LinearGradient>
              <Text style={[styles.catText, { color: colors.text }]}>{cat.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Top Rated Doctors</Text>
          <TouchableOpacity onPress={() => router.push("/(patient)/doctors")}><Text style={styles.seeMore}>View All</Text></TouchableOpacity>
        </View>

        <View style={styles.docList}>
          {loadingDocs ? (
            <ActivityIndicator color={COLORS.primary} style={{ marginTop: 20 }} />
          ) : (
            popularDocs.map((doc) => (
              <DoctorCard
                key={doc.id}
                doctor={{
                  ...doc,
                  id: String(doc.id),
                  experience: "5+ yrs",
                  location: "Main Clinic"
                }}
              />
            ))
          )}
        </View>
      </RNAnimated.ScrollView>

      {/* AI FULL REPORT MODAL */}
      <Modal visible={showAiModal} animationType="fade" transparent>
        <View style={styles.modalOverlayRefined}>
          <View style={[styles.modalContentRefined, { backgroundColor: colors.surface }]}>
            <View style={styles.modalHeaderRefined}>
              <Sparkles size={22} color="#0EA5E9" />
              <Text style={[styles.modalTitleRefined, { color: colors.text }]}>{isRTL ? "تقرير الصحة الذكي" : "Smart AI Report"}</Text>
              <TouchableOpacity onPress={() => setShowAiModal(false)}><Ionicons name="close" size={26} color={colors.textMuted} /></TouchableOpacity>
            </View>
            <ScrollView style={styles.modalBodyRefined}>
              <Text style={[styles.modalTextRefined, { color: colors.textMuted }, reportLang === 'ar' && { textAlign: 'right' }]}>
                {(() => {
                  try {
                    const parsed = JSON.parse(profile?.aiDiagnosisSummary || "{}");
                    return reportLang === 'ar' ? parsed.analysis_ar : parsed.analysis_en;
                  } catch { return profile?.aiDiagnosisSummary; }
                })()}
              </Text>
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* ULTRA-LUXURY FLOATING SEARCH */}
      <TouchableOpacity
        style={styles.floatingSearch}
        activeOpacity={0.95}
        onPress={() => router.push("/(patient)/doctors")}
      >
        <LinearGradient colors={["#10B981", "#059669"]} style={styles.fabGradient}>
          <Search size={28} color="#fff" />
        </LinearGradient>
      </TouchableOpacity>
    </View>
  );
}

function SmallMetric({ icon: Icon, color, bg, label, val, onPress }: any) {
  const { colors } = useTheme();
  return (
    <TouchableOpacity style={[styles.smallMetric, { backgroundColor: colors.surface, borderColor: colors.border }]} onPress={onPress} activeOpacity={0.7}>
      <View style={[styles.mIcon, { backgroundColor: bg }]}>
        <Icon size={20} color={color} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.mLabel, { color: colors.textMuted }]}>{label}</Text>
        <Text style={[styles.mVal, { color: colors.text }]} numberOfLines={1}>{val}</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  main: { flex: 1, backgroundColor: "#F8FAFC" },
  bgBubble: { position: 'absolute', borderRadius: 300, filter: 'blur(40px)' },
  bubbleTopLeft: { width: 350, height: 350, top: -100, left: -100 },
  bubbleBottomRight: { width: 400, height: 400, bottom: -150, right: -150 },
  bubbleCenter: { width: 250, height: 250, top: '40%', left: '20%' },
  scroll: { paddingBottom: 100 },
  magicHeaderContainer: { zIndex: 10 },
  magicHeader: {
    height: 250, paddingHorizontal: 20, paddingTop: 60,
    borderBottomLeftRadius: 40, borderBottomRightRadius: 40,
    position: 'relative', overflow: 'hidden'
  },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', zIndex: 10, marginBottom: 20 },
  findDoctorBtn: {
    zIndex: 10, marginTop: 10, backgroundColor: '#fff', height: 58, borderRadius: 20,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, elevation: 10, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 10
  },
  findDoctorContent: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  findDoctorText: { color: '#064E3B', fontSize: 15, fontWeight: '800' },
  findDoctorArrow: { width: 34, height: 34, borderRadius: 12, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center' },
  greetText: { fontSize: 13, color: "rgba(255,255,255,0.8)", fontWeight: '500' },
  nameRow: { flexDirection: 'row', alignItems: 'center', marginTop: 2 },
  userName: { fontSize: 19, fontWeight: "700", color: "#fff" },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 15 },
  headerIconBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center' },
  avatarWrap: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.2)', borderWidth: 2, borderColor: 'rgba(255,255,255,0.3)' },
  avatarImg: { width: '100%', height: '100%', borderRadius: 22 },
  avatarFallback: { width: '100%', height: '100%', borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
  headerDecor1: { position: 'absolute', width: 150, height: 150, borderRadius: 75, backgroundColor: 'rgba(255,255,255,0.1)', top: -50, right: -30 },
  headerDecor2: { position: 'absolute', width: 100, height: 100, borderRadius: 50, backgroundColor: 'rgba(255,255,255,0.05)', bottom: -20, left: 20 },

  tipWrapper: { paddingHorizontal: 20, marginTop: -30, zIndex: 20 },
  magicTipCard: {
    flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 28,
    elevation: 8, shadowColor: COLORS.primary, shadowOpacity: 0.15, shadowRadius: 15,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.8)'
  },
  tipIconBox: { width: 50, height: 50, borderRadius: 18, backgroundColor: '#FEF3C7', justifyContent: 'center', alignItems: 'center' },
  innerIconBox: { width: 38, height: 38, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  tipTitle: { fontSize: 14, fontWeight: '800', color: '#92400E', marginBottom: 2 },
  tipDesc: { fontSize: 12, color: '#B45309', lineHeight: 17, fontWeight: '500' },
  tipArrow: { width: 32, height: 32, borderRadius: 16, backgroundColor: 'rgba(5, 150, 105, 0.1)', justifyContent: 'center', alignItems: 'center' },

  metricsArea: { paddingHorizontal: 10, marginTop: 25, overflow: 'visible' },
  metricsRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 15, overflow: 'visible' },
  smallMetric: { flex: 1, marginHorizontal: 8, backgroundColor: '#fff', padding: 15, borderRadius: 22, flexDirection: 'row', alignItems: 'center', gap: 10, elevation: 5, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 8, borderWidth: 1, borderColor: '#F1F5F9', overflow: 'visible' },
  mIcon: { width: 36, height: 36, borderRadius: 10, justifyContent: 'center', alignItems: 'center' },
  mLabel: { fontSize: 9, color: '#94A3B8', fontWeight: '700', textTransform: 'uppercase' },
  mVal: { fontSize: 12, fontWeight: '700', color: '#1E293B', marginTop: 1 },

  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, marginTop: 25, marginBottom: 15 },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  seeMore: { fontSize: 13, color: COLORS.primary, fontWeight: "700" }, catScroll: { paddingHorizontal: 20, gap: 15, paddingBottom: 10 },
  catCard: { alignItems: 'center' },
  catIconBox: { width: 56, height: 56, borderRadius: 20, justifyContent: 'center', alignItems: 'center', elevation: 5, shadowOpacity: 0.05, backgroundColor: '#fff', borderWidth: 1, borderColor: '#F1F5F9' },
  catText: { fontSize: 10, fontWeight: '700', color: '#475569', marginTop: 6 },
  docList: { paddingHorizontal: 16, gap: 0, paddingBottom: 15 },
  floatingSearch: {
    position: 'absolute', bottom: 30, right: 25,
    width: 64, height: 64, borderRadius: 32,
    elevation: 12, shadowColor: '#059669', shadowOpacity: 0.4, shadowRadius: 15
  },
  fabGradient: { width: '100%', height: '100%', borderRadius: 32, justifyContent: 'center', alignItems: 'center' },

  reminderContainer: { paddingHorizontal: 20, marginTop: 15 },
  reminderCard: { borderRadius: 24, padding: 16, elevation: 8, shadowColor: '#064E3B', shadowOpacity: 0.25, shadowRadius: 15, overflow: 'hidden' },
  reminderHeaderCompact: { flexDirection: 'row', alignItems: 'center' },
  reminderIconBoxSmall: { width: 40, height: 40, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center' },
  reminderTitleSmall: { fontSize: 11, fontWeight: '600', color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', letterSpacing: 0.5 },
  reminderDoctorSmall: { fontSize: 16, fontWeight: '700', color: '#fff', marginTop: 2 },
  cardCircle: { position: 'absolute', bottom: -30, right: -30, width: 90, height: 90, borderRadius: 45, backgroundColor: 'rgba(255,255,255,0.08)' },

  vitalTaskContainer: { paddingHorizontal: 20, marginTop: 15 },
  vitalTaskCard: { flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 24, borderWidth: 1, borderColor: '#FED7AA', backgroundColor: '#FFF7ED' },
  vitalTaskIconBox: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  vitalTaskTitle: { fontSize: 11, fontWeight: '700', color: '#C2410C', textTransform: 'uppercase', letterSpacing: 0.5 },
  vitalTaskDesc: { fontSize: 15, fontWeight: '700', color: '#9A3412', marginTop: 1 },
  vitalTaskBtn: { backgroundColor: '#EA580C', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  vitalTaskBtnText: { color: '#fff', fontSize: 12, fontWeight: '800' },

  medTaskContainer: { paddingHorizontal: 20, marginTop: 15 },
  medTaskCard: { flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 24, borderWidth: 1, borderColor: '#99F6E4' },
  medTaskIconBox: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  medTaskTitle: { fontSize: 11, fontWeight: '700', color: '#0F766E', textTransform: 'uppercase', letterSpacing: 0.5 },
  medTaskName: { fontSize: 15, fontWeight: '700', color: '#134E4A', marginTop: 1 },
  medTaskBtn: { backgroundColor: '#14B8A6', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  medTaskBtnText: { color: '#fff', fontSize: 12, fontWeight: '800' },

  reminderCardCalm: { borderRadius: 24, padding: 16, elevation: 8, shadowColor: '#059669', shadowOpacity: 0.1, shadowRadius: 15, overflow: 'hidden', borderWidth: 1, borderColor: '#DCFCE7' },
  reminderIconBoxCalm: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  reminderTitleCalm: { fontSize: 11, fontWeight: '700', color: '#059669', textTransform: 'uppercase', letterSpacing: 0.5 },
  reminderDoctorCalm: { fontSize: 15, fontWeight: '700', color: '#064E3B', marginTop: 1 },

  quickAccessRow: { paddingHorizontal: 20, marginBottom: 15 },
  fullHistoryBtn: { borderRadius: 24, overflow: 'hidden', elevation: 6, shadowColor: '#059669', shadowOpacity: 0.1, shadowRadius: 15, borderWidth: 1, borderColor: '#DCFCE7' },
  fullHistoryGradient: { flexDirection: 'row', alignItems: 'center', padding: 20, gap: 15 },
  historyIconCircle: { width: 50, height: 50, borderRadius: 16, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', elevation: 4 },
  fullHistoryTitle: { fontSize: 15, fontWeight: '800', color: '#064E3B' },
  fullHistorySub: { fontSize: 11, color: '#059669', fontWeight: '600', marginTop: 2 },

  bottomMetricsRow: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 20, marginBottom: 25 },
  seeAllTxt: { fontSize: 12, fontWeight: '700', color: '#059669' },

  reminderDesc: { fontSize: 13, color: '#475569', lineHeight: 20, marginBottom: 18, fontWeight: '500' },
  reminderBtn: { borderRadius: 16, overflow: 'hidden', alignSelf: 'flex-start' },
  reminderBtnGradient: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 20, paddingVertical: 12 },
  reminderBtnText: { color: '#fff', fontSize: 14, fontWeight: '700' },

  healthDashboardContainer: { paddingHorizontal: 20, marginTop: 15 },
  healthDashboardCard: { borderRadius: 32, padding: 20, elevation: 12, shadowColor: '#0EA5E9', shadowOpacity: 0.12, shadowRadius: 20, borderWidth: 2, borderColor: '#BAE6FD' },
  dashboardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  dashboardTitle: { fontSize: 16, fontWeight: '900', color: '#1E293B' },
  dashboardSubTitle: { fontSize: 11, fontWeight: '600', color: '#64748B', marginTop: 2 },
  healthScoreCircle: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#F0F9FF', justifyContent: 'center', alignItems: 'center', borderWidth: 2, borderColor: '#BAE6FD' },
  healthScoreText: { fontSize: 12, fontWeight: '900', color: '#0EA5E9' },
  dashboardDivider: { height: 1, backgroundColor: '#F1F5F9', marginVertical: 15 },
  healthTasksRow: { flexDirection: 'row', alignItems: 'center' },
  healthTaskItem: { width: 95, backgroundColor: '#fff', padding: 12, borderRadius: 20, alignItems: 'center', borderWidth: 1, borderColor: '#F1F5F9', elevation: 2, marginRight: 10 },
  healthTaskDone: { borderColor: '#BAE6FD', backgroundColor: '#F0FDF4' },
  taskIconCircle: { width: 40, height: 40, borderRadius: 14, backgroundColor: '#F0F9FF', justifyContent: 'center', alignItems: 'center', marginBottom: 8 },
  taskIconDone: { backgroundColor: '#059669' },
  taskLabel: { fontSize: 11, fontWeight: '800', color: '#475569' },
  pendingBadge: { position: 'absolute', top: 5, right: 5, width: 16, height: 16, borderRadius: 8, backgroundColor: '#EF4444', justifyContent: 'center', alignItems: 'center' },
  pendingText: { color: '#fff', fontSize: 10, fontWeight: '900' },
  recordNowBtn: { marginTop: 15, borderRadius: 16, overflow: 'hidden' },
  recordNowGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, paddingVertical: 12 },
  recordNowText: { color: '#fff', fontSize: 13, fontWeight: '800' },

  aiInsightContainer: { paddingHorizontal: 20, marginTop: 15 },
  aiWhiteCardRefined: { backgroundColor: '#fff', borderRadius: 24, padding: 18, elevation: 6, shadowColor: '#0EA5E9', shadowOpacity: 0.1, shadowRadius: 15, borderWidth: 2, borderColor: '#BAE6FD' },
  aiHeaderRefined: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  aiIconBoxRefined: { width: 36, height: 36, borderRadius: 12, backgroundColor: '#F0F9FF', justifyContent: 'center', alignItems: 'center' },
  aiTitleRefined: { flex: 1, marginLeft: 12, fontSize: 15, fontWeight: '800', color: '#1E293B' },
  miniToggleBox: { flexDirection: 'row', backgroundColor: '#F1F5F9', borderRadius: 8, padding: 2 },
  miniToggleBtn: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  miniToggleBtnActive: { backgroundColor: '#fff', elevation: 2 },
  miniToggleText: { fontSize: 10, fontWeight: '800', color: '#64748B' },
  miniToggleTextActive: { color: '#0EA5E9' },
  aiContentRefined: {},
  aiTextRefined: { fontSize: 13, color: '#475569', lineHeight: 20, fontWeight: '500' },
  fullReportBtnRefined: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 12 },
  fullReportBtnText: { fontSize: 12, fontWeight: '800', color: '#0EA5E9' },
  aiEmptyStateRefined: { paddingVertical: 10 },
  aiEmptyTextRefined: { fontSize: 12, color: '#94A3B8', fontStyle: 'italic', textAlign: 'center' },

  modalOverlayRefined: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.6)', justifyContent: 'center', padding: 20 },
  modalContentRefined: { backgroundColor: '#fff', borderRadius: 32, padding: 25, maxHeight: '80%', elevation: 20 },
  modalHeaderRefined: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 },
  modalTitleRefined: { flex: 1, marginLeft: 12, fontSize: 18, fontWeight: '900', color: '#1E293B' },
  modalBodyRefined: {},
  modalTextRefined: { fontSize: 15, color: '#475569', lineHeight: 26, fontWeight: '500' },
});
