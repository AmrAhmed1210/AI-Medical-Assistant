import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar, ActivityIndicator, Alert, Image, Dimensions, Modal, TextInput, PanResponder
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import {
  Heart, Droplets, Calendar, Pill, Search,
  ChevronRight, Star, Bell, LayoutGrid, Stethoscope,
  ArrowRight, HeartPulse, Thermometer, Activity, User, Sparkles, Clock, CheckCircle2, EyeOff, Eye
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
import { getAllDoctors, getDoctorById, getReviewsByDoctor, Doctor, getRecommendedDoctorsForNeed, formatDoctorRecommendationsForAi, enrichDoctorsWithReviewStats, sortDoctorsFairly } from "../../services/doctorService";
import { getMyProfile, Profile } from "../../services/profileService";
import { getMyAppointments, Appointment } from "../../services/appointmentService";
import { getMedicationSchedule, type MedicationScheduleItem } from "../../services/medicationService";
import { getMyPatientId } from "../../services/authService";
import { getLatestVital, getPatientVitals, type VitalReading, addVitalReading, checkVitalNormal } from "../../services/vitalService";
import { NotificationBell } from "../../components/NotificationBell";
import DoctorCard from "../../components/DoctorCard";
import { addNotification } from "../../services/notificationService";
import { startSignalRConnection, onDoctorUpdated, onScheduleReady, onScheduleUpdated, onNewConsultation, onNewMedication } from "../../services/signalr";
import { scheduleAppointmentReminders } from "../../services/appointmentReminders";
import { analyzePatientHistory, getDailyHealthTip } from "../../services/aiService";
import { updateAiDiagnosis, getVitals, getSurgeries, getMedications, getAllergies, getChronicDiseases, getPatientDocuments } from "../../services/medicalRecordService";
import { getMyVisits, type PatientVisit } from "../../services/visitService";

const { width } = Dimensions.get("window");

const DEFAULT_AVATAR = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png";

function resolvePhotoUrl(url?: string | null): string {
  if (!url || !url.trim()) return DEFAULT_AVATAR;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  const separator = url.startsWith('/') ? '' : '/';
  return `${BASE_URL}${separator}${url}`;
}

function formatRecentVisitsForAi(visits: PatientVisit[]) {
  const cutoff = new Date();
  cutoff.setMonth(cutoff.getMonth() - 8);
  return visits
    .filter((v) => v.status === "closed" && new Date(v.visitDate || v.createdAt) >= cutoff)
    .map((v) => ({
      date: v.visitDate,
      doctor: v.doctorName,
      specialty: v.doctorSpecialty,
      complaint: v.chiefComplaint,
      assessment: v.assessment,
      plan: v.plan,
    }));
}

type AiVitalReading = {
  readingType: string;
  value: number;
  value2?: number;
  unit?: string;
  isNormal?: boolean;
  recordedAt: string;
};

function summarizeVitalsForAi(vitals: AiVitalReading[]) {
  const sortedVitals = [...vitals].sort(
    (a, b) => new Date(b.recordedAt).getTime() - new Date(a.recordedAt).getTime()
  );
  const latestByType = new Map<string, AiVitalReading>();

  for (const reading of sortedVitals) {
    if (!latestByType.has(reading.readingType)) {
      latestByType.set(reading.readingType, reading);
    }
  }

  const abnormalRecent = sortedVitals
    .filter((reading) => !reading.isNormal)
    .slice(0, 12);

  return {
    total_count: vitals.length,
    latest_by_type: Array.from(latestByType.values()).map((reading) => ({
      type: reading.readingType,
      value: reading.value,
      value2: reading.value2,
      unit: reading.unit,
      isNormal: reading.isNormal,
      recordedAt: reading.recordedAt,
    })),
    abnormal_recent: abnormalRecent.map((reading) => ({
      type: reading.readingType,
      value: reading.value,
      value2: reading.value2,
      unit: reading.unit,
      recordedAt: reading.recordedAt,
    })),
  };
}

function getAiReportText(summary: string | null | undefined, lang: 'en' | 'ar') {
  if (!summary) return "";
  try {
    const parsed = JSON.parse(summary);
    return lang === 'ar'
      ? (parsed.analysis_ar || parsed.ar || "")
      : (parsed.analysis_en || parsed.en || "");
  } catch {
    return summary;
  }
}

function formatReadableReport(raw: string) {
  return (raw || "")
    .replace(/\r\n/g, "\n")
    .replace(/\*\*/g, "")
    .replace(/#{1,6}\s*/g, "")
    .replace(/(?:^|\n)\s*[-•]\s*/g, "\n• ")
    .replace(/\. (?=(General Summary|Recent Visits|Warnings|Medications|Safe Advice|Recommended Doctors|الملخص|الزيارات|التحذيرات|الأدوية|النصائح|الأطباء))/g, ".\n\n")
    .replace(/(General Summary|Recent Visits Summary \(Last 8 Months\)|Warnings|Medications & Interactions|Safe Advice|Recommended Doctors|الملخص العام|ملخص الزيارات|التحذيرات|الأدوية والتداخلات|نصائح آمنة|الأطباء المقترحون)\s*:?/g, "\n\n$1\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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
  const [recommendedDoctors, setRecommendedDoctors] = useState<Doctor[]>([]);
  const [recommendedSpecialty, setRecommendedSpecialty] = useState<string | null>(null);
  const scrollY = useRef(new RNAnimated.Value(0)).current;

  const [dailySysBp, setDailySysBp] = useState("");
  const [dailyDiaBp, setDailyDiaBp] = useState("");
  const [dailySugar, setDailySugar] = useState("");
  const [dailyHeartRate, setDailyHeartRate] = useState("");
  const [dailyTemp, setDailyTemp] = useState("");
  const [dailyWeight, setDailyWeight] = useState("");
  const [isSavingDaily, setIsSavingDaily] = useState(false);
  const [showDailySuccess, setShowDailySuccess] = useState(false);
  const [showDailyVitalsModal, setShowDailyVitalsModal] = useState(false);
  const [dailyTip, setDailyTip] = useState<{ tip_en: string; tip_ar: string }>({
    tip_en: "Small steps lead to great health. Stay active and hydrated.",
    tip_ar: "خطوات صغيرة تقود لصحة عظيمة. ابقَ نشيطاً واشرب الماء."
  });

  const [isFabHidden, setIsFabHidden] = useState(false);
  const pan = useRef(new RNAnimated.ValueXY()).current;
  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) => {
        // Only start dragging if the user moves a bit, to not interfere with simple taps
        return Math.abs(gestureState.dx) > 5 || Math.abs(gestureState.dy) > 5;
      },
      onPanResponderGrant: () => {
        pan.setOffset({
          x: (pan.x as any)._value,
          y: (pan.y as any)._value
        });
        pan.setValue({ x: 0, y: 0 });
      },
      onPanResponderMove: RNAnimated.event(
        [null, { dx: pan.x, dy: pan.y }],
        { useNativeDriver: false }
      ),
      onPanResponderRelease: () => {
        pan.flattenOffset();
      }
    })
  ).current;

  // Animation values
  const pulseScale = useSharedValue(1);

  useEffect(() => {
    setReportLang(isRTL ? 'ar' : 'en');
  }, [isRTL]);

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
    getDailyHealthTip().then(setDailyTip).catch(() => {});

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

  useEffect(() => {
    if (profile?.aiDiagnosisSummary) {
      refreshDoctorRecommendations().catch(() => undefined);
    }
  }, [profile?.aiDiagnosisSummary]);

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
      refreshDoctorRecommendations().catch(() => undefined);
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
        // Automatically open the daily vitals modal once a day
        const lastModalDate = await AsyncStorage.getItem("last_vital_modal_date");
        if (lastModalDate !== today) {
          setShowDailyVitalsModal(true);
          await AsyncStorage.setItem("last_vital_modal_date", today);
        }

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
      const enriched = await enrichDoctorsWithReviewStats(data);
      setPopularDocs(sortDoctorsFairly(enriched).slice(0, 10));
    } finally {
      setLoadingDocs(false);
    }
  };

  const refreshDoctorRecommendations = async () => {
    try {
      const pid = await getMyPatientId();
      if (!pid) return;

      const [vitals, chronic, docs] = await Promise.all([
        getVitals(pid).catch(() => []),
        getChronicDiseases(pid).catch(() => []),
        getPatientDocuments(pid).catch(() => []),
      ]);

      const needText = [
        ...chronic.map(c => c.diseaseName),
        ...vitals.slice(0, 10).map(v => `${v.readingType} ${v.value}${v.value2 ? `/${v.value2}` : ""}`),
        ...docs.map(d => `${d.title ?? (d as any).Title ?? ""} ${d.description ?? (d as any).Description ?? ""}`),
        profile?.aiDiagnosisSummary ?? "",
      ].join(" ");

      const recommendation = await getRecommendedDoctorsForNeed(needText, 5);
      setRecommendedSpecialty(recommendation.specialty);
      setRecommendedDoctors(recommendation.doctors);
    } catch {
      setRecommendedSpecialty(null);
    }
  };

  const parseApptDate = (dateStr: string) => {
    if (!dateStr) return null;
    const cleaned = dateStr.trim();
    
    // Try YYYY-MM-DD
    const isoMatch = cleaned.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (isoMatch) {
      return new Date(Number(isoMatch[1]), Number(isoMatch[2]) - 1, Number(isoMatch[3]));
    }
    
    // Try "D MMM YYYY" or "DD MMM YYYY" (e.g., "21 Jun 2026" or "7 Mar 2026")
    const mmmMatch = cleaned.match(/^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/);
    if (mmmMatch) {
      const day = Number(mmmMatch[1]);
      const mmm = mmmMatch[2].toLowerCase().slice(0, 3);
      const year = Number(mmmMatch[3]);
      
      const months: Record<string, number> = {
        jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
        jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11
      };
      
      const month = months[mmm];
      if (month !== undefined) {
        return new Date(year, month, day);
      }
    }
    
    // Fallback to native parsing
    const parsed = new Date(cleaned);
    return isNaN(parsed.getTime()) ? null : parsed;
  };

  const getAppointmentDateTime = (dateStr: string, timeStr: string) => {
    const baseDate = parseApptDate(dateStr);
    if (!baseDate) return null;
    
    let hours = 0;
    let minutes = 0;
    if (timeStr) {
      const timeMatch = timeStr.match(/(\d+):(\d+)\s*(AM|PM)?/i);
      if (timeMatch) {
        hours = Number(timeMatch[1]);
        minutes = Number(timeMatch[2]);
        const ampm = timeMatch[3];
        if (ampm) {
          if (ampm.toUpperCase() === "PM" && hours !== 12) hours += 12;
          if (ampm.toUpperCase() === "AM" && hours === 12) hours = 0;
        }
      }
    }
    
    baseDate.setHours(hours, minutes, 0, 0);
    return baseDate;
  };

  const formatReminderDate = (dateStr: string) => {
    const dateObj = parseApptDate(dateStr);
    if (!dateObj) return dateStr;
    return dateObj.toLocaleDateString(isRTL ? "ar-EG" : "en-US", { day: "numeric", month: "short", year: "numeric" });
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

          const apptDateTime = getAppointmentDateTime(a.date, a.time);
          if (apptDateTime) {
            const isFuture = apptDateTime > now;
            console.log(`Checking Appt: Dr.${a.doctorName}, Date:${a.date}, Time:${a.time}, Status:${status}, isFuture:${isFuture}`);
            return isRelevant && isFuture;
          }
          return false;
        })
        .sort((a, b) => {
          const da = getAppointmentDateTime(a.date, a.time);
          const db = getAppointmentDateTime(b.date, b.time);
          return (da?.getTime() || 0) - (db?.getTime() || 0);
        });

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

      const [vitals, surgeries, meds, allergies, chronic, docs, visits] = await Promise.all([
        getVitals(pid),
        getSurgeries(pid),
        getMedications(pid),
        getAllergies(pid),
        getChronicDiseases(pid),
        getPatientDocuments(pid).catch(() => []),
        getMyVisits().catch(() => []),
      ]);

      console.log("DEBUG: Patient docs fetched in Home:", JSON.stringify(docs));
      const needText = [
        ...chronic.map(c => c.diseaseName),
        ...summarizeVitalsForAi(vitals).latest_by_type.map(v => `${v.type} ${v.value}${v.value2 ? `/${v.value2}` : ""}`),
        ...docs.map(d => `${d.title ?? (d as any).Title ?? ""} ${d.description ?? (d as any).Description ?? ""}`),
      ].join(" ");
      const doctorRecommendations = await getRecommendedDoctorsForNeed(needText, 5).catch(() => ({ specialty: null, doctors: [] }));
      setRecommendedDoctors(doctorRecommendations.doctors);
      setRecommendedSpecialty(doctorRecommendations.specialty);
      const analysis = await analyzePatientHistory({
        vitals: summarizeVitalsForAi(vitals),
        surgeries: surgeries.map(s => s.surgeryName),
        medications: meds.map(m => m.medicationName),
        allergies: allergies.map(a => a.allergenName),
        chronic_diseases: chronic.map(c => ({ diseaseName: c.diseaseName })),
        documents_analysis: docs.map(d => ({ 
          title: d.title ?? (d as any).Title ?? "", 
          ai_analysis: d.description ?? (d as any).Description ?? "" 
        })),
        recent_visits: formatRecentVisitsForAi(visits),
        recommended_doctors: formatDoctorRecommendationsForAi(doctorRecommendations.doctors),
        recommended_specialty: doctorRecommendations.specialty,
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

  const handleSaveDailyVitals = async () => {
    if ((!dailySysBp || !dailyDiaBp) && !dailySugar) {
      Alert.alert(isRTL ? "تنبيه" : "Alert", isRTL ? "يرجى إدخال الضغط أو السكر" : "Please enter BP or Sugar");
      return;
    }
    const pid = await getMyPatientId();
    if (!pid) return;

    setIsSavingDaily(true);
    try {
      if (dailySysBp && dailyDiaBp) {
        await addVitalReading(pid, {
          readingType: "Blood Pressure",
          value: parseFloat(dailySysBp),
          value2: parseFloat(dailyDiaBp),
          unit: "mmHg",
          isNormal: checkVitalNormal("Blood Pressure", parseFloat(dailySysBp), parseFloat(dailyDiaBp))
        });
      }
      if (dailySugar) {
        await addVitalReading(pid, {
          readingType: "Blood Sugar",
          value: parseFloat(dailySugar),
          unit: "mg/dL",
          isNormal: checkVitalNormal("Blood Sugar", parseFloat(dailySugar))
        });
      }
      if (dailyHeartRate) {
        await addVitalReading(pid, {
          readingType: "Heart Rate",
          value: parseFloat(dailyHeartRate),
          unit: "bpm",
          isNormal: checkVitalNormal("Heart Rate", parseFloat(dailyHeartRate))
        });
      }
      if (dailyTemp) {
        await addVitalReading(pid, {
          readingType: "Temperature",
          value: parseFloat(dailyTemp),
          unit: "C",
          isNormal: checkVitalNormal("Temperature", parseFloat(dailyTemp))
        });
      }
      if (dailyWeight) {
        await addVitalReading(pid, {
          readingType: "Weight",
          value: parseFloat(dailyWeight),
          unit: "kg",
          isNormal: checkVitalNormal("Weight", parseFloat(dailyWeight))
        });
      }

      // Refresh latest vitals
      const bp = await getLatestVital(pid, "Blood Pressure");
      const sugar = await getLatestVital(pid, "Blood Sugar");
      if (bp) setLastBP(bp);
      if (sugar) setLastSugar(sugar);

      setShowDailySuccess(true);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => {
        setShowDailyVitalsModal(false);
        setShowDailySuccess(false);
        setDailySysBp("");
        setDailyDiaBp("");
        setDailySugar("");
        setDailyHeartRate("");
        setDailyTemp("");
        setDailyWeight("");
      }, 1500);
    } catch (error) {
      Alert.alert("Error", "Could not save vitals.");
    } finally {
      setIsSavingDaily(false);
    }
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return tr("greeting_morning") || "Good Morning";
    if (hour < 18) return isRTL ? "مساء الخير" : "Good Afternoon";
    return isRTL ? "مساء الخير" : "Good Evening";
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
            <RNAnimated.View style={[styles.headerTop, { opacity: headerOpacity, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <View style={{ alignItems: isRTL ? 'flex-end' : 'flex-start' }}>
                <Text style={styles.greetText}>{getGreeting()}</Text>
                <View style={[styles.nameRow, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Text style={styles.userName}>{profile?.name?.split(" ")[0] || tr("patient")}</Text>
                  <Sparkles size={16} color="#FDE047" style={{ [isRTL ? "marginRight" : "marginLeft"]: 4 }} />
                </View>
              </View>
              <View style={[styles.headerRight, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
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
                style={[styles.findDoctorBtn, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}
                activeOpacity={0.8}
                onPress={() => router.push("/(patient)/doctors")}
              >
                <View style={[styles.findDoctorContent, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Stethoscope size={20} color="#059669" />
                  <Text style={styles.findDoctorText}>{tr("find_specialist")}</Text>
                </View>
                <View style={[styles.findDoctorArrow, { transform: [{ scaleX: isRTL ? -1 : 1 }] }]}>
                  <ArrowRight size={18} color="#059669" />
                </View>
              </TouchableOpacity>
            </RNAnimated.View>

            <RNAnimated.View style={[styles.headerDecor1, { transform: [{ translateY: blob1Move }] }]} />
            <RNAnimated.View style={[styles.headerDecor2, { transform: [{ translateY: RNAnimated.multiply(blob1Move, 1.5) }] }]} />
          </LinearGradient>
        </RNAnimated.View>

        {/* FLOATING GLASS TIP CARD — AI-generated, refreshes every 6h */}
        <View style={styles.tipWrapper}>
          <LinearGradient
            colors={isDark ? ["rgba(30, 41, 59, 0.95)", "rgba(15, 23, 42, 0.9)"] : ["rgba(255, 255, 255, 0.95)", "rgba(240, 253, 244, 0.9)"]}
            style={[styles.magicTipCard, { borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(5, 150, 105, 0.12)', flexDirection: isRTL ? 'row-reverse' : 'row' }]}
          >
            <View style={[styles.tipIconBox, { backgroundColor: isDark ? 'rgba(16, 185, 129, 0.2)' : '#ECFDF5' }]}>
              <View style={[styles.innerIconBox, { backgroundColor: isDark ? '#059669' : '#fff' }]}>
                <Ionicons name="bulb" size={22} color={isDark ? "#fff" : "#059669"} />
              </View>
            </View>
            <View style={{ flex: 1, marginHorizontal: 12, alignItems: isRTL ? 'flex-end' : 'flex-start' }}>
              <Text style={[styles.tipTitle, { color: isDark ? '#6EE7B7' : '#064E3B' }]}>{isRTL ? "نصيحة اليوم ✨" : "Daily Wellness ✨"}</Text>
              <Text style={[styles.tipDesc, { color: isDark ? '#34D399' : '#047857' }, { textAlign: isRTL ? 'right' : 'left' }]}>
                {isRTL ? dailyTip.tip_ar : dailyTip.tip_en}
              </Text>
            </View>
            <TouchableOpacity 
              style={styles.tipArrow}
              onPress={() => {
                AsyncStorage.removeItem("daily_health_tip");
                getDailyHealthTip().then(setDailyTip).catch(() => {});
              }}
            >
              <Ionicons name="refresh" size={16} color="#059669" />
            </TouchableOpacity>
          </LinearGradient>
        </View>

        {/* AI HEALTH INSIGHTS CARD - REFINED BILINGUAL */}
        <View style={styles.aiInsightContainer}>
          <View style={[styles.aiWhiteCardRefined, { backgroundColor: colors.surface, borderColor: isDark ? '#0284C7' : '#BAE6FD' }]}>
            <View style={[styles.aiHeaderRefined, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <View style={[styles.aiIconBoxRefined, { backgroundColor: isDark ? 'rgba(14, 165, 233, 0.2)' : '#F0F9FF' }]}>
                <Sparkles size={20} color="#0EA5E9" />
              </View>
              <Text style={[styles.aiTitleRefined, { color: colors.text, [isRTL ? "marginRight" : "marginLeft"]: 12 }]}>{isRTL ? "تحليل الذكاء الاصطناعي" : "AI Health Insights"}</Text>

              <View style={[styles.miniToggleBox, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                <TouchableOpacity onPress={() => setReportLang('en')} style={[styles.miniToggleBtn, reportLang === 'en' && styles.miniToggleBtnActive]}>
                  <Text style={[styles.miniToggleText, reportLang === 'en' && styles.miniToggleTextActive]}>EN</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setReportLang('ar')} style={[styles.miniToggleBtn, reportLang === 'ar' && styles.miniToggleBtnActive]}>
                  <Text style={[styles.miniToggleText, reportLang === 'ar' && styles.miniToggleTextActive]}>AR</Text>
                </TouchableOpacity>
              </View>

              {isAnalyzing ? (
                <ActivityIndicator size="small" color="#0EA5E9" style={{ [isRTL ? "marginRight" : "marginLeft"]: 10 }} />
              ) : (
                <TouchableOpacity onPress={runAiAnalysis} style={{ [isRTL ? "marginRight" : "marginLeft"]: 10 }}>
                  <Clock size={18} color="#0EA5E9" />
                </TouchableOpacity>
              )}
            </View>

            {profile?.aiDiagnosisSummary ? (
              <View style={[styles.aiContentRefined, { alignItems: isRTL ? 'flex-end' : 'flex-start' }]}>
                <Text style={[styles.aiTextRefined, { color: colors.textMuted }, { textAlign: reportLang === 'ar' ? 'right' : 'left' }]} numberOfLines={6}>
                  {formatReadableReport(getAiReportText(profile.aiDiagnosisSummary, reportLang))}
                </Text>
                {(() => {
                  let needsDoc = true;
                  try {
                    const parsed = JSON.parse(profile.aiDiagnosisSummary);
                    needsDoc = parsed.needsDoctor !== false;
                  } catch {}
                  if (!needsDoc) return null;
                  if (recommendedDoctors.length > 0) {
                    return (
                      <View style={[styles.docRecommendSection, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                        <Text style={[styles.docRecommendLabel, { color: colors.textMuted }]}>{tr("recommended_doctors")}</Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={[styles.docRecommendScroll, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                          {recommendedDoctors.map((doc) => (
                            <TouchableOpacity
                              key={doc.id}
                              style={[styles.docRecommendChip, { backgroundColor: colors.background, borderColor: colors.border }]}
                              onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { id: doc.id } } as any)}
                              activeOpacity={0.7}
                            >
                              <Text style={[styles.docRecommendName, { color: colors.text }]} numberOfLines={1}>{doc.name}</Text>
                              <Text style={[styles.docRecommendSpecialty, { color: colors.primary }]} numberOfLines={1}>{doc.specialty}</Text>
                            </TouchableOpacity>
                          ))}
                        </ScrollView>
                      </View>
                    );
                  }
                  if (recommendedSpecialty) {
                    return (
                      <View style={[styles.docRecommendSection, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                        <Text style={[styles.docRecommendLabel, { color: colors.textMuted }]}>{tr("recommended_specialty")}</Text>
                        <TouchableOpacity 
                          style={[styles.docRecommendChip, { backgroundColor: isDark ? 'rgba(234,179,8,0.1)' : '#FEF9C3', borderColor: isDark ? 'rgba(234,179,8,0.2)' : '#FEF08A', marginTop: 8, padding: 12, width: '100%', alignItems: 'center' }]}
                          onPress={() => router.push({ pathname: "/(patient)/doctors", params: { specialty: recommendedSpecialty } } as any)}
                        >
                          <Text style={[styles.docRecommendName, { color: isDark ? '#FDE047' : '#854D0E', textAlign: 'center', marginBottom: 4 }]} numberOfLines={2}>
                            {tr("no_doctors_specialty")} ({recommendedSpecialty}).
                          </Text>
                          <Text style={[styles.docRecommendSpecialty, { color: isDark ? '#FEF08A' : '#A16207', textAlign: 'center' }]} numberOfLines={2}>
                            {tr("tap_search_specialty")}
                          </Text>
                        </TouchableOpacity>
                      </View>
                    );
                  }
                  return null;
                })()}
                <TouchableOpacity
                  style={[styles.fullReportBtnRefined, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}
                  onPress={() => setShowAiModal(true)}
                >
                  <Text style={styles.fullReportBtnText}>{isRTL ? "عرض التقرير الكامل" : "Read Full Analysis"}</Text>
                  <ArrowRight size={14} color="#0EA5E9" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.aiEmptyStateRefined}>
                <Text style={[styles.aiEmptyTextRefined, { color: colors.textMuted }]}>{isRTL ? "لا يوجد تحليل حالياً. اضغط تحديث." : "No AI analysis yet. Tap refresh."}</Text>
              </View>
            )}
          </View>
        </View>

        {/* MEDICATION DUE REMINDER CARD */}
        {nextDose && (
          <View style={styles.medTaskContainer}>
            <LinearGradient colors={isDark ? ["#042F2E", "#134E4A"] : ["#F0FDFA", "#CCFBF1"]} style={[styles.medTaskCard, { borderColor: isDark ? '#0D9488' : '#99F6E4', flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <View style={[styles.medTaskIconBox, { backgroundColor: isDark ? 'rgba(20, 184, 166, 0.2)' : '#fff' }]}>
                <Pill size={20} color="#14B8A6" />
              </View>
              <View style={{ flex: 1, [isRTL ? "marginRight" : "marginLeft"]: 15, alignItems: isRTL ? 'flex-end' : 'flex-start' }}>
                <Text style={[styles.medTaskTitle, { color: isDark ? '#5EEAD4' : '#0F766E' }]}>{tr("medication_due")}</Text>
                <Text style={[styles.medTaskName, { color: isDark ? '#fff' : '#134E4A' }]}>{nextDose.medicationName} • {nextDose.dosage}</Text>
              </View>
              <TouchableOpacity
                style={[styles.medTaskBtn, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}
                onPress={() => router.push("/(patient)/medications")}
              >
                <Text style={styles.medTaskBtnText}>{tr("take")}</Text>
                <ArrowRight size={14} color="#fff" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
              </TouchableOpacity>
            </LinearGradient>
          </View>
        )}

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
                <View style={[styles.reminderHeaderCompact, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <View style={[styles.reminderIconBoxCalm, { backgroundColor: isDark ? 'rgba(5, 150, 105, 0.2)' : '#fff' }]}>
                    <Calendar size={18} color="#059669" />
                  </View>
                  <View style={{ flex: 1, [isRTL ? "marginRight" : "marginLeft"]: 15 }}>
                    <Text style={[styles.reminderTitleCalm, { color: isDark ? '#34D399' : '#059669' }, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("upcoming_appointment")}</Text>
                    <Text style={[styles.reminderDoctorCalm, { color: isDark ? '#fff' : '#064E3B' }, { textAlign: isRTL ? 'right' : 'left' }]}>{isRTL ? `د. ${nextBooking.doctorName}` : `Dr. ${nextBooking.doctorName}`} • {formatReminderDate(nextBooking.date)} • {nextBooking.time}</Text>
                  </View>
                  <ChevronRight size={20} color="#059669" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
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
            <View style={[styles.dashboardHeader, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <View style={{ alignItems: isRTL ? 'flex-end' : 'flex-start' }}>
                <Text style={[styles.dashboardTitle, { color: colors.text }]}>{isRTL ? "متابعة صحتك اليوم" : "Today's Health Tracker"}</Text>
                <Text style={[styles.dashboardSubTitle, { color: colors.textMuted }]}>{isRTL ? "سجل قياساتك للحصول على تحليل دقيق" : "Track your vitals & medications"}</Text>
              </View>
              <View style={[styles.healthScoreCircle, { backgroundColor: isDark ? 'rgba(14,165,233,0.15)' : '#E0F2FE' }]}>
                <Activity size={20} color="#0EA5E9" />
              </View>
            </View>

            <View style={[styles.dashboardDivider, { backgroundColor: colors.border }]} />

            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={[styles.healthTasksRow, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <TouchableOpacity
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border, marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }]}
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
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border, marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }]}
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
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border, marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }]}
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
                style={[styles.healthTaskItem, { backgroundColor: colors.surface, borderColor: colors.border, marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }]}
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
                <LinearGradient colors={["#0EA5E9", "#0284C7"]} style={[styles.recordNowGradient, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Text style={styles.recordNowText}>{isRTL ? "سجل قياساتك الآن" : "Record Vitals Now"}</Text>
                  <ArrowRight size={16} color="#fff" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
                </LinearGradient>
              </TouchableOpacity>
            )}
          </LinearGradient>
        </Animated.View>

        <View style={styles.metricsArea}>
          <View style={[styles.metricsRow, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
            <TouchableOpacity
              style={{ width: '48%' }}
              onPress={() => router.push("/(patient)/medications" as any)}
            >
              <SmallMetric icon={Pill} color="#14B8A6" bg={isDark ? "rgba(20, 184, 166, 0.2)" : "#F0FDFA"} label={isRTL ? "أدويتك" : "Your Medications"} val={nextDose ? nextDose.medicationName : (isRTL ? "مكتملة اليوم" : "Up to date")} isRTL={isRTL} />
            </TouchableOpacity>

            <TouchableOpacity
              style={{ width: '48%' }}
              onPress={() => router.push("/(patient)/vitals" as any)}
            >
              <SmallMetric icon={HeartPulse} color="#0EA5E9" bg={isDark ? "rgba(14, 165, 233, 0.2)" : "#F0F9FF"} label={isRTL ? "تسجيل القياسات" : "Record Vitals"} val={lastBP ? `${lastBP.value}/${lastBP.value2}` : (isRTL ? "لا توجد قياسات" : "No readings")} isRTL={isRTL} />
            </TouchableOpacity>
          </View>
        </View>

        <View style={[styles.sectionHeader, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{isRTL ? "السجل الطبي والمستندات" : "Medical History & Records"}</Text>
        </View>
        <View style={styles.quickAccessRow}>
          <TouchableOpacity
            style={styles.fullHistoryBtn}
            onPress={() => router.push("/(patient)/profile?tab=history")}
            activeOpacity={0.8}
          >
            <LinearGradient colors={isDark ? ["#064E3B", "#121B2E"] : ["#ECFDF5", "#fff"]} style={[styles.fullHistoryGradient, { borderColor: colors.border, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <View style={styles.historyIconCircle}>
                <Clock size={24} color="#059669" />
              </View>
              <View style={{ flex: 1, alignItems: isRTL ? 'flex-end' : 'flex-start', marginHorizontal: isRTL ? 15 : 0 }}>
                <Text style={[styles.fullHistoryTitle, { color: colors.text }]}>{tr("view_history")}</Text>
                <Text style={[styles.fullHistorySub, { color: colors.textMuted }]}>{isRTL ? "الحساسية، العمليات، المجلدات والتقارير" : "Allergies, Surgeries, Records & Folders"}</Text>
              </View>
              <ChevronRight size={20} color="#059669" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
            </LinearGradient>
          </TouchableOpacity>
        </View>

        <View style={[styles.bottomMetricsRow, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          <SmallMetric icon={Activity} color="#0EA5E9" bg={isDark ? "rgba(14, 165, 233, 0.2)" : "#F0F9FF"} label={isRTL ? "العلامات الحيوية" : "Vitals"} val={isRTL ? "أحدث القياسات" : "Latest"} onPress={() => router.push("/(patient)/vitals")} isRTL={isRTL} />
          <SmallMetric icon={Pill} color="#14B8A6" bg={isDark ? "rgba(20, 184, 166, 0.2)" : "#F0FDFA"} label={isRTL ? "الأدوية" : "Meds"} val={isRTL ? "الجدول اليومي" : "Schedule"} onPress={() => router.push("/(patient)/medications")} isRTL={isRTL} />
        </View>

        <View style={[styles.sectionHeader, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{isRTL ? "التخصصات الطبية" : "Medical Specialists"}</Text>
        </View>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={[styles.catScroll, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          {CATEGORIES.map((cat, i) => (
            <TouchableOpacity
              key={i}
              style={[styles.catCard, { marginRight: isRTL ? 0 : 15, marginLeft: isRTL ? 15 : 0 }]}
              onPress={() => router.push({ pathname: "/(patient)/doctors", params: { specialty: cat.specialty } } as any)}
            >
              <LinearGradient colors={isDark ? ["#1E293B", "#121B2E"] : ["#fff", "#F1F5F9"]} style={[styles.catIconBox, { borderColor: colors.border }]}>
                <cat.icon size={24} color={colors.primary} />
              </LinearGradient>
              <Text style={[styles.catText, { color: colors.text }]}>{cat.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <View style={[styles.sectionHeader, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>
            {recommendedDoctors.length > 0 ? tr("recommended_for_you") : tr("top_rated_doctors")}
          </Text>
          <TouchableOpacity onPress={() => router.push("/(patient)/doctors")}><Text style={styles.seeMore}>{tr("view_all")}</Text></TouchableOpacity>
        </View>
        {recommendedSpecialty && recommendedDoctors.length > 0 && (
          <Text style={[styles.docRecommendLabel, { color: colors.textMuted, paddingHorizontal: 20, marginBottom: 10, textAlign: isRTL ? "right" : "left" }]}>
            {tr("best_match")}: {recommendedSpecialty}
          </Text>
        )}

        <View style={styles.docList}>
          {loadingDocs ? (
            <ActivityIndicator color={COLORS.primary} style={{ marginTop: 20 }} />
          ) : (
            (recommendedDoctors.length > 0 ? recommendedDoctors : popularDocs).slice(0, 5).map((doc) => (
              <DoctorCard
                key={doc.id}
                doctor={{
                  ...doc,
                  id: String(doc.id),
                  experience: doc.yearsExperience ? `${doc.yearsExperience}+ yrs` : "5+ yrs",
                  location: doc.location || "Medical Center"
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
            <View style={[styles.modalHeaderRefined, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <Sparkles size={22} color="#0EA5E9" />
              <Text style={[styles.modalTitleRefined, { color: colors.text, [isRTL ? "marginRight" : "marginLeft"]: 12, textAlign: isRTL ? "right" : "left" }]}>{isRTL ? "تقرير الصحة الذكي" : "Smart AI Report"}</Text>
              <TouchableOpacity onPress={() => setShowAiModal(false)}><Ionicons name="close" size={26} color={colors.textMuted} /></TouchableOpacity>
            </View>
            <ScrollView style={styles.modalBodyRefined}>
              <Text selectable style={[styles.modalTextRefined, { color: colors.textMuted }, { textAlign: reportLang === 'ar' ? 'right' : 'left' }]}>
                {formatReadableReport(getAiReportText(profile?.aiDiagnosisSummary, reportLang))}
              </Text>
              {(() => {
                let needsDoc = true;
                try {
                  const parsed = JSON.parse(profile?.aiDiagnosisSummary || "{}");
                  needsDoc = parsed.needsDoctor !== false;
                } catch {}
                if (!needsDoc) return null;
                if (recommendedDoctors.length > 0) {
                  return (
                    <View style={[styles.modalDocSection, { borderTopColor: colors.border }]}>
                      <Text style={[styles.modalDocLabel, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}>{tr("recommended_doctors")}</Text>
                      {recommendedDoctors.map((doc) => (
                        <TouchableOpacity
                          key={doc.id}
                          style={[styles.modalDocCard, { backgroundColor: colors.background, borderColor: colors.border, flexDirection: isRTL ? 'row-reverse' : 'row' }]}
                          onPress={() => { setShowAiModal(false); router.push({ pathname: "/(patient)/doctor-details", params: { id: doc.id } } as any); }}
                          activeOpacity={0.7}
                        >
                          <View style={[styles.modalDocInfo, { alignItems: isRTL ? 'flex-end' : 'flex-start', marginHorizontal: isRTL ? 12 : 0 }]}>
                            <Text style={[styles.modalDocName, { color: colors.text }]}>{doc.name}</Text>
                            <Text style={[styles.modalDocSpecialty, { color: colors.primary }]}>{doc.specialty}</Text>
                            <Text style={[styles.modalDocMeta, { color: colors.textMuted }]}>{doc.location} · {doc.rating?.toFixed(1)} ⭐</Text>
                          </View>
                          <ChevronRight size={18} color={colors.textMuted} style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
                        </TouchableOpacity>
                      ))}
                    </View>
                  );
                }
                if (recommendedSpecialty) {
                  return (
                    <View style={[styles.modalDocSection, { borderTopColor: colors.border }]}>
                      <Text style={[styles.modalDocLabel, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}>{tr("recommended_specialty")}</Text>
                      <TouchableOpacity 
                        style={[styles.modalDocCard, { backgroundColor: isDark ? 'rgba(234,179,8,0.1)' : '#FEF9C3', borderColor: isDark ? 'rgba(234,179,8,0.2)' : '#FEF08A', alignItems: 'center', paddingVertical: 16 }]}
                        onPress={() => { setShowAiModal(false); router.push({ pathname: "/(patient)/doctors", params: { specialty: recommendedSpecialty } } as any); }}
                        activeOpacity={0.7}
                      >
                        <Text style={[styles.modalDocName, { color: isDark ? '#FDE047' : '#854D0E', textAlign: 'center' }]}>
                          {tr("no_doctors_specialty")} ({recommendedSpecialty})
                        </Text>
                        <Text style={[styles.modalDocMeta, { color: isDark ? '#FEF08A' : '#A16207', textAlign: 'center', marginTop: 4 }]}>
                          {tr("tap_search_specialty")}
                        </Text>
                      </TouchableOpacity>
                    </View>
                  );
                }
                return null;
              })()}
              
              {/* Disclaimer */}
              <Text style={{ marginTop: 24, fontSize: 11, color: colors.textMuted, textAlign: 'center', fontStyle: 'italic', paddingHorizontal: 16 }}>
                {isRTL 
                  ? "ملاحظة: الذكاء الاصطناعي قد يخطئ في التحليل والترشيحات. يجب دائمًا استشارة طبيب متخصص وتأكيد المعلومات الطبية بنفسك." 
                  : "Note: AI can make mistakes in analysis and recommendations. Always consult a specialist doctor and verify medical information yourself."}
              </Text>
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* DRAGGABLE & HIDEABLE STACKED FABs */}
      <RNAnimated.View 
        {...panResponder.panHandlers}
        style={[
          styles.stackedFabContainer,
          isRTL ? { left: 25, right: undefined } : { right: 25, left: undefined },
          { transform: [{ translateX: pan.x }, { translateY: pan.y }] }
        ]}
      >
        {isFabHidden ? (
          <TouchableOpacity 
            style={styles.fabHiddenHandle} 
            activeOpacity={0.8}
            onPress={() => setIsFabHidden(false)}
          >
            <Eye size={20} color="rgba(255,255,255,0.8)" />
          </TouchableOpacity>
        ) : (
          <View style={styles.fabVisibleContainer}>
            {/* Hide Button */}
            <TouchableOpacity 
              style={styles.fabHideBtn} 
              activeOpacity={0.8}
              onPress={() => setIsFabHidden(true)}
            >
              <EyeOff size={16} color="rgba(255,255,255,0.9)" />
            </TouchableOpacity>

            {/* Search FAB */}
            <TouchableOpacity
              style={[styles.fabItem, { backgroundColor: '#10B981' }]}
              activeOpacity={0.9}
              onPress={() => router.push("/(patient)/doctors")}
            >
              <Search size={22} color="#fff" />
            </TouchableOpacity>

            {/* Vitals FAB */}
            <TouchableOpacity 
              style={[styles.fabItem, { backgroundColor: '#059669', marginTop: 12 }]}
              activeOpacity={0.9}
              onPress={() => setShowDailyVitalsModal(true)}
            >
              <Activity size={22} color="#fff" />
            </TouchableOpacity>
          </View>
        )}
      </RNAnimated.View>

      {/* DAILY VITALS BOTTOM SHEET MODAL */}
      <Modal visible={showDailyVitalsModal} animationType="slide" transparent={true}>
        <View style={styles.dailyModalOverlay}>
          <TouchableOpacity style={styles.dailyModalBg} onPress={() => setShowDailyVitalsModal(false)} />
          <View style={[styles.dailyModalContent, { backgroundColor: colors.card }]}>
            <View style={styles.dailyModalHandle} />
            <Text style={[styles.dailyModalTitle, { color: colors.text }]}>
              {isRTL ? "الفحص اليومي 🩺" : "Daily Check-in 🩺"}
            </Text>
            
            {!showDailySuccess ? (
              <ScrollView style={styles.dailyModalForm} contentContainerStyle={{ paddingBottom: 20 }}>
                <Text style={{ color: colors.textMuted, marginBottom: 12, textAlign: 'center' }}>
                  {isRTL ? "سجل قياساتك اليومية لمتابعة صحتك" : "Record your daily vitals to track your health"}
                </Text>

                {/* Blood Pressure Input */}
                <View style={[styles.modalInputGroup, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc', borderColor: colors.border, marginBottom: 12, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Activity size={20} color="#059669" style={isRTL ? { marginLeft: 12 } : styles.dailyIcon} />
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "الانقباضي" : "Sys"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailySysBp}
                    onChangeText={setDailySysBp}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 18 }}>/</Text>
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "الانبساطي" : "Dia"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailyDiaBp}
                    onChangeText={setDailyDiaBp}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 12, [isRTL ? "marginRight" : "marginLeft"]: 8 }}>mmHg</Text>
                </View>

                {/* Blood Sugar Input */}
                <View style={[styles.modalInputGroup, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc', borderColor: colors.border, marginBottom: 12, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Droplets size={20} color="#3B82F6" style={isRTL ? { marginLeft: 12 } : styles.dailyIcon} />
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "السكر" : "Blood Sugar"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailySugar}
                    onChangeText={setDailySugar}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 12, [isRTL ? "marginRight" : "marginLeft"]: 8 }}>mg/dL</Text>
                </View>

                {/* Heart Rate Input */}
                <View style={[styles.modalInputGroup, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc', borderColor: colors.border, marginBottom: 12, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <HeartPulse size={20} color="#EF4444" style={isRTL ? { marginLeft: 12 } : styles.dailyIcon} />
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "معدل النبض" : "Heart Rate"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailyHeartRate}
                    onChangeText={setDailyHeartRate}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 12, [isRTL ? "marginRight" : "marginLeft"]: 8 }}>bpm</Text>
                </View>

                {/* Temperature Input */}
                <View style={[styles.modalInputGroup, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc', borderColor: colors.border, marginBottom: 12, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <Thermometer size={20} color="#F59E0B" style={isRTL ? { marginLeft: 12 } : styles.dailyIcon} />
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "درجة الحرارة" : "Temperature"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailyTemp}
                    onChangeText={setDailyTemp}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 12, [isRTL ? "marginRight" : "marginLeft"]: 8 }}>C°</Text>
                </View>

                {/* Weight Input */}
                <View style={[styles.modalInputGroup, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc', borderColor: colors.border, marginBottom: 12, flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                  <User size={20} color="#8B5CF6" style={isRTL ? { marginLeft: 12 } : styles.dailyIcon} />
                  <TextInput
                    style={[styles.modalInput, { color: colors.text, textAlign: isRTL ? 'right' : 'left' }]}
                    placeholder={isRTL ? "الوزن" : "Weight"}
                    placeholderTextColor={colors.textMuted}
                    keyboardType="numeric"
                    value={dailyWeight}
                    onChangeText={setDailyWeight}
                  />
                  <Text style={{ color: colors.textMuted, fontSize: 12, [isRTL ? "marginRight" : "marginLeft"]: 8 }}>kg</Text>
                </View>

                <TouchableOpacity 
                  style={[styles.modalSaveBtn, (!dailySysBp && !dailyDiaBp && !dailySugar && !dailyHeartRate && !dailyTemp && !dailyWeight) && { opacity: 0.5 }]}
                  onPress={handleSaveDailyVitals}
                  disabled={isSavingDaily || (!dailySysBp && !dailyDiaBp && !dailySugar && !dailyHeartRate && !dailyTemp && !dailyWeight)}
                >
                  {isSavingDaily ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text style={styles.modalSaveBtnText}>{isRTL ? "حفظ القياسات" : "Save Vitals"}</Text>
                  )}
                </TouchableOpacity>
              </ScrollView>
            ) : (
              <View style={styles.dailySuccessState}>
                <CheckCircle2 size={50} color="#059669" style={{ marginBottom: 16 }} />
                <Text style={{ color: '#059669', fontWeight: 'bold', fontSize: 18, textAlign: 'center' }}>
                  {isRTL ? "تم تسجيل قياسات اليوم بنجاح! 🌟" : "Today's vitals saved successfully! 🌟"}
                </Text>
              </View>
            )}
          </View>
        </View>
      </Modal>

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
  findDoctorArrow: {
    width: 30, height: 30, borderRadius: 15, backgroundColor: 'rgba(5, 150, 105, 0.1)',
    justifyContent: 'center', alignItems: 'center',
  },
  fabContainer: {
    position: 'absolute', bottom: 30, left: 25, width: 60, height: 60, borderRadius: 30,
    justifyContent: 'center', alignItems: 'center', elevation: 8, shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3, shadowRadius: 8, zIndex: 100
  },
  dailyModalOverlay: { flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.5)' },
  dailyModalBg: { ...StyleSheet.absoluteFillObject },
  dailyModalContent: { padding: 24, borderTopLeftRadius: 30, borderTopRightRadius: 30, minHeight: 300, elevation: 10 },
  dailyModalHandle: { width: 40, height: 5, backgroundColor: 'rgba(150,150,150,0.3)', borderRadius: 3, alignSelf: 'center', marginBottom: 20 },
  dailyModalTitle: { fontSize: 20, fontWeight: '800', textAlign: 'center', marginBottom: 16 },
  dailyModalForm: { gap: 16 },
  modalInputGroup: { flexDirection: 'row', alignItems: 'center', borderRadius: 16, paddingHorizontal: 16, height: 56, borderWidth: 1 },
  dailyIcon: { marginRight: 12 },
  modalInput: { flex: 1, fontSize: 16, textAlign: 'center', minWidth: 40 },
  modalSaveBtn: { height: 56, borderRadius: 16, backgroundColor: '#059669', justifyContent: 'center', alignItems: 'center', marginTop: 10, shadowColor: '#059669', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8 },
  modalSaveBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  dailySuccessState: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingVertical: 40 },
  tipWrapper: { paddingHorizontal: 16, marginTop: -12, marginBottom: 20, zIndex: 10 },
  magicTipCard: { flexDirection: 'row', alignItems: 'center', padding: 14, borderRadius: 20, borderWidth: 1, elevation: 4, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 12 },
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
  stackedFabContainer: {
    position: 'absolute', bottom: 30, right: 25, zIndex: 100,
    elevation: 12, shadowColor: '#059669', shadowOpacity: 0.4, shadowRadius: 15
  },
  fabVisibleContainer: {
    alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.9)', 
    padding: 8, borderRadius: 30, borderWidth: 1, borderColor: 'rgba(16, 185, 129, 0.2)'
  },
  fabItem: {
    width: 50, height: 50, borderRadius: 25, 
    justifyContent: 'center', alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 4
  },
  fabHideBtn: {
    width: 30, height: 30, borderRadius: 15, backgroundColor: '#94A3B8',
    justifyContent: 'center', alignItems: 'center', marginBottom: 12
  },
  fabHiddenHandle: {
    width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(16, 185, 129, 0.6)',
    justifyContent: 'center', alignItems: 'center', borderWidth: 2, borderColor: '#fff'
  },

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
  docRecommendSection: { marginTop: 12 },
  docRecommendLabel: { fontSize: 11, fontWeight: '800', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 },
  docRecommendScroll: { gap: 10, paddingRight: 20 },
  docRecommendChip: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: 16, borderWidth: 1, minWidth: 120 },
  docRecommendName: { fontSize: 13, fontWeight: '700' },
  docRecommendSpecialty: { fontSize: 10, fontWeight: '600', marginTop: 2 },
  modalDocSection: { marginTop: 20, paddingTop: 16, borderTopWidth: 1, gap: 10 },
  modalDocLabel: { fontSize: 14, fontWeight: '800', marginBottom: 4 },
  modalDocCard: { flexDirection: 'row', alignItems: 'center', padding: 14, borderRadius: 16, borderWidth: 1 },
  modalDocInfo: { flex: 1 },
  modalDocName: { fontSize: 14, fontWeight: '700' },
  modalDocSpecialty: { fontSize: 12, fontWeight: '600', marginTop: 2 },
  modalDocMeta: { fontSize: 11, fontWeight: '500', marginTop: 4 },
});


