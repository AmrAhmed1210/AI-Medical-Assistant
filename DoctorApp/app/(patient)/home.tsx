import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar, ActivityIndicator, Alert, Image, Dimensions
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import {
  Heart, Droplets, Calendar, Pill, Search,
  ChevronRight, Star, Bell, LayoutGrid, Stethoscope,
  ArrowRight, HeartPulse, Thermometer, Activity, User, Sparkles, Clock
} from "lucide-react-native";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useLanguage } from "../../context/LanguageContext";
import { useEffect, useState, useCallback, useRef } from "react";
import { Animated, Platform } from "react-native";
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
// Service loaded successfully

const { width } = Dimensions.get("window");

export default function HomeScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
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
  const scrollY = useRef(new Animated.Value(0)).current;

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

    return () => {
      unsubMed();
    };
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchHealthData();
      fetchNextBooking();
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
    <View style={styles.main}>
      <StatusBar barStyle="light-content" />
      <Animated.ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
        decelerationRate="fast"
        bounces={true}
      >
        <Animated.View style={[styles.magicHeaderContainer, { transform: [{ translateY: headerTranslateY }] }]}>
          <LinearGradient colors={["#064E3B", "#059669"]} style={styles.magicHeader}>
            <Animated.View style={[styles.headerTop, { opacity: headerOpacity }]}>
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
                  {profile?.photoUrl ? (
                    <Image source={{ uri: profile.photoUrl }} style={styles.avatarImg} />
                  ) : (
                    <View style={styles.avatarFallback}><User size={18} color="#fff" /></View>
                  )}
                </TouchableOpacity>
              </View>
            </Animated.View>

            <Animated.View style={{ transform: [{ scale: searchScale }] }}>
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
            </Animated.View>

            <Animated.View style={[styles.headerDecor1, { transform: [{ translateY: blob1Move }] }]} />
            <Animated.View style={[styles.headerDecor2, { transform: [{ translateY: Animated.multiply(blob1Move, 1.5) }] }]} />
          </LinearGradient>
        </Animated.View>

        {/* FLOATING GLASS TIP CARD */}
        <View style={styles.tipWrapper}>
          <LinearGradient
            colors={["rgba(255, 255, 255, 0.95)", "rgba(254, 252, 232, 0.9)"]}
            style={styles.magicTipCard}
          >
            <View style={styles.tipIconBox}>
              <View style={styles.innerIconBox}>
                <Ionicons name="bulb" size={22} color="#D97706" />
              </View>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.tipTitle}>Daily Wellness</Text>
              <Text style={styles.tipDesc}>"Small steps lead to great health. Stay active and hydrated."</Text>
            </View>
            <TouchableOpacity style={styles.tipArrow}>
              <ChevronRight size={18} color="#D97706" />
            </TouchableOpacity>
          </LinearGradient>
        </View>

        {nextBooking && (
          <View style={styles.reminderContainer}>
            <TouchableOpacity
              onPress={() => router.push("/(patient)/profile?tab=activity")}
              activeOpacity={0.9}
            >
              <LinearGradient
                colors={["#ECFDF5", "#F0FDF4"]}
                style={styles.reminderCardCalm}
              >
                <View style={styles.reminderHeaderCompact}>
                  <View style={styles.reminderIconBoxCalm}>
                    <Calendar size={18} color="#059669" />
                  </View>
                  <View style={{ flex: 1, marginLeft: 15 }}>
                    <Text style={styles.reminderTitleCalm}>Upcoming Appointment</Text>
                    <Text style={styles.reminderDoctorCalm}>Dr. {nextBooking.doctorName} • {nextBooking.time}</Text>
                  </View>
                  <ChevronRight size={20} color="#059669" />
                </View>

                {/* Decorative element */}
                <View style={[styles.cardCircle, { backgroundColor: 'rgba(5, 150, 105, 0.05)' }]} />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        )}

        {/* DAILY VITAL REMINDER CARD - Refined as a Health Task */}
        {showVitalReminder && (
          <View style={styles.vitalTaskContainer}>
            <LinearGradient colors={["#F0F9FF", "#E0F2FE"]} style={styles.vitalTaskCard}>
              <View style={styles.vitalTaskIconBox}>
                <Activity size={20} color="#0EA5E9" />
              </View>
              <View style={{ flex: 1, marginLeft: 15 }}>
                <Text style={styles.vitalTaskTitle}>Daily Vitals Record</Text>
                <Text style={styles.vitalTaskDesc}>Track your heart rate & BP today</Text>
              </View>
              <TouchableOpacity 
                style={styles.vitalTaskBtn}
                onPress={() => router.push("/(patient)/vitals")}
              >
                <Text style={styles.vitalTaskBtnText}>Record</Text>
                <ChevronRight size={14} color="#fff" />
              </TouchableOpacity>
            </LinearGradient>
          </View>
        )}

        {/* MEDICATION DUE REMINDER CARD */}
        {nextDose && (
          <View style={styles.medTaskContainer}>
            <LinearGradient colors={["#EEF2FF", "#E0E7FF"]} style={styles.medTaskCard}>
              <View style={styles.medTaskIconBox}>
                <Pill size={20} color="#4F46E5" />
              </View>
              <View style={{ flex: 1, marginLeft: 15 }}>
                <Text style={styles.medTaskTitle}>Medication Due</Text>
                <Text style={styles.medTaskName}>{nextDose.medicationName} • {nextDose.dosage}</Text>
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
              <SmallMetric icon={Pill} color="#6366F1" bg="#EEF2FF" label="Your Medications" val={nextDose ? nextDose.medicationName : "Up to date"} />
            </TouchableOpacity>

            <TouchableOpacity
              style={{ width: '48%' }}
              onPress={() => router.push("/(patient)/vitals" as any)}
            >
              <SmallMetric icon={HeartPulse} color="#EC4899" bg="#FDF2F8" label="Record Vitals" val={lastBP ? `${lastBP.value}/${lastBP.value2}` : "No readings"} />
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Medical History & Records</Text>
        </View>
        <View style={styles.quickAccessRow}>
          <TouchableOpacity 
            style={styles.fullHistoryBtn} 
            onPress={() => router.push("/(patient)/profile?tab=history")}
            activeOpacity={0.8}
          >
            <LinearGradient colors={["#ECFDF5", "#fff"]} style={styles.fullHistoryGradient}>
              <View style={styles.historyIconCircle}>
                <Clock size={24} color="#059669" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.fullHistoryTitle}>View Full Medical History</Text>
                <Text style={styles.fullHistorySub}>Allergies, Surgeries, Records & Folders</Text>
              </View>
              <ChevronRight size={20} color="#059669" />
            </LinearGradient>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomMetricsRow}>
          <SmallMetric icon={Activity} color="#0EA5E9" bg="#F0F9FF" label="Vitals" val="Latest" onPress={() => router.push("/(patient)/vitals")} />
          <SmallMetric icon={Pill} color="#F59E0B" bg="#FFF7ED" label="Meds" val="Schedule" onPress={() => router.push("/(patient)/medications")} />
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Medical Specialists</Text>
        </View>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.catScroll}>
          {CATEGORIES.map((cat, i) => (
            <TouchableOpacity
              key={i}
              style={styles.catCard}
              onPress={() => router.push({ pathname: "/(patient)/doctors", params: { specialty: cat.specialty } } as any)}
            >
              <LinearGradient colors={["#fff", "#F1F5F9"]} style={styles.catIconBox}>
                <cat.icon size={24} color={COLORS.primary} />
              </LinearGradient>
              <Text style={styles.catText}>{cat.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Top Rated Doctors</Text>
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
      </Animated.ScrollView>

      {/* ULTRA-LUXURY FLOATING SEARCH */}
      <TouchableOpacity
        style={styles.floatingSearch}
        activeOpacity={0.95}
        onPress={() => router.push("/(patient)/doctors")}
      >
        <LinearGradient colors={["#FBBF24", "#D97706"]} style={styles.fabGradient}>
          <Search size={28} color="#fff" />
        </LinearGradient>
      </TouchableOpacity>
    </View>
  );
}

function SmallMetric({ icon: Icon, color, bg, label, val, onPress }: any) {
  return (
    <TouchableOpacity style={styles.smallMetric} onPress={onPress} activeOpacity={0.7}>
      <View style={[styles.mIcon, { backgroundColor: bg }]}>
        <Icon size={20} color={color} />
      </View>
      <View>
        <Text style={styles.mLabel}>{label}</Text>
        <Text style={styles.mVal}>{val}</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  main: { flex: 1, backgroundColor: "#F8FAFC" },
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
  tipArrow: { width: 32, height: 32, borderRadius: 16, backgroundColor: 'rgba(217, 119, 6, 0.1)', justifyContent: 'center', alignItems: 'center' },

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
    elevation: 12, shadowColor: '#D97706', shadowOpacity: 0.4, shadowRadius: 15
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
  vitalTaskCard: { flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 24, borderWidth: 1, borderColor: '#BAE6FD' },
  vitalTaskIconBox: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  vitalTaskTitle: { fontSize: 11, fontWeight: '700', color: '#0EA5E9', textTransform: 'uppercase', letterSpacing: 0.5 },
  vitalTaskDesc: { fontSize: 15, fontWeight: '700', color: '#0C4A6E', marginTop: 1 },
  vitalTaskBtn: { backgroundColor: '#0EA5E9', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  vitalTaskBtnText: { color: '#fff', fontSize: 12, fontWeight: '800' },

  medTaskContainer: { paddingHorizontal: 20, marginTop: 15 },
  medTaskCard: { flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 24, borderWidth: 1, borderColor: '#C7D2FE' },
  medTaskIconBox: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  medTaskTitle: { fontSize: 11, fontWeight: '700', color: '#4F46E5', textTransform: 'uppercase', letterSpacing: 0.5 },
  medTaskName: { fontSize: 15, fontWeight: '700', color: '#1E1B4B', marginTop: 1 },
  medTaskBtn: { backgroundColor: '#4F46E5', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
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
});
