import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar, ActivityIndicator,
} from "react-native";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEffect, useState, useCallback } from "react";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useLanguage } from "../../context/LanguageContext";
import { getAllDoctors, Doctor } from "../../services/doctorService";
import { getMyAppointments, Appointment } from "../../services/appointmentService";

export default function HomeScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
  const [userName,     setUserName]     = useState("");
  const [popularDocs,  setPopularDocs]  = useState<Doctor[]>([]);
  const [nextBooking,  setNextBooking]  = useState<Appointment | null>(null);
  const [loadingDocs,  setLoadingDocs]  = useState(true);

  const CATEGORIES = [
    { icon: "heart-outline",   label: tr("spec_cardiology"),  specialty: "Cardiology"  },
    { icon: "eye-outline",     label: tr("spec_eye"),         specialty: "Eye"         },
    { icon: "fitness-outline", label: tr("spec_ortho"),       specialty: "Orthopedics" },
    { icon: "body-outline",    label: tr("spec_neurology"),   specialty: "Neurology"   },
    { icon: "bandage-outline", label: tr("spec_dermatology"), specialty: "Dermatology" },
    { icon: "person-outline",  label: tr("spec_general"),     specialty: "General"     },
  ];

  useEffect(() => {
    AsyncStorage.getItem("userName").then((n) => { if (n) setUserName(n); });
    fetchPopularDoctors();
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchNextBooking();
    }, [])
  );

  const fetchPopularDoctors = async () => {
    try {
      const data = await getAllDoctors();
      // أعلى 3 دكاترة تقييماً
      const sorted = [...data].sort((a, b) => b.rating - a.rating).slice(0, 3);
      setPopularDocs(sorted);
    } catch {
      // silently fail
    } finally {
      setLoadingDocs(false);
    }
  };

  const fetchNextBooking = async () => {
    try {
      const appts = await getMyAppointments();
      const active = appts.find(a => a.status !== "cancelled");
      setNextBooking(active ?? null);
    } catch {
      setNextBooking(null);
    }
  };

  const firstName = userName ? userName.split(" ")[0] : "Guest";
  const goToSpecialty = (specialty: string) =>
    router.push({ pathname: "/(patient)/doctors", params: { specialty } });

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      contentContainerStyle={styles.content}
    >
      <StatusBar barStyle="dark-content" backgroundColor="#F4F6FA" />

      {/* Header */}
      <View style={[styles.header, isRTL && styles.rowReverse]}>
        <View>
          <Text style={[styles.greeting, isRTL && styles.textRight]}>{tr("greeting")}</Text>
          <Text style={[styles.userName, isRTL && styles.textRight]}>{firstName}</Text>
        </View>
        <TouchableOpacity onPress={() => router.push("/(patient)/profile")} style={styles.avatarBtn}>
          <Ionicons name="person-outline" size={20} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      {/* Upcoming Booking Banner */}
      {nextBooking && (
        <TouchableOpacity
          style={[styles.bookingBanner, isRTL && { borderLeftWidth: 0, borderRightWidth: 4, borderRightColor: COLORS.primary }]}
          onPress={() => router.push("/(patient)/profile")}
          activeOpacity={0.85}
        >
          <View style={styles.bannerLeft}>
            <Ionicons name="calendar" size={20} color="#fff" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[styles.bannerTitle, isRTL && styles.textRight]}>{tr("upcoming_appointment")}</Text>
            <Text style={[styles.bannerSub, isRTL && styles.textRight]}>
              {nextBooking.doctorName} · {nextBooking.date} {nextBooking.time}
            </Text>
          </View>
          <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={16} color={COLORS.primary} />
        </TouchableOpacity>
      )}

      {/* Hero Card */}
      <View style={styles.heroCard}>
        <View style={styles.heroLeft}>
          <Text style={[styles.heroTitle, isRTL && styles.textRight]}>{tr("hero_title")}</Text>
          <TouchableOpacity style={styles.heroBtn} onPress={() => router.push("/(patient)/doctors")}>
            <Text style={styles.heroBtnText}>{tr("hero_btn")}</Text>
            <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={12} color={COLORS.primary} />
          </TouchableOpacity>
        </View>
        <View style={styles.heroDecor}>
          <View style={styles.decorCircle1} />
          <View style={styles.decorCircle2} />
          <Ionicons name="medical" size={42} color="rgba(255,255,255,0.75)" />
        </View>
      </View>

      {/* Stats */}
      <View style={styles.statsRow}>
        {[
          { val: "200+", lbl: tr("stats_doctors"),      accent: false },
          { val: "98%",  lbl: tr("stats_satisfaction"), accent: true  },
          { val: "10k+", lbl: tr("stats_patients"),     accent: false },
        ].map((s, i) => (
          <View key={i} style={[styles.statCard, s.accent && styles.statCardAccent]}>
            <Text style={[styles.statNum, s.accent && styles.statNumAccent]}>{s.val}</Text>
            <Text style={[styles.statLbl, s.accent && styles.statLblAccent]}>{s.lbl}</Text>
          </View>
        ))}
      </View>

      {/* Specialties */}
      <View style={[styles.rowHeader, isRTL && styles.rowReverse]}>
        <Text style={[styles.sectionTitle, isRTL && styles.textRight]}>{tr("specialties")}</Text>
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.catsRow}>
        {CATEGORIES.map((cat, i) => (
          <TouchableOpacity key={i} style={styles.catItem} onPress={() => goToSpecialty(cat.specialty)}>
            <View style={styles.catIcon}>
              <Ionicons name={cat.icon as any} size={18} color={COLORS.primary} />
            </View>
            <Text style={styles.catLabel}>{cat.label}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Popular Doctors */}
      <View style={[styles.rowHeader, isRTL && styles.rowReverse]}>
        <Text style={[styles.sectionTitle, isRTL && styles.textRight]}>{tr("popular_doctors")}</Text>
        <TouchableOpacity onPress={() => router.push("/(patient)/doctors")}>
          <Text style={styles.seeAll}>{tr("see_all")}</Text>
        </TouchableOpacity>
      </View>

      {loadingDocs ? (
        <ActivityIndicator color={COLORS.primary} style={{ marginTop: 20 }} />
      ) : (
        popularDocs.map((doc) => (
          <TouchableOpacity
            key={doc.id}
            style={styles.docCard}
            activeOpacity={0.82}
            onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doc.id) } })}
          >
            <View style={styles.docAvatar}>
              <Text style={styles.docAvatarTxt}>{doc.name.charAt(0)}</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[styles.docName, isRTL && styles.textRight]}>{doc.name}</Text>
              <Text style={[styles.docSpec, isRTL && styles.textRight]}>{doc.specialty}</Text>
              <View style={[styles.ratingRow, isRTL && styles.rowReverse]}>
                <Ionicons name="star" size={11} color="#FFB300" />
                <Text style={styles.ratingVal}>{doc.rating}</Text>
                <Text style={styles.ratingCnt}>({doc.reviewCount})</Text>
              </View>
            </View>
            <TouchableOpacity
              style={styles.bookBtn}
              onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doc.id) } })}
            >
              <Text style={styles.bookTxt}>{tr("book")}</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: "#F4F6FA" },
  content:    { paddingBottom: 28 },
  rowReverse: { flexDirection: "row-reverse" },
  textRight:  { textAlign: "right" },
  header: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 18, paddingTop: 52, paddingBottom: 14,
  },
  greeting:  { fontSize: 12, color: "#999", marginBottom: 1 },
  userName:  { fontSize: 22, fontWeight: "800", color: "#1A1A1A", letterSpacing: -0.3 },
  avatarBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: COLORS.primary + "15",
    justifyContent: "center", alignItems: "center",
    borderWidth: 1.5, borderColor: COLORS.primary + "30",
  },
  heroCard: {
    marginHorizontal: 18, borderRadius: 20, backgroundColor: COLORS.primary,
    flexDirection: "row", alignItems: "center", paddingLeft: 20, paddingVertical: 18,
    overflow: "hidden", shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.3, shadowRadius: 12, elevation: 6,
  },
  heroLeft:     { flex: 1, zIndex: 1 },
  heroTitle:    { color: "#fff", fontSize: 17, fontWeight: "800", lineHeight: 24, marginBottom: 12 },
  heroBtn: {
    flexDirection: "row", alignItems: "center", gap: 5,
    backgroundColor: "#fff", paddingHorizontal: 14, paddingVertical: 8,
    borderRadius: 18, alignSelf: "flex-start",
  },
  heroBtnText:  { color: COLORS.primary, fontWeight: "700", fontSize: 12 },
  heroDecor:    { width: 90, alignItems: "center", justifyContent: "center", position: "relative" },
  decorCircle1: { position: "absolute", width: 90, height: 90, borderRadius: 45, backgroundColor: "rgba(255,255,255,0.1)", right: -20 },
  decorCircle2: { position: "absolute", width: 60, height: 60, borderRadius: 30, backgroundColor: "rgba(255,255,255,0.07)", right: 5, top: -10 },
  statsRow:     { flexDirection: "row", marginHorizontal: 18, marginTop: 14, gap: 10 },
  statCard: {
    flex: 1, backgroundColor: "#fff", borderRadius: 14, paddingVertical: 10, alignItems: "center",
    shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 4, elevation: 1,
  },
  statCardAccent: { backgroundColor: COLORS.primary },
  statNum:        { fontSize: 15, fontWeight: "800", color: "#1A1A1A" },
  statNumAccent:  { color: "#fff" },
  statLbl:        { fontSize: 10, color: "#999", marginTop: 1 },
  statLblAccent:  { color: "rgba(255,255,255,0.75)" },
  rowHeader: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 18, marginTop: 22, marginBottom: 12,
  },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1A1A1A" },
  seeAll:       { fontSize: 12, color: COLORS.primary, fontWeight: "600" },
  catsRow:      { paddingHorizontal: 18, gap: 14 },
  catItem:      { alignItems: "center", gap: 5 },
  catIcon: {
    width: 48, height: 48, borderRadius: 16,
    backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center",
  },
  catLabel: { fontSize: 10, color: "#555", fontWeight: "500" },
  docCard: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#fff",
    marginHorizontal: 18, marginBottom: 10, borderRadius: 16, padding: 12, gap: 10,
    shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2,
  },
  docAvatar:    { width: 50, height: 50, borderRadius: 25, backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center" },
  docAvatarTxt: { fontSize: 20, fontWeight: "700", color: COLORS.primary },
  docName:      { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  docSpec:      { fontSize: 11, color: COLORS.primary, fontWeight: "500", marginTop: 1 },
  ratingRow:    { flexDirection: "row", alignItems: "center", gap: 3, marginTop: 3 },
  ratingVal:    { fontSize: 11, fontWeight: "600", color: "#333" },
  ratingCnt:    { fontSize: 10, color: "#AAA" },
  bookBtn:      { backgroundColor: COLORS.primary, paddingHorizontal: 14, paddingVertical: 7, borderRadius: 18 },
  bookTxt:      { color: "#fff", fontWeight: "700", fontSize: 12 },
  bookingBanner: {
    flexDirection: "row", alignItems: "center", gap: 12,
    backgroundColor: "#fff", marginHorizontal: 18, marginBottom: 14,
    borderRadius: 16, padding: 14,
    borderLeftWidth: 4, borderLeftColor: COLORS.primary,
    shadowColor: "#000", shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 3,
  },
  bannerLeft:  { width: 38, height: 38, borderRadius: 12, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center" },
  bannerTitle: { fontSize: 12, fontWeight: "700", color: "#1A1A1A" },
  bannerSub:   { fontSize: 11, color: "#888", marginTop: 2 },
});