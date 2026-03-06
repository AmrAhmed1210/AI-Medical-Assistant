import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar,
} from "react-native";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEffect, useState, useCallback } from "react";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

const MOCK_POPULAR = [
  { id: "1", name: "Dr. Sarah Ahmed",  specialty: "Cardiology",  rating: 4.8, reviews: 124 },
  { id: "2", name: "Dr. Mohamed Ali",  specialty: "Dermatology", rating: 4.6, reviews: 89  },
  { id: "3", name: "Dr. Nour Hassan",  specialty: "Pediatrics",  rating: 4.9, reviews: 210 },
];

const CATEGORIES = [
  { icon: "heart-outline",    label: "Cardiology",   specialty: "Cardiology"   },
  { icon: "eye-outline",      label: "Eye",          specialty: "Eye"          },
  { icon: "fitness-outline",  label: "Ortho",        specialty: "Orthopedics"  },
  { icon: "body-outline",     label: "Neurology",    specialty: "Neurology"    },
  { icon: "bandage-outline",  label: "Dermatology",  specialty: "Dermatology"  },
  { icon: "person-outline",   label: "General",      specialty: "General"      },
];

export default function HomeScreen() {
  const router = useRouter();
  const [userName,      setUserName]      = useState("");
  const [nextBooking,   setNextBooking]   = useState<any>(null);

  useEffect(() => {
    AsyncStorage.getItem("userName").then((n) => { if (n) setUserName(n); });
  }, []);

  // يتحدث كل ما الصفحة تتفتح عشان يعرض الحجز الجديد
  useFocusEffect(
    useCallback(() => {
      AsyncStorage.getItem("my_bookings").then((raw) => {
        if (raw) {
          const list = JSON.parse(raw);
          if (list.length > 0) setNextBooking(list[0]); // أحدث حجز
          else setNextBooking(null);
        } else {
          setNextBooking(null);
        }
      });
    }, [])
  );

  const firstName = userName ? userName.split(" ")[0] : "Guest";

  const goToSpecialty = (specialty: string) => {
    router.push({ pathname: "/(patient)/doctors", params: { specialty } });
  };

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      contentContainerStyle={styles.content}
    >
      <StatusBar barStyle="dark-content" backgroundColor="#F4F6FA" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>Good morning 🌤</Text>
          <Text style={styles.userName}>{firstName}</Text>
        </View>
        <TouchableOpacity onPress={() => router.push("/(patient)/profile")} style={styles.avatarBtn}>
          <Ionicons name="person-outline" size={20} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      {/* ── Upcoming Booking Banner ── */}
      {nextBooking && (
        <TouchableOpacity
          style={styles.bookingBanner}
          onPress={() => router.push("/(patient)/profile")}
          activeOpacity={0.85}
        >
          <View style={styles.bannerLeft}>
            <Ionicons name="calendar" size={20} color="#fff" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.bannerTitle}>Upcoming Appointment</Text>
            <Text style={styles.bannerSub}>
              {nextBooking.doctorName} · {nextBooking.date} at {nextBooking.time}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={16} color={COLORS.primary} />
        </TouchableOpacity>
      )}

      {/* Hero Card */}
      <View style={styles.heroCard}>
        <View style={styles.heroLeft}>
          <Text style={styles.heroTitle}>Find your{"\n"}trusted doctor</Text>
          <TouchableOpacity
            style={styles.heroBtn}
            onPress={() => router.push("/(patient)/doctors")}
          >
            <Text style={styles.heroBtnText}>Search now</Text>
            <Ionicons name="arrow-forward" size={12} color={COLORS.primary} />
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
          { val: "200+", lbl: "Doctors",      accent: false },
          { val: "98%",  lbl: "Satisfaction", accent: true  },
          { val: "10k+", lbl: "Patients",     accent: false },
        ].map((s, i) => (
          <View key={i} style={[styles.statCard, s.accent && styles.statCardAccent]}>
            <Text style={[styles.statNum, s.accent && styles.statNumAccent]}>{s.val}</Text>
            <Text style={[styles.statLbl, s.accent && styles.statLblAccent]}>{s.lbl}</Text>
          </View>
        ))}
      </View>

      {/* Specialties */}
      <View style={styles.rowHeader}>
        <Text style={styles.sectionTitle}>Specialties</Text>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.catsRow}
      >
        {CATEGORIES.map((cat, i) => (
          <TouchableOpacity
            key={i}
            style={styles.catItem}
            onPress={() => goToSpecialty(cat.specialty)}
          >
            <View style={styles.catIcon}>
              <Ionicons name={cat.icon as any} size={18} color={COLORS.primary} />
            </View>
            <Text style={styles.catLabel}>{cat.label}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Popular Doctors */}
      <View style={styles.rowHeader}>
        <Text style={styles.sectionTitle}>Popular Doctors</Text>
        <TouchableOpacity onPress={() => router.push("/(patient)/doctors")}>
          <Text style={styles.seeAll}>See all</Text>
        </TouchableOpacity>
      </View>

      {MOCK_POPULAR.map((doc) => (
        <TouchableOpacity
          key={doc.id}
          style={styles.docCard}
          activeOpacity={0.82}
          onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: doc.id } })}
        >
          <View style={styles.docAvatar}>
            <Text style={styles.docAvatarTxt}>{doc.name.charAt(4)}</Text>
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.docName}>{doc.name}</Text>
            <Text style={styles.docSpec}>{doc.specialty}</Text>
            <View style={styles.ratingRow}>
              <Ionicons name="star" size={11} color="#FFB300" />
              <Text style={styles.ratingVal}>{doc.rating}</Text>
              <Text style={styles.ratingCnt}>({doc.reviews})</Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.bookBtn}
            onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: doc.id } })}
          >
            <Text style={styles.bookTxt}>Book</Text>
          </TouchableOpacity>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F4F6FA" },
  content:   { paddingBottom: 28 },
  header: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 18, paddingTop: 52, paddingBottom: 14,
  },
  greeting: { fontSize: 12, color: "#999", marginBottom: 1 },
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
  heroLeft:    { flex: 1, zIndex: 1 },
  heroTitle:   { color: "#fff", fontSize: 17, fontWeight: "800", lineHeight: 24, marginBottom: 12 },
  heroBtn: {
    flexDirection: "row", alignItems: "center", gap: 5,
    backgroundColor: "#fff", paddingHorizontal: 14, paddingVertical: 8,
    borderRadius: 18, alignSelf: "flex-start",
  },
  heroBtnText: { color: COLORS.primary, fontWeight: "700", fontSize: 12 },
  heroDecor:   { width: 90, alignItems: "center", justifyContent: "center", position: "relative" },
  decorCircle1: {
    position: "absolute", width: 90, height: 90, borderRadius: 45,
    backgroundColor: "rgba(255,255,255,0.1)", right: -20,
  },
  decorCircle2: {
    position: "absolute", width: 60, height: 60, borderRadius: 30,
    backgroundColor: "rgba(255,255,255,0.07)", right: 5, top: -10,
  },
  statsRow: { flexDirection: "row", marginHorizontal: 18, marginTop: 14, gap: 10 },
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
  catsRow:  { paddingHorizontal: 18, gap: 14 },
  catItem:  { alignItems: "center", gap: 5 },
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
  docAvatar: {
    width: 50, height: 50, borderRadius: 25,
    backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center",
  },
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
  bannerLeft: {
    width: 38, height: 38, borderRadius: 12,
    backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center",
  },
  bannerTitle: { fontSize: 12, fontWeight: "700", color: "#1A1A1A" },
  bannerSub:   { fontSize: 11, color: "#888", marginTop: 2 },
});