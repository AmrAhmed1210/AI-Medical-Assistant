import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  View, Text, FlatList, StyleSheet, ActivityIndicator,
  TextInput, TouchableOpacity, RefreshControl, ScrollView,
  Platform, StatusBar,
} from "react-native";
import { useLocalSearchParams, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { COLORS } from "../../constants/colors";
import DoctorCard from "@/components/DoctorCard";
import { useLanguage } from "../../context/LanguageContext";
import { getAllDoctors, getDoctorById, getReviewsByDoctor, Doctor } from "../../services/doctorService";
import { onDoctorCreated, onDoctorUpdated, startSignalRConnection } from "../../services/signalr";

const SPECIALTIES = ["All", "Cardiology", "Dermatology", "Neurology", "Orthopedics", "Pediatrics", "Gynecology", "Ophthalmology", "ENT"];

const STATUS_BAR_HEIGHT = Platform.OS === "android" ? (StatusBar.currentHeight ?? 24) : 44;

export default function DoctorsScreen() {
  const params = useLocalSearchParams<{ specialty?: string }>();
  const { tr, isRTL } = useLanguage();

  const [doctors,    setDoctors]    = useState<Doctor[]>([]);
  const [filtered,   setFiltered]   = useState<Doctor[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error,      setError]      = useState("");
  const [search,     setSearch]     = useState("");
  const [activeSpec, setActiveSpec] = useState("All");
  const [highlightedDoctorId, setHighlightedDoctorId] = useState<string | null>(null);
  const highlightTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (params.specialty) {
      const match = SPECIALTIES.find((s) => s.toLowerCase() === params.specialty!.toLowerCase());
      if (match) setActiveSpec(match);
    }
  }, [params.specialty]);

  const fetchDoctors = useCallback(async () => {
    try {
      setLoading(true)
      setError("")
      const data = await getAllDoctors()
      const enriched = await Promise.all(
        data.map(async (doctor) => {
          try {
            const [details, reviews] = await Promise.all([
              getDoctorById(doctor.id),
              getReviewsByDoctor(doctor.id).catch(() => []),
            ])
            const avgRating = reviews.length > 0
              ? reviews.reduce((sum, r) => sum + Number(r.rating || 0), 0) / reviews.length
              : Number((details as any).rating ?? doctor.rating ?? 0)

            return {
              ...doctor,
              bio: details.bio ?? "",
              imageUrl: (details as any).imageUrl ?? (details as any).photoUrl ?? doctor.imageUrl,
              photoUrl: (details as any).photoUrl ?? null,
              rating: Number.isFinite(avgRating) ? Number(avgRating.toFixed(1)) : 0,
              reviewCount: reviews.length > 0 ? reviews.length : Number((details as any).reviewCount ?? doctor.reviewCount ?? 0),
            }
          } catch {
            return doctor
          }
        })
      )
      setDoctors(enriched)
    } catch (e: any) {
      console.error('Failed to fetch doctors:', e)
      setError("Failed to load doctors. Check your connection.")
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      fetchDoctors();
    }, [fetchDoctors])
  );

  useEffect(() => {
    let unsubCreated: (() => void) | undefined;
    let unsubUpdated: (() => void) | undefined;
    let mounted = true;

    const bindRealtime = async () => {
      try {
        // Wait for auth to settle
        await new Promise(r => setTimeout(r, 1500))

        const token = await AsyncStorage.getItem('token')
        if (!token || !mounted) return

        const conn = await startSignalRConnection();
        if (!conn || !mounted) return;

        unsubCreated = onDoctorCreated(() => {
          fetchDoctors().catch(() => undefined);
        });

        unsubUpdated = onDoctorUpdated(async (payload) => {
          const doctorId = payload?.doctorId;
          if (!doctorId) return;
          try {
            const details: any = await getDoctorById(doctorId);
            if (!mounted) return;

            setDoctors((prev) =>
              prev.map((d) =>
                d.id === doctorId
                  ? {
                      ...d,
                      name: details.name ?? details.fullName ?? d.name,
                      specialty: details.specialty ?? d.specialty,
                      consultationFee: details.consultationFee ?? details.consultFee ?? d.consultationFee,
                      imageUrl: details.imageUrl ?? details.photoUrl ?? d.imageUrl,
                      isAvailable: typeof details.isAvailable === "boolean" ? details.isAvailable : d.isAvailable,
                      bio: details.bio ?? "",
                      photoUrl: details.photoUrl ?? null,
                    }
                  : d
              )
            );

            setHighlightedDoctorId(String(doctorId));
            if (highlightTimeoutRef.current) {
              clearTimeout(highlightTimeoutRef.current);
            }
            highlightTimeoutRef.current = setTimeout(() => {
              setHighlightedDoctorId(null);
            }, 3500);
          } catch {
            // If targeted update fails, leave current list untouched.
          }
        });
      } catch {
        // realtime is optional; fallback to manual refresh
      }
    };

    bindRealtime();
    return () => {
      mounted = false;
      unsubCreated?.();
      unsubUpdated?.();
      if (highlightTimeoutRef.current) {
        clearTimeout(highlightTimeoutRef.current);
      }
    };
  }, [fetchDoctors]);

  useEffect(() => {
    let res = doctors;
    if (activeSpec !== "All")
      res = res.filter((d) => d.specialty?.toLowerCase() === activeSpec.toLowerCase());
    if (search.trim()) {
      const q = search.toLowerCase();
      res = res.filter((d) => d.name?.toLowerCase().includes(q) || d.specialty?.toLowerCase().includes(q));
    }
    setFiltered(res);
  }, [search, activeSpec, doctors]);

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color={COLORS.primary} /></View>;
  if (error) return (
    <View style={styles.center}>
      <Ionicons name="cloud-offline-outline" size={48} color="#e53935" />
      <Text style={styles.errorTxt}>{error}</Text>
      <TouchableOpacity style={styles.retryBtn} onPress={fetchDoctors}>
        <Text style={styles.retryTxt}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F4F6FA" />

      <View style={styles.header}>
        <View>
          <Text style={[styles.title, isRTL && styles.textRight]}>{tr("find_doctor")}</Text>
          <Text style={[styles.subtitle, isRTL && styles.textRight]}>{filtered.length} {tr("stats_doctors")}</Text>
        </View>
        <View style={styles.headerAvatar}>
          <Ionicons name="search" size={18} color={COLORS.primary} />
        </View>
      </View>

      <View style={[styles.searchBox, isRTL && styles.rowReverse]}>
        <Ionicons name="search-outline" size={16} color="#BBB" />
        <TextInput
          style={[styles.searchInput, isRTL && styles.textRight]}
          placeholder={tr("search")}
          placeholderTextColor="#BBB"
          value={search}
          onChangeText={setSearch}
          textAlign={isRTL ? "right" : "left"}
        />
        {search.length > 0 && (
          <TouchableOpacity onPress={() => setSearch("")}>
            <Ionicons name="close-circle" size={16} color="#CCC" />
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.chipsWrapper}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipsRow}>
          {SPECIALTIES.map((s) => (
            <TouchableOpacity
              key={s}
              style={[styles.chip, activeSpec === s && styles.chipActive]}
              onPress={() => setActiveSpec(s)}
            >
              <Text style={[styles.chipTxt, activeSpec === s && styles.chipTxtActive]}>{s}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <FlatList
        data={filtered}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => {
          const years = (item as any).yearsExperience ?? (item as any).experience ?? 0;
          const hasSchedule = ((item as any).hasSchedule ?? true) && ((item as any).isScheduleVisible ?? true);
          return (
            <DoctorCard
              doctor={{
                ...item,
                id: String(item.id),
                experience: `${years} yrs`,
                hasSchedule,
              }}
              highlight={String(item.id) === highlightedDoctorId}
            />
          );
        }}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => { setRefreshing(true); fetchDoctors(); }}
            colors={[COLORS.primary]}
          />
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="search-outline" size={48} color="#DDD" />
            <Text style={styles.emptyTxt}>{tr("no_doctors")}</Text>
            <Text style={styles.emptySubTxt}>{tr("try_different")}</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container:   { flex: 1, backgroundColor: "#F4F6FA", paddingTop: STATUS_BAR_HEIGHT },
  center:      { flex: 1, justifyContent: "center", alignItems: "center", gap: 12, backgroundColor: "#F4F6FA" },
  rowReverse:  { flexDirection: "row-reverse" },
  textRight:   { textAlign: "right" },
  errorTxt:    { color: "#e53935", fontSize: 14, textAlign: "center", paddingHorizontal: 32 },
  retryBtn:    { backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 9, borderRadius: 18 },
  retryTxt:    { color: "#fff", fontWeight: "600", fontSize: 13 },
  header: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 20, paddingTop: 12, paddingBottom: 14,
  },
  headerAvatar: {
    width: 42, height: 42, borderRadius: 21,
    backgroundColor: COLORS.primary + "15",
    justifyContent: "center", alignItems: "center",
    borderWidth: 1.5, borderColor: COLORS.primary + "30",
  },
  title:       { fontSize: 22, fontWeight: "800", color: "#1A1A1A", letterSpacing: -0.3 },
  subtitle:    { fontSize: 12, color: "#AAA", marginTop: 2 },
  searchBox: {
    flexDirection: "row", alignItems: "center", gap: 8,
    backgroundColor: "#fff", borderRadius: 14,
    marginHorizontal: 18, paddingHorizontal: 14, paddingVertical: 11,
    shadowColor: "#000", shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
  },
  searchInput:   { flex: 1, fontSize: 13, color: "#1A1A1A", padding: 0 },
  chipsWrapper:  { marginTop: 8, marginBottom: 4 },
  chipsRow:      { paddingHorizontal: 18, paddingVertical: 6, gap: 8, alignItems: "center" },
  chip:          { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#E8E8E8" },
  chipActive:    { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  chipTxt:       { fontSize: 13, color: "#777", fontWeight: "500" },
  chipTxtActive: { color: "#fff", fontWeight: "600" },
  listContent:   { paddingTop: 12, paddingBottom: 24 },
  empty:         { alignItems: "center", paddingTop: 60, gap: 8 },
  emptyTxt:      { fontSize: 15, fontWeight: "600", color: "#BBB" },
  emptySubTxt:   { fontSize: 12, color: "#CCC" },
});
