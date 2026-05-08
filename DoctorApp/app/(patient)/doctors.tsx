import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  View, Text, FlatList, StyleSheet, ActivityIndicator,
  TextInput, TouchableOpacity, RefreshControl, ScrollView,
  Platform, StatusBar, Animated
} from "react-native";
import { useLocalSearchParams, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { LinearGradient } from "expo-linear-gradient";
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

  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [filtered, setFiltered] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [activeSpec, setActiveSpec] = useState("All");
  const [highlightedDoctorId, setHighlightedDoctorId] = useState<string | null>(null);
  const highlightTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollY = useRef(new Animated.Value(0)).current;

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
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* ANIMATED HEADER: Luxury Emerald */}
      <Animated.View style={[styles.magicHeader, { 
        height: scrollY.interpolate({
          inputRange: [0, 100],
          outputRange: [220, 120],
          extrapolate: 'clamp',
        })
      }]}>
        <LinearGradient
          colors={["#064E3B", "#059669"]}
          style={StyleSheet.absoluteFill}
        >
          <Animated.View style={[styles.headerTop, {
            opacity: scrollY.interpolate({
              inputRange: [0, 80],
              outputRange: [1, 0],
              extrapolate: 'clamp',
            })
          }]}>
            <View>
              <Text style={styles.headerTitle}>{tr("find_doctor")}</Text>
              <View style={styles.statBadge}>
                <Text style={styles.statBadgeText}>{filtered.length} {tr("stats_doctors")}</Text>
              </View>
            </View>
            <TouchableOpacity style={styles.headerActionBtn} onPress={() => fetchDoctors()}>
              <Ionicons name="refresh" size={20} color="#fff" />
            </TouchableOpacity>
          </Animated.View>

          {/* SEARCH BAR GLASS */}
          <Animated.View style={[styles.searchContainer, {
            opacity: scrollY.interpolate({
              inputRange: [0, 100],
              outputRange: [1, 0],
              extrapolate: 'clamp',
            }),
            transform: [{
              translateY: scrollY.interpolate({
                inputRange: [0, 100],
                outputRange: [0, -20],
                extrapolate: 'clamp',
              })
            }]
          }]}>
            <View style={styles.searchGlass}>
              <Ionicons name="search" size={20} color="rgba(255,255,255,0.7)" />
              <TextInput
                style={styles.searchInput}
                placeholder={tr("search")}
                placeholderTextColor="rgba(255,255,255,0.5)"
                value={search}
                onChangeText={setSearch}
                textAlign={isRTL ? "right" : "left"}
              />
              {search.length > 0 && (
                <TouchableOpacity onPress={() => setSearch("")}>
                  <Ionicons name="close-circle" size={20} color="rgba(255,255,255,0.5)" />
                </TouchableOpacity>
              )}
            </View>
          </Animated.View>

          {/* DECOR */}
          <View style={[styles.liquidBlob, { top: -30, right: -30, width: 180, height: 180, backgroundColor: '#10B981', opacity: 0.1 }]} />
          <View style={[styles.liquidBlob, { bottom: -20, left: -20, width: 140, height: 140, backgroundColor: '#34D399', opacity: 0.1 }]} />
        </LinearGradient>
      </Animated.View>

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

      <Animated.FlatList
        data={filtered}
        keyExtractor={(item) => String(item.id)}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
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
              compact={true}
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
  container: { flex: 1, backgroundColor: "#fff" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", gap: 12, backgroundColor: "#fff" },
  magicHeader: { height: 220, borderBottomLeftRadius: 40, borderBottomRightRadius: 40, overflow: 'hidden', elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.2, shadowRadius: 20 },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingTop: 60, zIndex: 10 },
  headerTitle: { fontSize: 20, fontWeight: '900', color: '#fff', letterSpacing: -0.5 },
  headerActionBtn: { width: 40, height: 40, borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  statBadge: { backgroundColor: 'rgba(255,255,255,0.15)', alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 10, marginTop: 4 },
  statBadgeText: { color: '#fff', fontSize: 11, fontWeight: '700' },
  searchContainer: { paddingHorizontal: 20, marginTop: 25, zIndex: 10 },
  searchGlass: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.15)', height: 52, borderRadius: 18, paddingHorizontal: 16, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  searchInput: { flex: 1, color: '#fff', fontSize: 15, marginLeft: 10, fontWeight: '600' },
  liquidBlob: { position: 'absolute', borderRadius: 100 },
  chipsWrapper: { marginTop: 15, marginBottom: 5 },
  chipsRow: { paddingHorizontal: 20, gap: 12 },
  chip: { paddingHorizontal: 20, paddingVertical: 10, borderRadius: 18, backgroundColor: '#F8FAFC', borderWidth: 1, borderColor: '#F1F5F9' },
  chipActive: { backgroundColor: '#064E3B', borderColor: '#064E3B', elevation: 5, shadowColor: '#064E3B', shadowOpacity: 0.3, shadowRadius: 10 },
  chipTxt: { fontSize: 12, color: '#64748B', fontWeight: '800' },
  chipTxtActive: { color: '#fff' },
  listContent: { paddingHorizontal: 15, paddingTop: 10, paddingBottom: 100 },
  empty: { alignItems: 'center', paddingTop: 80, gap: 15 },
  emptyTxt: { fontSize: 16, fontWeight: '900', color: '#1E293B' },
  emptySubTxt: { fontSize: 14, color: '#94A3B8', textAlign: 'center' },
  errorTxt: { color: '#EF4444', fontSize: 14, textAlign: 'center', paddingHorizontal: 32 },
  retryBtn: { backgroundColor: '#059669', paddingHorizontal: 24, paddingVertical: 12, borderRadius: 20, marginTop: 15 },
  retryTxt: { color: '#fff', fontWeight: '900', fontSize: 14 },
});
