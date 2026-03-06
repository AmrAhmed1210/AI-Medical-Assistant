import React, { useState, useEffect, useCallback } from "react";
import {
  View, Text, FlatList, StyleSheet, ActivityIndicator,
  TextInput, TouchableOpacity, RefreshControl, ScrollView,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import DoctorCard from "@/components/DoctorCard";

const MOCK_DOCTORS = [
  { id: "1", name: "Dr. Sarah Ahmed",   specialty: "Cardiology",   rating: 4.8, reviewCount: 124, location: "Cairo, Egypt",      experience: "12 yrs", consultationFee: 50, isAvailable: true  },
  { id: "2", name: "Dr. Mohamed Ali",   specialty: "Dermatology",  rating: 4.6, reviewCount: 89,  location: "Giza, Egypt",       experience: "8 yrs",  consultationFee: 40, isAvailable: true  },
  { id: "3", name: "Dr. Nour Hassan",   specialty: "Pediatrics",   rating: 4.9, reviewCount: 210, location: "Alexandria, Egypt", experience: "15 yrs", consultationFee: 60, isAvailable: false },
  { id: "4", name: "Dr. Ahmed Karim",   specialty: "Neurology",    rating: 4.7, reviewCount: 67,  location: "Cairo, Egypt",      experience: "10 yrs", consultationFee: 70, isAvailable: true  },
  { id: "5", name: "Dr. Layla Mostafa", specialty: "General",      rating: 4.5, reviewCount: 145, location: "Cairo, Egypt",      experience: "6 yrs",  consultationFee: 30, isAvailable: true  },
  { id: "6", name: "Dr. Omar Farouk",   specialty: "Cardiology",   rating: 4.3, reviewCount: 55,  location: "Giza, Egypt",       experience: "9 yrs",  consultationFee: 55, isAvailable: false },
];

const SPECIALTIES = ["All", "General", "Cardiology", "Dermatology", "Pediatrics", "Neurology"];

export default function DoctorsScreen() {
  const params = useLocalSearchParams<{ specialty?: string }>();

  const [doctors,    setDoctors]    = useState<any[]>([]);
  const [filtered,   setFiltered]   = useState<any[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error,      setError]      = useState("");
  const [search,     setSearch]     = useState("");
  const [activeSpec, setActiveSpec] = useState("All");

  // لو جاي من الـ home بـ specialty معين، حدد الفلتر تلقائياً
  useEffect(() => {
    if (params.specialty) {
      const match = SPECIALTIES.find(
        (s) => s.toLowerCase() === params.specialty!.toLowerCase()
      );
      if (match) setActiveSpec(match);
    }
  }, [params.specialty]);

  const fetchDoctors = useCallback(async () => {
    try {
      setError("");
      await new Promise((r) => setTimeout(r, 400));
      setDoctors(MOCK_DOCTORS);
    } catch (e: any) {
      setError(e.message || "Failed to load");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { fetchDoctors(); }, [fetchDoctors]);

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
  if (error)   return (
    <View style={styles.center}>
      <Text style={styles.errorTxt}>⚠️ {error}</Text>
      <TouchableOpacity style={styles.retryBtn} onPress={fetchDoctors}>
        <Text style={styles.retryTxt}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Find Doctors</Text>
        <Text style={styles.subtitle}>{filtered.length} doctors available</Text>
      </View>

      <View style={styles.searchBox}>
        <Ionicons name="search-outline" size={16} color="#BBB" />
        <TextInput
          style={styles.searchInput}
          placeholder="Search by name or specialty..."
          placeholderTextColor="#BBB"
          value={search}
          onChangeText={setSearch}
        />
        {search.length > 0 && (
          <TouchableOpacity onPress={() => setSearch("")}>
            <Ionicons name="close-circle" size={16} color="#CCC" />
          </TouchableOpacity>
        )}
      </View>

      {/* Filter chips */}
      <View style={styles.chipsWrapper}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.chipsRow}
        >
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
        renderItem={({ item }) => <DoctorCard doctor={item} />}
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
            <Text style={styles.emptyTxt}>No doctors found</Text>
            <Text style={styles.emptySubTxt}>Try a different search or filter</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: "#F4F6FA" },
  center:     { flex: 1, justifyContent: "center", alignItems: "center", gap: 12, backgroundColor: "#F4F6FA" },
  errorTxt:   { color: "#e53935", fontSize: 14, textAlign: "center", paddingHorizontal: 32 },
  retryBtn:   { backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 9, borderRadius: 18 },
  retryTxt:   { color: "#fff", fontWeight: "600", fontSize: 13 },
  header:     { paddingHorizontal: 18, paddingTop: 52, paddingBottom: 10 },
  title:      { fontSize: 24, fontWeight: "800", color: "#1A1A1A", letterSpacing: -0.3 },
  subtitle:   { fontSize: 12, color: "#AAA", marginTop: 2 },
  searchBox:  {
    flexDirection: "row", alignItems: "center", gap: 8,
    backgroundColor: "#fff", borderRadius: 14,
    marginHorizontal: 18, paddingHorizontal: 14, paddingVertical: 11,
    shadowColor: "#000", shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
  },
  searchInput: { flex: 1, fontSize: 13, color: "#1A1A1A", padding: 0 },

  /* الـ wrapper بيمنع الـ chips من الاتقطع */
  chipsWrapper: { marginTop: 12, marginBottom: 0 },
  chipsRow: {
    paddingHorizontal: 18,
    paddingVertical: 6,
    gap: 8,
    alignItems: "center",
  },
  chip: {
    paddingHorizontal: 16, paddingVertical: 8,
    borderRadius: 20, backgroundColor: "#fff",
    borderWidth: 1.5, borderColor: "#E8E8E8",
  },
  chipActive:    { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  chipTxt:       { fontSize: 13, color: "#777", fontWeight: "500" },
  chipTxtActive: { color: "#fff", fontWeight: "600" },
  listContent:   { paddingTop: 12, paddingBottom: 24 },
  empty:         { alignItems: "center", paddingTop: 60, gap: 8 },
  emptyTxt:      { fontSize: 15, fontWeight: "600", color: "#BBB" },
  emptySubTxt:   { fontSize: 12, color: "#CCC" },
});