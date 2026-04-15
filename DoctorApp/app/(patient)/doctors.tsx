import React, { useState, useEffect, useCallback } from "react";
import {
  View, Text, FlatList, StyleSheet, ActivityIndicator,
  TextInput, TouchableOpacity, RefreshControl, ScrollView,
  Platform, StatusBar,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import DoctorCard from "@/components/DoctorCard";
import { useLanguage } from "../../context/LanguageContext";
import { getAllDoctors, Doctor } from "../../services/doctorService";

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

  useEffect(() => {
    if (params.specialty) {
      const match = SPECIALTIES.find((s) => s.toLowerCase() === params.specialty!.toLowerCase());
      if (match) setActiveSpec(match);
    }
  }, [params.specialty]);

  const fetchDoctors = useCallback(async () => {
    try {
      setError("");
      const data = await getAllDoctors();
      setDoctors(data);
    } catch (e: any) {
      setError(e.message || "Failed to load doctors");
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
  if (error) return (
    <View style={styles.center}>
      <Text style={styles.errorTxt}>⚠️ {error}</Text>
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
        renderItem={({ item }) => <DoctorCard doctor={{
          ...item,
          id: String(item.id),
          experience: `${(item as any).experience ?? 0} yrs`,
        }} />}
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