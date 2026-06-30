import React, { useState, useCallback } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator,
} from "react-native";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../../constants/colors";
import { getMyPatientId } from "../../../services/authService";
import {
  getAllergies, getChronicDiseases, getMedications, getVitals, getSurgeries, getPatientDocuments,
} from "../../../services/medicalRecordService";
import Toast from "react-native-toast-message";
import { useLanguage } from "../../../context/LanguageContext";
import { useTheme } from "../../../context/ThemeContext";
import PatientBackgroundBubbles from "@/components/PatientBackgroundBubbles";

type CategoryId = "allergies" | "chronic" | "medications" | "vitals" | "surgeries" | "documents";

interface CategoryInfo {
  id: CategoryId;
  icon: any;
  titleKey: string;
  countKey: string;
  color: string;
  bgColor: string;
}

export default function MedicalRecordsHub() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
  const { isDark, colors } = useTheme();
  const [loading, setLoading] = useState(true);
  const [counts, setCounts] = useState<Record<CategoryId, number>>({
    allergies: 0, chronic: 0, medications: 0, vitals: 0, surgeries: 0, documents: 0,
  });

  useFocusEffect(
    useCallback(() => {
      loadCounts();
    }, [])
  );

  const loadCounts = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (pid <= 0) {
        Toast.show({ type: "error", text1: tr("patient_profile_not_found" as any) });
        return;
      }
      const [a, c, m, v, s, d] = await Promise.all([
        getAllergies(pid), getChronicDiseases(pid), getMedications(pid),
        getVitals(pid), getSurgeries(pid), getPatientDocuments(pid),
      ]);
      setCounts({
        allergies: a.length, chronic: c.length, medications: m.length,
        vitals: v.length, surgeries: s.length, documents: d.length,
      });
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || tr("failed_load_records") });
    } finally {
      setLoading(false);
    }
  };

  const categories: CategoryInfo[] = [
    { id: "allergies", icon: "warning-outline", titleKey: "allergies", countKey: "allergies_count", color: COLORS.primary, bgColor: "#E8F6F2" },
    { id: "chronic", icon: "fitness-outline", titleKey: "chronic_diseases", countKey: "chronic_count", color: COLORS.primary, bgColor: "#E8F6F2" },
    { id: "medications", icon: "medical-outline", titleKey: "medications", countKey: "medications_count", color: COLORS.primary, bgColor: "#E8F6F2" },
    { id: "vitals", icon: "pulse-outline", titleKey: "vitals", countKey: "vitals_count", color: COLORS.primary, bgColor: "#E8F6F2" },
    { id: "surgeries", icon: "cut-outline", titleKey: "surgeries", countKey: "surgeries_count", color: COLORS.primary, bgColor: "#E8F6F2" },
    { id: "documents", icon: "document-text-outline", titleKey: "scans_labs", countKey: "documents_count", color: COLORS.primary, bgColor: "#E8F6F2" },
  ];

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: "center", alignItems: "center" }]}>
        <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor={colors.background} />
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />
      <PatientBackgroundBubbles isDark={isDark} />
      <View style={styles.header}>
        <View style={[styles.headerTop, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={22} color="#fff" style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{tr("medical_records")}</Text>
          <View style={{ width: 22 }} />
        </View>
      </View>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        <Text style={[styles.sectionTitle, { color: colors.text, textAlign: isRTL ? "right" : "left" }]}>{tr("health_categories")}</Text>
        <View style={[styles.grid, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          {categories.map((cat) => (
            <TouchableOpacity
              key={cat.id}
              style={[
                styles.card, 
                { 
                  backgroundColor: colors.surface, 
                  borderColor: colors.border,
                  borderLeftWidth: isRTL ? 0 : 4,
                  borderRightWidth: isRTL ? 4 : 0,
                  borderLeftColor: isRTL ? "transparent" : cat.color,
                  borderRightColor: isRTL ? cat.color : "transparent",
                  alignItems: isRTL ? "flex-end" : "flex-start"
                }
              ]}
              onPress={() => router.push({ pathname: "/(patient)/medical-records/[category]", params: { category: cat.id } })}
              activeOpacity={0.8}
            >
              <View style={[styles.iconWrap, { backgroundColor: cat.bgColor }]}>
                <Ionicons name={cat.icon} size={22} color={cat.color} />
              </View>
              <View style={{ flex: 1, [isRTL ? "marginRight" : "marginLeft"]: 12, alignItems: isRTL ? "flex-end" : "flex-start", width: "100%" }}>
                <Text style={[styles.cardTitle, { color: colors.text, textAlign: isRTL ? "right" : "left" }]}>{tr(cat.titleKey as any)}</Text>
                <Text style={[styles.cardCount, { color: cat.color, textAlign: isRTL ? "right" : "left" }]}>
                  {counts[cat.id]} {tr("items")}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  header: { backgroundColor: COLORS.primary, paddingTop: 50, paddingBottom: 20, borderBottomLeftRadius: 20, borderBottomRightRadius: 20 },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 20 },
  headerTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 100 },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginBottom: 16 },
  grid: { flexDirection: "row", flexWrap: "wrap", justifyContent: "space-between" },
  card: {
    width: "48%",
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 14,
    marginBottom: 14,
    flexDirection: "column",
    borderLeftWidth: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  iconWrap: {
    width: 44,
    height: 44,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  cardTitle: { fontSize: 13, fontWeight: "700", color: "#1E293B", marginBottom: 4 },
  cardCount: { fontSize: 12, fontWeight: "600" },
});
