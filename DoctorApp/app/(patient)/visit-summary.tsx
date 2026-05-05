import React, { useEffect, useState } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { getVisitSummary, VisitSummary } from "../../services/visitService";
import Toast from "react-native-toast-message";

export default function VisitSummaryScreen() {
  const router = useRouter();
  const { visitId } = useLocalSearchParams<{ visitId: string }>();
  const [summary, setSummary] = useState<VisitSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!visitId) return;
    loadSummary();
  }, [visitId]);

  const loadSummary = async () => {
    try {
      setLoading(true);
      const data = await getVisitSummary(Number(visitId));
      setSummary(data);
    } catch (e: any) {
      Toast.show({ type: "error", text1: "Failed to load visit summary" });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  if (!summary) {
    return (
      <View style={styles.center}>
        <Text style={styles.emptyText}>Visit summary not found.</Text>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <Text style={styles.backBtnTxt}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Visit Summary</Text>
          <View style={{ width: 22 }} />
        </View>
        <View style={styles.headerInfo}>
          <Text style={styles.patientName}>{summary.patientName}</Text>
          <Text style={styles.visitDate}>{summary.visitDate}</Text>
        </View>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Chief Complaint */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Chief Complaint</Text>
          <Text style={styles.cardText}>{summary.chiefComplaint}</Text>
        </View>

        {/* Examination Findings */}
        {summary.examinationFindings && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Examination Findings</Text>
            <Text style={styles.cardText}>{summary.examinationFindings}</Text>
          </View>
        )}

        {/* Assessment */}
        {summary.assessment && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Assessment</Text>
            <Text style={styles.cardText}>{summary.assessment}</Text>
          </View>
        )}

        {/* Plan */}
        {summary.plan && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Plan</Text>
            <Text style={styles.cardText}>{summary.plan}</Text>
          </View>
        )}

        {/* Symptoms */}
        {summary.symptoms.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Symptoms</Text>
            {summary.symptoms.map((s, i) => (
              <View key={i} style={styles.rowItem}>
                <Text style={styles.rowName}>{s.name}</Text>
                <Text style={styles.rowMeta}>Severity: {s.severity} • {s.onset}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Prescriptions */}
        {summary.prescriptions.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Prescriptions</Text>
            {summary.prescriptions.map((p, i) => (
              <View key={i} style={styles.rowItem}>
                <Text style={styles.rowName}>{p.medicationName}</Text>
                <Text style={styles.rowMeta}>{p.dosage} • {p.frequency}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Vitals */}
        {summary.vitalSigns.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Vital Signs</Text>
            {summary.vitalSigns.map((v, i) => (
              <View key={i} style={styles.rowItem}>
                <Text style={styles.rowName}>{v.type}</Text>
                <Text style={[styles.rowMeta, v.isAbnormal && styles.abnormal]}>
                  {v.value}{v.value2 ? `/${v.value2}` : ""} {v.unit}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Allergies */}
        {summary.allergies.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Allergies</Text>
            {summary.allergies.map((a, i) => (
              <View key={i} style={styles.rowItem}>
                <Text style={styles.rowName}>{a.allergenName}</Text>
                <Text style={styles.rowMeta}>{a.severity} • {a.reaction}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Follow Up */}
        {summary.followUpRequired && (
          <View style={[styles.card, styles.followUpCard]}>
            <Text style={styles.cardTitle}>Follow Up Required</Text>
            <Text style={styles.cardText}>After {summary.followUpAfterDays} days</Text>
            {summary.followUpNotes && <Text style={styles.cardText}>{summary.followUpNotes}</Text>}
          </View>
        )}

        {/* Notes */}
        {summary.notes && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Additional Notes</Text>
            <Text style={styles.cardText}>{summary.notes}</Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  emptyText: { fontSize: 16, color: "#64748B", marginBottom: 20 },
  backBtn: { backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 12, borderRadius: 20 },
  backBtnTxt: { color: "#fff", fontWeight: "700" },
  header: { backgroundColor: COLORS.primary, paddingBottom: 24, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 20, paddingTop: 60, paddingBottom: 16 },
  headerTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  headerInfo: { paddingHorizontal: 20 },
  patientName: { fontSize: 22, fontWeight: "800", color: "#fff" },
  visitDate: { fontSize: 14, color: "rgba(255,255,255,0.8)", marginTop: 4 },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 40 },
  card: { backgroundColor: "#fff", borderRadius: 20, padding: 18, marginBottom: 14, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  followUpCard: { backgroundColor: "#FFF7ED", borderWidth: 1, borderColor: "#FDBA74" },
  cardTitle: { fontSize: 13, fontWeight: "700", color: "#64748B", marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.5 },
  cardText: { fontSize: 15, color: "#1E293B", lineHeight: 22 },
  rowItem: { paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  rowName: { fontSize: 15, fontWeight: "600", color: "#1E293B" },
  rowMeta: { fontSize: 12, color: "#64748B", marginTop: 2 },
  abnormal: { color: "#E11D48", fontWeight: "700" },
});
