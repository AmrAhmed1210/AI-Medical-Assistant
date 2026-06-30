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
import { useLanguage } from "../../context/LanguageContext";

export default function VisitSummaryScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
  const { visitId } = useLocalSearchParams<{ visitId: string }>();
  const [summary, setSummary] = useState<VisitSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [reportLang, setReportLang] = useState<"en" | "ar">(isRTL ? "ar" : "en");

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
      Toast.show({ type: "error", text1: isRTL ? "فشل تحميل ملخص الزيارة" : "Failed to load visit summary" });
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
        <Text style={styles.emptyText}>{isRTL ? "لم يتم العثور على ملخص الزيارة." : "Visit summary not found."}</Text>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <Text style={styles.backBtnTxt}>{isRTL ? "رجوع" : "Go Back"}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Header */}
      <View style={styles.header}>
        <View style={[styles.headerTop, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
          <TouchableOpacity onPress={() => router.back()} style={{ transform: [{ scaleX: isRTL ? -1 : 1 }] }}>
            <Ionicons name="arrow-back" size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{isRTL ? "ملخص الزيارة" : "Visit Summary"}</Text>
          <View style={{ width: 22 }} />
        </View>
        <View style={[styles.headerInfo, { alignItems: isRTL ? 'flex-end' : 'flex-start' }]}>
          <Text style={styles.patientName}>{summary.patientName}</Text>
          <Text style={styles.visitDate}>{summary.visitDate}</Text>
        </View>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Past Visits Summary (Last 8 Months) */}
        {((summary.recentVisits?.length ?? 0) > 0 || summary.visitsTimelineSummaryEn || summary.visitsTimelineSummaryAr) && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <View style={[styles.cardHeaderRow, { flexDirection: isRTL ? 'row-reverse' : 'row', width: '100%' }]}>
              <Text style={styles.cardTitle}>
                {reportLang === "ar" ? "ملخص الزيارات (آخر 8 أشهر)" : "Past Visits (Last 8 Months)"}
              </Text>
              <View style={styles.langToggle}>
                <TouchableOpacity
                  onPress={() => setReportLang("en")}
                  style={[styles.langBtn, reportLang === "en" && styles.langBtnActive]}
                >
                  <Text style={[styles.langBtnText, reportLang === "en" && styles.langBtnTextActive]}>EN</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setReportLang("ar")}
                  style={[styles.langBtn, reportLang === "ar" && styles.langBtnActive]}
                >
                  <Text style={[styles.langBtnText, reportLang === "ar" && styles.langBtnTextActive]}>AR</Text>
                </TouchableOpacity>
              </View>
            </View>
            <Text style={[styles.cardText, { textAlign: reportLang === "ar" ? "right" : "left", width: '100%' }]}>
              {reportLang === "ar"
                ? (summary.visitsTimelineSummaryAr || "لا توجد زيارات مسجلة في آخر 8 أشهر.")
                : (summary.visitsTimelineSummaryEn || "No visits recorded in the last 8 months.")}
            </Text>
            {(summary.recentVisits || []).map((visit) => (
              <View key={visit.id} style={[styles.visitItem, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                <Text style={[styles.rowName, { textAlign: isRTL ? 'right' : 'left' }]}>{visit.visitDate} · {visit.chiefComplaint}</Text>
                {!!visit.doctorName && (
                  <Text style={[styles.rowMeta, { textAlign: isRTL ? 'right' : 'left' }]}>
                    {isRTL ? `د. ${visit.doctorName}` : `Dr. ${visit.doctorName}`}{visit.doctorSpecialty ? ` · ${visit.doctorSpecialty}` : ""}
                  </Text>
                )}
                {(reportLang === "ar" ? (visit.summaryAr || visit.summary) : (visit.summaryEn || visit.summary)) ? (
                  <Text style={[styles.rowMeta, { textAlign: reportLang === "ar" ? "right" : "left" }]}>
                    {reportLang === "ar" ? (visit.summaryAr || visit.summary) : (visit.summaryEn || visit.summary)}
                  </Text>
                ) : null}
              </View>
            ))}
          </View>
        )}

        {/* Chief Complaint */}
        <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
          <Text style={styles.cardTitle}>{tr("chief_complaint")}</Text>
          <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.chiefComplaint}</Text>
        </View>

        {/* Examination Findings */}
        {summary.examinationFindings && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "نتائج الفحص الطبي" : "Examination Findings"}</Text>
            <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.examinationFindings}</Text>
          </View>
        )}

        {/* Assessment */}
        {summary.assessment && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{tr("assessment")}</Text>
            <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.assessment}</Text>
          </View>
        )}

        {/* Plan */}
        {summary.plan && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{tr("plan")}</Text>
            <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.plan}</Text>
          </View>
        )}

        {/* Symptoms */}
        {summary.symptoms.length > 0 && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "الأعراض" : "Symptoms"}</Text>
            {summary.symptoms.map((s, i) => (
              <View key={i} style={[styles.rowItem, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                <Text style={styles.rowName}>{s.name}</Text>
                <Text style={[styles.rowMeta, { textAlign: isRTL ? 'right' : 'left' }]}>
                  {isRTL 
                    ? `الشدة: ${s.severity === "Mild" ? "خفيفة" : s.severity === "Moderate" ? "متوسطة" : s.severity === "Severe" ? "شديدة" : s.severity} • ${s.onset}` 
                    : `Severity: ${s.severity} • ${s.onset}`}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Prescriptions */}
        {summary.prescriptions.length > 0 && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "الوصفات الطبية" : "Prescriptions"}</Text>
            {summary.prescriptions.map((p, i) => (
              <View key={i} style={[styles.rowItem, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                <Text style={styles.rowName}>{p.medicationName}</Text>
                <Text style={[styles.rowMeta, { textAlign: isRTL ? 'right' : 'left' }]}>{p.dosage} • {p.frequency}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Vitals */}
        {summary.vitalSigns.length > 0 && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "العلامات الحيوية" : "Vital Signs"}</Text>
            {summary.vitalSigns.map((v, i) => (
              <View key={i} style={[styles.rowItem, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                <Text style={styles.rowName}>
                  {isRTL 
                    ? (v.type === "Blood Pressure" ? "ضغط الدم" : v.type === "Blood Sugar" ? "سكر الدم" : v.type === "Heart Rate" ? "نبض القلب" : v.type === "Temperature" ? "الحرارة" : v.type === "SpO2" ? "الأكسجين" : v.type === "Weight" ? "الوزن" : v.type) 
                    : v.type}
                </Text>
                <Text style={[styles.rowMeta, v.isAbnormal && styles.abnormal, { textAlign: isRTL ? 'right' : 'left' }]}>
                  {v.value}{v.value2 ? `/${v.value2}` : ""} {v.unit === "C" ? (isRTL ? "م°" : "C") : v.unit}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Allergies */}
        {summary.allergies.length > 0 && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "الحساسية" : "Allergies"}</Text>
            {summary.allergies.map((a, i) => (
              <View key={i} style={[styles.rowItem, { alignItems: isRTL ? 'flex-end' : 'flex-start', width: '100%' }]}>
                <Text style={styles.rowName}>{a.allergenName}</Text>
                <Text style={[styles.rowMeta, { textAlign: isRTL ? 'right' : 'left' }]}>
                  {isRTL 
                    ? `الشدة: ${a.severity === "Mild" ? "خفيفة" : a.severity === "Moderate" ? "متوسطة" : a.severity === "Severe" ? "شديدة" : a.severity} • ${a.reaction}` 
                    : `Severity: ${a.severity} • ${a.reaction}`}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Follow Up */}
        {summary.followUpRequired && (
          <View style={[styles.card, styles.followUpCard, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "زيارة متابعة مطلوبة" : "Follow Up Required"}</Text>
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <Ionicons name="calendar-outline" size={18} color="#9333EA" />
              <Text style={{ fontSize: 16, fontWeight: "800", color: "#1E293B" }}>
                {isRTL 
                  ? `${summary.followUpDate}${summary.followUpTime ? ` في تمام الساعة ${summary.followUpTime}` : ""}` 
                  : `${summary.followUpDate}${summary.followUpTime ? ` at ${summary.followUpTime}` : ""}`}
              </Text>
            </View>
            {summary.followUpNotes && <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.followUpNotes}</Text>}
          </View>
        )}

        {/* Notes */}
        {summary.notes && (
          <View style={[styles.card, { alignItems: isRTL ? 'flex-end' : 'stretch' }]}>
            <Text style={styles.cardTitle}>{isRTL ? "ملاحظات إضافية" : "Additional Notes"}</Text>
            <Text style={[styles.cardText, { textAlign: isRTL ? 'right' : 'left' }]}>{summary.notes}</Text>
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
  cardHeaderRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10, gap: 8 },
  langToggle: { flexDirection: "row", backgroundColor: "#F1F5F9", borderRadius: 10, padding: 2 },
  langBtn: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  langBtnActive: { backgroundColor: "#fff" },
  langBtnText: { fontSize: 11, fontWeight: "700", color: "#64748B" },
  langBtnTextActive: { color: COLORS.primary },
  visitItem: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: "#F1F5F9" },
  cardText: { fontSize: 15, color: "#1E293B", lineHeight: 22 },
  rowItem: { paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  rowName: { fontSize: 15, fontWeight: "600", color: "#1E293B" },
  rowMeta: { fontSize: 12, color: "#64748B", marginTop: 2 },
  abnormal: { color: "#E11D48", fontWeight: "700" },
});
