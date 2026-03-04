import React, { useState } from "react";
import { 
  View, Text, StyleSheet, ScrollView, TouchableOpacity, 
  LayoutAnimation, Platform, UIManager 
} from "react-native";
import { Brain, ChevronDown, ChevronUp, FileText } from "lucide-react-native";
import { COLORS } from "../../constants/colors";

if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

const severityConfig = {
  low: { color: "#10B981", bg: "#DCFCE7", text: "#065F46" },
  medium: { color: "#F59E0B", bg: "#FEF3C7", text: "#92400E" },
  high: { color: "#F97316", bg: "#FFEDD5", text: "#9A3412" },
  critical: { color: "#EF4444", bg: "#FEE2E2", text: "#991B1B" },
};

const urgencyConfig = {
  routine: { bg: "#DCFCE7", text: "#065F46" },
  soon: { bg: "#FEF3C7", text: "#92400E" },
  urgent: { bg: "#FFEDD5", text: "#9A3412" },
  emergency: { bg: "#FEE2E2", text: "#991B1B" },
};

function ReportCard({ report }: { report: any }) {
  const [expanded, setExpanded] = useState(false);
  const sev = severityConfig[report.severity as keyof typeof severityConfig];

  const toggleExpand = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpanded(!expanded);
  };

  return (
    <View style={styles.reportCard}>
      <TouchableOpacity onPress={toggleExpand} style={styles.reportHeader} activeOpacity={0.7}>
        <View style={[styles.iconBox, { backgroundColor: sev.bg }]}>
          <Brain size={20} color={sev.color} />
        </View>
        <View style={styles.headerInfo}>
          <View style={styles.nameRow}>
            <Text style={styles.patientName}>{report.patientName}</Text>
            <View style={[styles.miniBadge, { backgroundColor: sev.bg }]}>
              <Text style={[styles.miniBadgeText, { color: sev.text }]}>{report.severity}</Text>
            </View>
          </View>
          <Text style={styles.diagnosisText}>{report.diagnosis}</Text>
          <Text style={styles.dateText}>{report.date}</Text>
        </View>
        <View style={styles.headerRight}>
          <Text style={styles.confidenceMain}>{Math.round(report.confidence * 100)}%</Text>
          {expanded ? <ChevronUp size={16} color="#999" /> : <ChevronDown size={16} color="#999" />}
        </View>
      </TouchableOpacity>

      {expanded && (
        <View style={styles.expandedContent}>
          <View style={styles.divider} />
          
          <View style={styles.badgeRow}>
            <View style={[styles.detailBadge, { backgroundColor: sev.bg }]}>
              <Text style={[styles.detailBadgeText, { color: sev.text }]}>Severity: {report.severity}</Text>
            </View>
            <View style={[styles.detailBadge, { backgroundColor: urgencyConfig[report.urgency as keyof typeof urgencyConfig].bg }]}>
              <Text style={[styles.detailBadgeText, { color: urgencyConfig[report.urgency as keyof typeof urgencyConfig].text }]}>
                Urgency: {report.urgency}
              </Text>
            </View>
          </View>

          <View style={styles.detailSection}>
            <Text style={styles.detailTitle}>Reported Symptoms</Text>
            <View style={styles.symptomsWrap}>
              {report.symptoms.map((s: string, i: number) => (
                <View key={i} style={styles.symptomTag}>
                  <Text style={styles.symptomText}>{s}</Text>
                </View>
              ))}
            </View>
          </View>

          <View style={styles.detailSection}>
            <Text style={styles.detailTitle}>AI Recommendations</Text>
            {report.recommendations.map((rec: string, i: number) => (
              <View key={i} style={styles.recRow}>
                <View style={styles.bullet} />
                <Text style={styles.recText}>{rec}</Text>
              </View>
            ))}
          </View>

          <View style={styles.detailSection}>
            <View style={styles.barLabelRow}>
              <Text style={styles.detailTitle}>AI Confidence</Text>
              <Text style={styles.confidenceValue}>{Math.round(report.confidence * 100)}%</Text>
            </View>
            <View style={styles.barBg}>
              <View style={[styles.barFill, { width: `${report.confidence * 100}%` }]} />
            </View>
          </View>

          <View style={styles.actionRow}>
            <TouchableOpacity style={styles.approveBtn}>
              <FileText size={14} color="#FFF" />
              <Text style={styles.approveBtnText}>Approve Report</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.editBtn}>
              <Text style={styles.editBtnText}>Edit Diagnosis</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
}

export default function DoctorAIReports() {
  const aiReports = [
    {
      id: "1",
      patientName: "Alice Johnson",
      severity: "critical",
      diagnosis: "Acute Coronary Syndrome",
      date: "Oct 24, 2024",
      confidence: 0.94,
      urgency: "emergency",
      symptoms: ["Chest pain", "Shortness of breath", "Nausea"],
      recommendations: ["Immediate ECG", "Administer Aspirin", "Refer to ICU"]
    },
    {
      id: "2",
      patientName: "David Miller",
      severity: "medium",
      diagnosis: "Minor Migraine",
      date: "Oct 23, 2024",
      confidence: 0.82,
      urgency: "routine",
      symptoms: ["Headache", "Light sensitivity"],
      recommendations: ["Rest in dark room", "Oral hydration"]
    }
  ];

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 50 }}>
      <View style={styles.pageHeader}>
        <Text style={styles.title}>AI Medical Reports</Text>
        <Text style={styles.subtitle}>AI-generated diagnosis reports for your patients</Text>
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.summaryScroll} contentContainerStyle={{ gap: 10 }}>
        <View style={[styles.summaryCard, { backgroundColor: "#E0F2F1" }]}>
          <Text style={[styles.summaryCount, { color: COLORS.primary }]}>{aiReports.length}</Text>
          <Text style={[styles.summaryLabel, { color: COLORS.primary }]}>Total</Text>
        </View>
        <View style={[styles.summaryCard, { backgroundColor: "#FEE2E2" }]}>
          <Text style={[styles.summaryCount, { color: "#EF4444" }]}>1</Text>
          <Text style={[styles.summaryLabel, { color: "#EF4444" }]}>Critical</Text>
        </View>
      </ScrollView>

      <View style={styles.listContainer}>
        {aiReports.map((report) => (
          <ReportCard key={report.id} report={report} />
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FBFBFB", paddingTop: 60 },
  pageHeader: { paddingHorizontal: 20, marginBottom: 20 },
  title: { fontSize: 22, fontWeight: "bold", color: "#1A1A1A" },
  subtitle: { fontSize: 13, color: "#888", marginTop: 4 },
  summaryScroll: { paddingLeft: 20, marginBottom: 25 },
  summaryCard: { minWidth: 80, padding: 15, borderRadius: 18, alignItems: "center" },
  summaryCount: { fontSize: 20, fontWeight: "bold" },
  summaryLabel: { fontSize: 10, fontWeight: "600", marginTop: 2 },
  listContainer: { paddingHorizontal: 20, gap: 12 },
  reportCard: { backgroundColor: "#FFF", borderRadius: 18, borderWidth: 1, borderColor: "#F0F0F0", overflow: "hidden" },
  reportHeader: { flexDirection: "row", padding: 15, alignItems: "center" },
  iconBox: { width: 45, height: 45, borderRadius: 14, justifyContent: "center", alignItems: "center" },
  headerInfo: { flex: 1, marginLeft: 12 },
  nameRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  patientName: { fontSize: 14, fontWeight: "bold", color: "#333" },
  miniBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 },
  miniBadgeText: { fontSize: 8, fontWeight: "bold", textTransform: "capitalize" },
  diagnosisText: { fontSize: 11, color: COLORS.primary, fontWeight: "600", marginTop: 2 },
  dateText: { fontSize: 10, color: "#AAA", marginTop: 2 },
  headerRight: { alignItems: "flex-end", gap: 4 },
  confidenceMain: { fontSize: 12, fontWeight: "bold", color: "#333" },
  expandedContent: { padding: 15, paddingTop: 0 },
  divider: { height: 1, backgroundColor: "#F5F5F5", marginBottom: 15 },
  badgeRow: { flexDirection: "row", gap: 8, marginBottom: 15 },
  detailBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  detailBadgeText: { fontSize: 10, fontWeight: "600" },
  detailSection: { marginBottom: 15 },
  detailTitle: { fontSize: 11, fontWeight: "bold", color: "#333", marginBottom: 8 },
  symptomsWrap: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  symptomTag: { backgroundColor: "#F5F5F5", paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  symptomText: { fontSize: 10, color: "#666" },
  recRow: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 5 },
  bullet: { width: 5, height: 5, borderRadius: 3, backgroundColor: COLORS.primary },
  recText: { fontSize: 10, color: "#777", flex: 1 },
  barLabelRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: 6 },
  confidenceValue: { fontSize: 11, color: COLORS.primary, fontWeight: "bold" },
  barBg: { height: 6, backgroundColor: "#F0F0F0", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", backgroundColor: COLORS.primary },
  actionRow: { flexDirection: "row", gap: 10, marginTop: 10 },
  approveBtn: { flex: 1.5, backgroundColor: COLORS.primary, flexDirection: "row", justifyContent: "center", alignItems: "center", paddingVertical: 12, borderRadius: 12, gap: 8 },
  approveBtnText: { color: "#FFF", fontSize: 11, fontWeight: "bold" },
  editBtn: { flex: 1, borderWidth: 1, borderColor: "#EEE", justifyContent: "center", alignItems: "center", borderRadius: 12 },
  editBtnText: { color: "#333", fontSize: 11, fontWeight: "600" },
});