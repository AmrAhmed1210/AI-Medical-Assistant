import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { COLORS } from "../../constants/colors";
import {
  closeVisit,
  getPatientHistory,
  getVisitById,
  updateVisit,
  type PatientHistory,
} from "../../services/visitService";
import { getMedicationSchedule, type MedicationScheduleItem } from "../../services/medicationService";
import { useLanguage } from "../../context/LanguageContext";

const normalRanges = {
  bpSystolic: { min: 90, max: 120, label: "90-120 mmHg" },
  bpDiastolic: { min: 60, max: 80, label: "60-80 mmHg" },
  bloodSugar: { min: 70, max: 100, label: "70-100 mg/dL" },
  heartRate: { min: 60, max: 100, label: "60-100 bpm" },
  temperature: { min: 36.1, max: 37.2, label: "36.1-37.2 C" },
  spo2: { min: 95, max: 100, label: "95-100%" },
};

function parseNum(v: string): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

export default function DoctorWorkspaceScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
  const params = useLocalSearchParams<{ visitId?: string; patientId?: string }>();
  const visitId = Number(params.visitId ?? 0);
  const patientIdFromParam = Number(params.patientId ?? 0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [history, setHistory] = useState<PatientHistory | null>(null);
  const [schedule, setSchedule] = useState<MedicationScheduleItem[]>([]);
  const [isClosed, setIsClosed] = useState(false);
  const [visitLang, setVisitLang] = useState<"en" | "ar">("en");

  const [chiefComplaint, setChiefComplaint] = useState("");
  const [hpi, setHpi] = useState("");
  const [exam, setExam] = useState("");
  const [assessment, setAssessment] = useState("");
  const [plan, setPlan] = useState("");
  const [notes, setNotes] = useState("");
  const [followUpRequired, setFollowUpRequired] = useState(false);
  const [followUpDay, setFollowUpDay] = useState("");
  const [followUpTime, setFollowUpTime] = useState("");
  const [followUpNotes, setFollowUpNotes] = useState("");
  const DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const DAYS_OF_WEEK_AR = ["الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"];

  const calculateNextDateForDay = (dayName: string) => {
    if (!dayName) return "";
    let dayIndex = DAYS_OF_WEEK.indexOf(dayName);
    if (dayIndex === -1) {
      dayIndex = DAYS_OF_WEEK_AR.indexOf(dayName);
    }
    if (dayIndex === -1) return "";
    const today = new Date();
    let diff = dayIndex - today.getDay();
    if (diff <= 0) diff += 7; // Always next occurrence
    const nextDate = new Date(today);
    nextDate.setDate(today.getDate() + diff);
    return nextDate.toISOString().split('T')[0];
  };

  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [sugar, setSugar] = useState("");
  const [hr, setHr] = useState("");
  const [temp, setTemp] = useState("");
  const [spo2, setSpo2] = useState("");

  useEffect(() => {
    if (!visitId) return;
    void loadAll();
  }, [visitId]);

  const loadAll = async () => {
    try {
      setLoading(true);
      const visit = await getVisitById(visitId);
      const pid = visit.patientId || patientIdFromParam;
      const [patientHistory, medsSchedule] = await Promise.all([
        pid ? getPatientHistory(pid).catch(() => null) : Promise.resolve(null),
        pid ? getMedicationSchedule(pid).catch(() => []) : Promise.resolve([]),
      ]);
      setHistory(patientHistory);
      setSchedule(medsSchedule);
      setChiefComplaint(visit.chiefComplaint || "");
      setHpi(visit.presentIllnessHistory || "");
      setFollowUpRequired(visit.followUpRequired || false);
      if (visit.followUpDate) {
         // Try to map back to day of week, or leave followUpDay empty if it's already a past date
         const d = new Date(visit.followUpDate);
         if (!isNaN(d.getTime())) {
           setFollowUpDay(isRTL ? DAYS_OF_WEEK_AR[d.getDay()] : DAYS_OF_WEEK[d.getDay()]);
         }
      }
      setFollowUpTime(visit.followUpTime || "");
      setFollowUpNotes(visit.followUpNotes || "");
      setExam(visit.examinationFindings || "");
      setAssessment(visit.assessment || "");
      setPlan(visit.plan || "");
      setNotes(visit.notes || "");
      setIsClosed(String(visit.status || "").toLowerCase() === "closed");
    } catch (e: any) {
      Toast.show({ type: "error", text1: e?.message || tr("error") });
    } finally {
      setLoading(false);
    }
  };

  const criticalAlert = useMemo(() => {
    const s = parseNum(bpSys);
    const d = parseNum(bpDia);
    const glu = parseNum(sugar);
    if ((s != null && s > 180) || (d != null && d > 120) || (glu != null && glu > 200)) {
      return isRTL ? "قيمة حرجة - تتطلب اتخاذ إجراء فوري" : "CRITICAL VALUE - Requires immediate attention";
    }
    return "";
  }, [bpSys, bpDia, sugar, isRTL]);

  const vitalWarnings = useMemo(
    () => ({
      bpSys: (() => {
        const n = parseNum(bpSys);
        if (n == null) return "";
        return n < normalRanges.bpSystolic.min || n > normalRanges.bpSystolic.max ? normalRanges.bpSystolic.label : "";
      })(),
      bpDia: (() => {
        const n = parseNum(bpDia);
        if (n == null) return "";
        return n < normalRanges.bpDiastolic.min || n > normalRanges.bpDiastolic.max ? normalRanges.bpDiastolic.label : "";
      })(),
      sugar: (() => {
        const n = parseNum(sugar);
        if (n == null) return "";
        return n < normalRanges.bloodSugar.min || n > normalRanges.bloodSugar.max ? normalRanges.bloodSugar.label : "";
      })(),
      hr: (() => {
        const n = parseNum(hr);
        if (n == null) return "";
        return n < normalRanges.heartRate.min || n > normalRanges.heartRate.max ? normalRanges.heartRate.label : "";
      })(),
      temp: (() => {
        const n = parseNum(temp);
        if (n == null) return "";
        return n < normalRanges.temperature.min || n > normalRanges.temperature.max ? normalRanges.temperature.label : "";
      })(),
      spo2: (() => {
        const n = parseNum(spo2);
        if (n == null) return "";
        return n < normalRanges.spo2.min || n > normalRanges.spo2.max ? normalRanges.spo2.label : "";
      })(),
    }),
    [bpSys, bpDia, sugar, hr, temp, spo2]
  );

  const buildVitalPayload = () => {
    const result = [];
    const s = parseNum(bpSys);
    const d = parseNum(bpDia);
    if (s != null) {
      result.push({
        type: "Blood Pressure",
        value: s,
        value2: d,
        unit: "mmHg",
        isAbnormal: !!vitalWarnings.bpSys || !!vitalWarnings.bpDia,
      });
    }
    const glu = parseNum(sugar);
    if (glu != null) result.push({ type: "Blood Sugar", value: glu, unit: "mg/dL", isAbnormal: !!vitalWarnings.sugar });
    const pulse = parseNum(hr);
    if (pulse != null) result.push({ type: "Heart Rate", value: pulse, unit: "bpm", isAbnormal: !!vitalWarnings.hr });
    const t = parseNum(temp);
    if (t != null) result.push({ type: "Temperature", value: t, unit: "C", isAbnormal: !!vitalWarnings.temp });
    const oxygen = parseNum(spo2);
    if (oxygen != null) result.push({ type: "SpO2", value: oxygen, unit: "%", isAbnormal: !!vitalWarnings.spo2 });
    return result;
  };

  const onSaveDraft = async () => {
    if (!visitId || isClosed) return;
    try {
      setSaving(true);
      await updateVisit(visitId, {
        chiefComplaint,
        presentIllnessHistory: hpi,
        examinationFindings: exam,
        assessment,
        plan,
        notes,
        followUpRequired,
        followUpDate: followUpRequired ? calculateNextDateForDay(followUpDay) : undefined,
        followUpTime: followUpTime || undefined,
        followUpNotes,
        vitalSigns: buildVitalPayload(),
      });
      Toast.show({ type: "success", text1: isRTL ? "تم حفظ المسودة بنجاح" : "Draft saved" });
    } catch (e: any) {
      Toast.show({ type: "error", text1: e?.message || tr("error") });
    } finally {
      setSaving(false);
    }
  };

  const onCloseVisit = () => {
    if (!visitId || isClosed) return;
    Alert.alert(
      isRTL ? "إنهاء الزيارة" : "Close Visit",
      isRTL ? "بمجرد إغلاق الزيارة، لا يمكن تعديلها. هل تؤكد الإغلاق؟" : "Once closed, this visit cannot be edited. Confirm?",
      [
        { text: tr("cancel"), style: "cancel" },
        {
          text: tr("confirm"),
          style: "destructive",
          onPress: async () => {
            try {
              setSaving(true);
              await onSaveDraft();
              await closeVisit(visitId);
              setIsClosed(true);
              router.push({ pathname: "/(doctor)/visit-summary", params: { visitId: String(visitId) } });
            } catch (e: any) {
              Toast.show({ type: "error", text1: e?.message || tr("error") });
            } finally {
              setSaving(false);
            }
          },
        },
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={[styles.pageTitle, { textAlign: isRTL ? "right" : "left" }]}>{tr("clinical_workspace")}</Text>
      <Text style={[styles.pageSubtitle, { textAlign: isRTL ? "right" : "left" }]}>{tr("workspace_desc")}</Text>
      {isClosed && <Text style={[styles.closedHint, { textAlign: isRTL ? "right" : "left" }]}>{tr("visit_read_only")}</Text>}
      {!!criticalAlert && <Text style={[styles.criticalBanner, { textAlign: isRTL ? "right" : "left" }]}>{criticalAlert}</Text>}

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <Text style={styles.stepTitle}>{tr("step_1_overview")}</Text>
        <Text style={styles.meta}>{isRTL ? "فصيلة الدم: " : "Blood Type: "}{history?.bloodType || "-"}</Text>
        <Text style={styles.meta}>{isRTL ? "الحساسية" : "Allergies"}</Text>
        <View style={[styles.tagWrap, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          {(history?.allergies || []).slice(0, 4).map((a, idx) => (
            <View key={`${a.allergenName}_${idx}`} style={styles.tagItem}>
              <Text style={styles.tagText}>{a.allergenName}</Text>
            </View>
          ))}
          {(history?.allergies || []).length === 0 && <Text style={styles.metaMuted}>{tr("no_allergies_known")}</Text>}
        </View>
        <Text style={[styles.meta, { marginTop: 8 }]}>{isRTL ? "الأدوية النشطة" : "Active Medications"}</Text>
        <View style={[styles.tagWrap, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          {(history?.medications || []).slice(0, 4).map((m, idx) => (
            <View key={`${m.medicationName}_${idx}`} style={styles.tagItem}>
              <Text style={styles.tagText}>{m.medicationName}</Text>
            </View>
          ))}
          {(history?.medications || []).length === 0 && <Text style={styles.metaMuted}>{tr("no_meds_active")}</Text>}
        </View>
      </View>

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <View style={[styles.sectionHeaderRow, { flexDirection: isRTL ? "row-reverse" : "row", width: "100%" }]}>
          <Text style={styles.sectionTitle}>{tr("past_visits_8m")}</Text>
          <View style={styles.langToggle}>
            <TouchableOpacity onPress={() => setVisitLang("en")} style={[styles.langBtn, visitLang === "en" && styles.langBtnActive]}>
              <Text style={[styles.langBtnText, visitLang === "en" && styles.langBtnTextActive]}>EN</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setVisitLang("ar")} style={[styles.langBtn, visitLang === "ar" && styles.langBtnActive]}>
              <Text style={[styles.langBtnText, visitLang === "ar" && styles.langBtnTextActive]}>AR</Text>
            </TouchableOpacity>
          </View>
        </View>
        {(history?.lastVisits || []).length === 0 ? (
          <Text style={[styles.metaMuted, { textAlign: isRTL ? "right" : "left" }]}>
            {visitLang === "ar" ? "لا توجد زيارات في آخر 8 أشهر." : "No previous visits found in the last 8 months."}
          </Text>
        ) : (
          (history?.lastVisits || []).map((v: any, idx: number) => (
            <View key={v.id || idx} style={{ marginBottom: 12, paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: '#F1F5F9', width: "100%" }}>
              <View style={{ flexDirection: isRTL ? 'row-reverse' : 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <Text style={{ fontWeight: '700', fontSize: 14, color: '#1E293B' }}>{v.visitDate}</Text>
                <Text style={{ fontSize: 12, color: '#059669', fontWeight: '600' }}>
                  {v.doctorName ? (isRTL ? `د. ${v.doctorName}` : `Dr. ${v.doctorName}`) : (isRTL ? "غير معروف" : "Unknown")}
                </Text>
              </View>
              {v.doctorSpecialty && (
                <Text style={{ fontSize: 11, color: '#64748B', marginBottom: 6, textAlign: isRTL ? 'right' : 'left' }}>{v.doctorSpecialty}</Text>
              )}
              <Text style={{ fontSize: 13, color: '#475569', marginBottom: 4, textAlign: isRTL ? 'right' : 'left' }}>
                <Text style={{ fontWeight: '600' }}>{visitLang === "ar" ? "الشكوى: " : "Complaint: "}</Text>{v.chiefComplaint}
              </Text>
              {(visitLang === "ar" ? (v.summaryAr || v.summary) : (v.summaryEn || v.summary)) && (
                <Text style={{ fontSize: 13, color: '#475569', textAlign: visitLang === "ar" ? "right" : "left" }}>
                  <Text style={{ fontWeight: '600' }}>{visitLang === "ar" ? "الملخص: " : "Summary: "}</Text>
                  {visitLang === "ar" ? (v.summaryAr || v.summary) : (v.summaryEn || v.summary)}
                </Text>
              )}
            </View>
          ))
        )}
      </View>

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <Text style={styles.sectionTitle}>{isRTL ? "أدوية اليوم" : "Today Medications"}</Text>
        {schedule.length === 0 ? (
          <Text style={styles.metaMuted}>{isRTL ? "لا توجد أدوية مجدولة لهذا المريض اليوم." : "No schedule for this patient today."}</Text>
        ) : (
          <View style={[styles.tagWrap, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
            {schedule.slice(0, 8).map((item, idx) => (
              <View key={`${item.medicationTrackerId}_${idx}`} style={styles.scheduleTag}>
                <Text style={styles.scheduleTagText}>
                  {new Date(item.scheduledAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} - {item.medicationName}
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <Text style={styles.stepTitle}>{tr("step_2_notes")}</Text>
        
        <Text style={styles.fieldLabel}>{tr("chief_complaint")}</Text>
        <TextInput editable={!isClosed} value={chiefComplaint} onChangeText={setChiefComplaint} placeholder={isRTL ? "مثال: صداع لمدة يومين" : "Example: headache for 2 days"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} />
        
        <Text style={styles.fieldLabel}>{isRTL ? "تاريخ المرض الحالي (HPI)" : "HPI"}</Text>
        <TextInput editable={!isClosed} value={hpi} onChangeText={setHpi} placeholder={isRTL ? "تاريخ مختصر للأعراض" : "Brief history of symptoms"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} multiline />
        
        <Text style={styles.fieldLabel}>{isRTL ? "الفحص السريري" : "Examination"}</Text>
        <TextInput editable={!isClosed} value={exam} onChangeText={setExam} placeholder={isRTL ? "نتائج الفحص البدني والسريري" : "What you found on exam"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} multiline />
        
        <Text style={styles.fieldLabel}>{tr("assessment")}</Text>
        <TextInput editable={!isClosed} value={assessment} onChangeText={setAssessment} placeholder={isRTL ? "التشخيص المحتمل" : "Likely diagnosis"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} multiline />
        
        <Text style={styles.fieldLabel}>{tr("plan")}</Text>
        <TextInput editable={!isClosed} value={plan} onChangeText={setPlan} placeholder={isRTL ? "العلاج والخطوات التالية" : "Treatment and next steps"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} multiline />
        
        <Text style={styles.fieldLabel}>{isRTL ? "ملاحظات إضافية" : "Additional Notes"}</Text>
        <TextInput editable={!isClosed} value={notes} onChangeText={setNotes} placeholder={isRTL ? "ملاحظة اختيارية للسجل الطبي" : "Optional note for the record"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} multiline />
      </View>

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <Text style={styles.sectionTitle}>{isRTL ? "العلامات الحيوية السريعة" : "Quick Vitals"}</Text>
        <Text style={[styles.metaMuted, { textAlign: isRTL ? "right" : "left" }]}>{isRTL ? "أدخل القراءات المتاحة فقط. سيتم تمييز القيم غير الطبيعية تلقائياً." : "Enter only available readings. Abnormal values are highlighted automatically."}</Text>
        <View style={[styles.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TextInput editable={!isClosed} value={bpSys} onChangeText={setBpSys} placeholder={isRTL ? "الضغط الانقباضي" : "BP Sys"} style={[styles.input, styles.smallInput, vitalWarnings.bpSys && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
          <TextInput editable={!isClosed} value={bpDia} onChangeText={setBpDia} placeholder={isRTL ? "الضغط الانبساطي" : "BP Dia"} style={[styles.input, styles.smallInput, vitalWarnings.bpDia && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
        </View>
        <View style={[styles.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TextInput editable={!isClosed} value={sugar} onChangeText={setSugar} placeholder={isRTL ? "السكر" : "Sugar"} style={[styles.input, styles.smallInput, vitalWarnings.sugar && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
          <TextInput editable={!isClosed} value={hr} onChangeText={setHr} placeholder={isRTL ? "نبض القلب" : "HR"} style={[styles.input, styles.smallInput, vitalWarnings.hr && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
        </View>
        <View style={[styles.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TextInput editable={!isClosed} value={temp} onChangeText={setTemp} placeholder={isRTL ? "الحرارة" : "Temp"} style={[styles.input, styles.smallInput, vitalWarnings.temp && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
          <TextInput editable={!isClosed} value={spo2} onChangeText={setSpo2} placeholder={isRTL ? "الأكسجين" : "SpO2"} style={[styles.input, styles.smallInput, vitalWarnings.spo2 && styles.inputWarn, { textAlign: isRTL ? "right" : "left" }]} />
        </View>
      </View>

      <View style={[styles.card, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
        <Text style={styles.stepTitle}>{isRTL ? "الخطوة 3 - المتابعة" : "Step 3 - Follow-up"}</Text>
        <View style={[styles.switchRow, { flexDirection: isRTL ? "row-reverse" : "row", width: "100%" }]}>
          <Text style={styles.meta}>{isRTL ? "زيارة متابعة مطلوبة" : "Follow-up required"}</Text>
          <Switch disabled={isClosed} value={followUpRequired} onValueChange={setFollowUpRequired} />
        </View>
        {followUpRequired && (
          <>
            <Text style={[styles.meta, { marginTop: 10, marginBottom: 5 }]}>{isRTL ? "اختر اليوم" : "Select Day"}</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 10 }} contentContainerStyle={{ flexDirection: isRTL ? "row-reverse" : "row" }}>
              {(isRTL ? DAYS_OF_WEEK_AR : DAYS_OF_WEEK).map(d => (
                <TouchableOpacity
                  key={d}
                  disabled={isClosed}
                  onPress={() => setFollowUpDay(d)}
                  style={{
                    paddingHorizontal: 16, paddingVertical: 8,
                    backgroundColor: followUpDay === d ? '#10B981' : '#E2E8F0',
                    borderRadius: 20, marginRight: isRTL ? 0 : 8, marginLeft: isRTL ? 8 : 0
                  }}
                >
                  <Text style={{ color: followUpDay === d ? '#fff' : '#475569', fontWeight: '600' }}>{isRTL ? d : d.substring(0,3)}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            {followUpDay ? (
              <Text style={{ color: '#059669', fontSize: 13, marginBottom: 10, fontWeight: '600', textAlign: isRTL ? "right" : "left" }}>
                {isRTL ? `سيتم الجدولة بتاريخ: ${calculateNextDateForDay(followUpDay)}` : `Will schedule for: ${calculateNextDateForDay(followUpDay)}`}
              </Text>
            ) : null}
            <TextInput editable={!isClosed} value={followUpTime} onChangeText={setFollowUpTime} placeholder={isRTL ? "الوقت (ساعة:دقيقة)" : "Time (HH:MM)"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} />
            <TextInput editable={!isClosed} value={followUpNotes} onChangeText={setFollowUpNotes} placeholder={isRTL ? "تعليمات للمريض" : "Notes for patient"} style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]} />
          </>
        )}
      </View>

      {!isClosed && (
        <View style={[styles.footerBtns, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TouchableOpacity style={styles.saveBtn} onPress={onSaveDraft} disabled={saving}>
            {saving ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnTxt}>{isRTL ? "حفظ كمسودة" : "Save Draft"}</Text>}
          </TouchableOpacity>
          <TouchableOpacity style={styles.closeBtn} onPress={onCloseVisit} disabled={saving}>
            <Text style={styles.btnTxt}>{isRTL ? "إنهاء الزيارة" : "Finish Visit"}</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F3F6FB" },
  content: { padding: 16, paddingBottom: 42 },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  pageTitle: { fontSize: 24, fontWeight: "800", color: "#1A237E", marginBottom: 4 },
  pageSubtitle: { fontSize: 13, color: "#64748B", marginBottom: 10 },
  closedHint: { color: "#C62828", fontWeight: "700", marginBottom: 8 },
  criticalBanner: { backgroundColor: "#C62828", color: "#fff", padding: 12, borderRadius: 12, marginBottom: 12, fontWeight: "700" },
  card: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    shadowColor: "#0F172A",
    shadowOpacity: 0.05,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1C1C1E", marginBottom: 8 },
  sectionHeaderRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8 },
  langToggle: { flexDirection: "row", backgroundColor: "#F1F5F9", borderRadius: 10, padding: 2 },
  langBtn: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  langBtnActive: { backgroundColor: "#fff" },
  langBtnText: { fontSize: 11, fontWeight: "700", color: "#64748B" },
  langBtnTextActive: { color: COLORS.primary },
  stepTitle: { fontSize: 16, fontWeight: "800", color: "#1A237E", marginBottom: 8 },
  meta: { fontSize: 13, color: "#6B7280", marginBottom: 4 },
  metaMuted: { fontSize: 12, color: "#94A3B8", marginBottom: 6 },
  fieldLabel: { fontSize: 12, color: "#64748B", fontWeight: "600", marginBottom: 4 },
  tagWrap: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  tagItem: {
    backgroundColor: "#EEF2FF",
    borderColor: "#C7D2FE",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  tagText: { fontSize: 12, color: "#3730A3", fontWeight: "600" },
  scheduleTag: {
    backgroundColor: "#E0F2F1",
    borderColor: "#99F6E4",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  scheduleTagText: { fontSize: 12, color: "#065F46", fontWeight: "600" },
  input: {
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 11,
    marginBottom: 8,
    backgroundColor: "#F8FAFC",
  },
  inputWarn: { borderColor: "#C62828", backgroundColor: "#FFEBEE" },
  row: { flexDirection: "row", gap: 8 },
  smallInput: { flex: 1 },
  switchRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  footerBtns: { flexDirection: "row", gap: 10, marginTop: 4 },
  saveBtn: { flex: 1, backgroundColor: "#1A237E", borderRadius: 12, paddingVertical: 13, alignItems: "center" },
  closeBtn: { flex: 1, backgroundColor: "#00695C", borderRadius: 12, paddingVertical: 13, alignItems: "center" },
  btnTxt: { color: "#fff", fontWeight: "700", fontSize: 13 },
});
