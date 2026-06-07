import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity, StatusBar,
  TextInput, FlatList, KeyboardAvoidingView, Platform,
  Keyboard, Dimensions, ScrollView, Animated, Easing, Image
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import * as ImagePicker from "expo-image-picker";
import Toast from "react-native-toast-message";
import { COLORS } from "../../constants/colors";
import { useTheme } from "../../context/ThemeContext";
import { useLanguage } from "../../context/LanguageContext";
import { getMyPatientId } from "../../services/authService";
import { parseMedicalProfile, analyzeMedicalImage } from "../../services/aiService";
import {
  getChronicDiseases, getMedications, getAllergies, getSurgeries, getVitals, getMedicalProfile,
  createChronicDisease, createMedication, createAllergy, createSurgery, createVital,
  updateChronicDisease, updateMedication, updateAllergy, updateSurgery, updateVital, updateMedicalProfile,
  deleteChronicDisease, deleteMedication, deleteAllergy, deleteSurgery, deleteVital,
  ChronicDisease, Medication, AllergyRecord, SurgeryRecord, VitalReading
} from "../../services/medicalRecordService";

const { width } = Dimensions.get("window");

/* ────────────────────── Types ────────────────────── */
interface ChatMsg {
  id: number;
  role: "user" | "assistant";
  content: string;
  image?: string;
  chips?: { label: string; value: string }[];
  timestamp: string;
}

type Intent = "loading" | "home" | "onboarding" | "add" | "edit" | "delete" | "after_action";
type SubIntent = "disease" | "med" | "allergy" | "surgery" | "vital" | "general" | null;

interface AppState {
  intent: Intent;
  subIntent: SubIntent;
  step: number;
  onboardingSection: "disease" | "med" | "allergy" | "surgery" | "vital" | "general" | "done";
  draft: any;
  messages: ChatMsg[];
}

/* ────────────────────── Animations ────────────────────── */
const TypingIndicator = () => {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animateDot = (val: Animated.Value, delay: number) => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(val, { toValue: 1, duration: 400, delay, useNativeDriver: true }),
          Animated.timing(val, { toValue: 0, duration: 400, useNativeDriver: true })
        ])
      ).start();
    };
    animateDot(dot1, 0);
    animateDot(dot2, 200);
    animateDot(dot3, 400);
  }, []);

  const getStyle = (val: Animated.Value) => ({
    opacity: val.interpolate({ inputRange: [0, 1], outputRange: [0.3, 1] }),
    transform: [{ translateY: val.interpolate({ inputRange: [0, 1], outputRange: [0, -4] }) }]
  });

  return (
    <View style={{ flexDirection: 'row', gap: 4, alignItems: 'center', height: 20 }}>
      <Animated.View style={[styles.typingDot, getStyle(dot1)]} />
      <Animated.View style={[styles.typingDot, getStyle(dot2)]} />
      <Animated.View style={[styles.typingDot, getStyle(dot3)]} />
    </View>
  );
};

const AnimatedMessage = ({ children, index }: { children: React.ReactNode, index: number }) => {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.spring(anim, {
      toValue: 1,
      friction: 8,
      tension: 40,
      useNativeDriver: true,
    }).start();
  }, []);

  return (
    <Animated.View style={{
      opacity: anim,
      transform: [
        { translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [20, 0] }) },
        { scale: anim.interpolate({ inputRange: [0, 1], outputRange: [0.95, 1] }) }
      ]
    }}>
      {children}
    </Animated.View>
  );
};

/* ────────────────────── Component ────────────────────── */
export default function AIProfileAssistantScreen() {
  const router = useRouter();
  const { isDark, colors } = useTheme();
  const { lang: globalLang } = useLanguage();

  const [localLang, setLocalLang] = useState(globalLang);
  const isAr = localLang === "ar";

  const [diseases, setDiseases] = useState<ChronicDisease[]>([]);
  const [meds, setMeds] = useState<Medication[]>([]);
  const [allergies, setAllergies] = useState<AllergyRecord[]>([]);
  const [surgeries, setSurgeries] = useState<SurgeryRecord[]>([]);
  const [vitals, setVitals] = useState<VitalReading[]>([]);
  const [profile, setProfile] = useState<any>(null);

  const [state, setState] = useState<AppState>({
    intent: "loading", subIntent: null, step: 0,
    onboardingSection: "disease", draft: {}, messages: [],
  });
  const [historyStack, setHistoryStack] = useState<AppState[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  /* ── helpers ── */
  const t = (ar: string, en: string) => (isAr ? ar : en);

  const pushState = (patch: Partial<AppState>) => {
    setHistoryStack((prev) => [...prev, state]);
    setState((prev) => ({ ...prev, ...patch }));
  };
  const handleUndo = () => {
    if (historyStack.length === 0) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const prev = historyStack[historyStack.length - 1];
    setHistoryStack((s) => s.slice(0, -1));
    setState(prev);
  };

  const addMsg = (
    content: string, role: "user" | "assistant",
    chips?: { label: string; value: string }[],
    image?: string
  ): ChatMsg => ({
    id: Date.now() + Math.random(),
    role, content, chips, image,
    timestamp: new Date().toISOString(),
  });

  /* ── data ── */
  const loadData = async () => {
    try {
      const pid = await getMyPatientId();
      if (pid > 0) {
        const [d, m, a, s, v, p] = await Promise.all([
          getChronicDiseases(pid), getMedications(pid), getAllergies(pid),
          getSurgeries(pid), getVitals(pid), getMedicalProfile(pid).catch(() => null)
        ]);
        const activeD = d.filter((x: any) => x.isActive !== false);
        const activeM = m.filter((x: any) => x.isActive !== false);
        const activeA = a.filter((x: any) => x.isActive !== false);
        setDiseases(activeD); setMeds(activeM); setAllergies(activeA);
        setSurgeries(s); setVitals(v); setProfile(p);
        return { count: activeD.length + activeM.length + activeA.length + s.length + v.length, pid };
      }
      return { count: 0, pid: 0 };
    } catch { return { count: 0, pid: 0 }; }
  };

  useEffect(() => {
    (async () => {
      const { count } = await loadData();
      count === 0 ? startOnboarding() : goHome();
    })();
  }, []);

  /* ── navigation helpers ── */
  const homeChips = () => [
    { label: t("إضافة جديد", "Add New"), value: "ADD" },
    { label: t("تعديل", "Edit"), value: "EDIT" },
    { label: t("حذف", "Delete"), value: "DELETE" },
    { label: t("إعداد كامل", "Full Setup"), value: "SETUP" },
  ];
  const afterChips = () => [
    { label: t("إضافة", "Add"), value: "ADD" },
    { label: t("تعديل", "Edit"), value: "EDIT" },
    { label: t("حذف", "Delete"), value: "DELETE" },
    { label: t("الرئيسية", "Home"), value: "HOME" },
  ];
  const catChips = () => [
    { label: t("أمراض", "Diseases"), value: "DISEASE" },
    { label: t("أدوية", "Medications"), value: "MED" },
    { label: t("حساسية", "Allergies"), value: "ALLERGY" },
    { label: t("جراحات", "Surgeries"), value: "SURGERY" },
    { label: t("قياسات", "Vitals"), value: "VITAL" },
    { label: t("بيانات عامة", "General"), value: "GENERAL" },
  ];
  const skipChips = () => [
    { label: t("تخطي", "Skip"), value: "SKIP" },
  ];
  const skipNoneChips = () => [
    { label: t("تخطي", "Skip"), value: "SKIP" },
    { label: t("لا يوجد", "None"), value: "NONE" },
  ];

  const startOnboarding = () => {
    setState({
      intent: "onboarding", subIntent: "disease", step: 1,
      onboardingSection: "disease", draft: {},
      messages: [addMsg(
        t("أهلاً بك!\nسنساعدك في إعداد ملفك الطبي بسرعة.\n\nهل لديك أمراض مزمنة؟ (مثال: سكر، ضغط)",
          "Welcome!\nLet's set up your medical profile.\n\nDo you have any chronic diseases? (e.g., Diabetes)"),
        "assistant", skipNoneChips()
      )],
    });
    setHistoryStack([]);
  };

  const goHome = () => {
    setState({
      intent: "home", subIntent: null, step: 0,
      onboardingSection: "done", draft: {},
      messages: [addMsg(
        t("مرحباً!\nكيف يمكنني مساعدتك اليوم؟", "Hi!\nHow can I help you today?"),
        "assistant", homeChips()
      )],
    });
    setHistoryStack([]);
  };

  const askAfterAction = (msgs: ChatMsg[]) => [
    ...msgs,
    addMsg(t("تم بنجاح!\nهل تريد فعل شيء آخر؟", "Done!\nAnything else?"), "assistant", afterChips()),
  ];

  /* ── silent AI fallback ── */
  const runSilentAI = async (text: string) => {
    try {
      const result = await parseMedicalProfile(text, true);
      if (result && (result.chronic_diseases.length > 0 || result.medications.length > 0 || result.allergies.length > 0))
        await loadData();
    } catch {}
  };

  /* ──────────────── STATE MACHINE ──────────────── */
  const processInput = async (label: string, value?: string) => {
    const val = value || label;
    const userMsg = addMsg(label, "user");
    const { intent, subIntent, step, draft, messages } = state;
    let next = [...messages, userMsg];
    const isKnownChip = [
      "SKIP","NONE","ADD","EDIT","DELETE","SETUP","HOME",
      "DISEASE","MED","ALLERGY","SURGERY","VITAL","GENERAL","YES","NO",
      "DRUG","FOOD","OTHER","CONFIRM_DEL","CANCEL_DEL","MALE","FEMALE",
    ].includes(val) || val.startsWith("ITEM_") || val.startsWith("DEL_");
    if (!isKnownChip) runSilentAI(label);

    try {
      setIsLoading(true);
      const pid = await getMyPatientId();

      /* ── HOME / AFTER_ACTION ── */
      if (intent === "home" || intent === "after_action") {
        if (val === "ADD") {
          next.push(addMsg(t("ماذا تريد أن تضيف؟", "What do you want to add?"), "assistant", catChips()));
          pushState({ intent: "add", step: 0, messages: next });
        } else if (val === "EDIT") {
          next.push(addMsg(t("ماذا تريد أن تعدّل؟", "What do you want to edit?"), "assistant", catChips()));
          pushState({ intent: "edit", step: 0, messages: next });
        } else if (val === "DELETE") {
          next.push(addMsg(t("ماذا تريد أن تحذف؟", "What do you want to delete?"), "assistant", catChips()));
          pushState({ intent: "delete", step: 0, messages: next });
        } else if (val === "SETUP") { startOnboarding(); }
        else if (val === "HOME") { goHome(); }
        else {
          next.push(addMsg(t("اختر من الخيارات المتاحة", "Choose from the options below"), "assistant", homeChips()));
          pushState({ messages: next });
        }
        return;
      }

      /* ── ADD ── */
      if (intent === "add" && step === 0) {
        if (val === "DISEASE") {
          next.push(addMsg(t("ما اسم المرض؟", "Disease name?"), "assistant", skipChips()));
          pushState({ subIntent: "disease", step: 1, draft: {}, messages: next });
        } else if (val === "SURGERY") {
          next.push(addMsg(t("ما اسم العملية الجراحية؟", "Surgery name?"), "assistant", skipChips()));
          pushState({ subIntent: "surgery", step: 1, draft: {}, messages: next });
        } else if (val === "VITAL") {
          next.push(addMsg(t("ما نوع القياس؟", "Vital type?"), "assistant", [
            { label: t("الضغط", "Blood Pressure"), value: "Blood Pressure" },
            { label: t("السكر", "Blood Sugar"), value: "Blood Sugar" },
          ]));
          pushState({ subIntent: "vital", step: 1, draft: {}, messages: next });
        } else if (val === "GENERAL") {
          next.push(addMsg(t("هل أنت مدخن؟", "Are you a smoker?"), "assistant", [
            { label: t("نعم", "Yes"), value: "YES" }, { label: t("لا", "No"), value: "NO" },
          ]));
          pushState({ subIntent: "general", step: 1, draft: profile || {}, messages: next });
        } else if (val === "MED") {
          next.push(addMsg(t("ما اسم الدواء؟", "Medication name?"), "assistant", skipChips()));
          pushState({ subIntent: "med", step: 1, draft: {}, messages: next });
        } else if (val === "ALLERGY") {
          next.push(addMsg(t("الحساسية من دواء أم طعام أم غير ذلك؟", "Drug, food, or other allergy?"), "assistant", [
            { label: t("دواء", "Drug"), value: "DRUG" },
            { label: t("طعام", "Food"), value: "FOOD" },
            { label: t("أخرى", "Other"), value: "OTHER" },
          ]));
          pushState({ subIntent: "allergy", step: 1, draft: {}, messages: next });
        }
        return;
      }

      /* ── EDIT (select category then item) ── */
      if (intent === "edit" && step === 0) {
        let chips: { label: string; value: string }[] = [];
        if (val === "DISEASE") {
          chips = diseases.map((d) => ({ label: d.diseaseName, value: `ITEM_${d.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر المرض:", "Select disease:"), "assistant", chips));
          pushState({ subIntent: "disease", step: 0.5, messages: next });
        } else if (val === "MED") {
          chips = meds.map((m) => ({ label: m.medicationName, value: `ITEM_${m.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الدواء:", "Select medication:"), "assistant", chips));
          pushState({ subIntent: "med", step: 0.5, messages: next });
} else if (val === "ALLERGY") {
          chips = allergies.map((a) => ({ label: a.allergenName, value: `ITEM_${a.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الحساسية:", "Select allergy:"), "assistant", chips));
          pushState({ subIntent: "allergy", step: 0.5, messages: next });
        } else if (val === "SURGERY") {
          chips = surgeries.map((s) => ({ label: s.surgeryName, value: `ITEM_${s.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر العملية لتعديلها:", "Select surgery:"), "assistant", chips));
          pushState({ subIntent: "surgery", step: 0.5, messages: next });
        } else if (val === "VITAL") {
          chips = vitals.map((v) => ({ label: v.readingType, value: `ITEM_${v.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر القياس لتعديله:", "Select vital:"), "assistant", chips));
          pushState({ subIntent: "vital", step: 0.5, messages: next });
        } else if (val === "GENERAL") {
          next.push(addMsg(t("هل أنت مدخن؟", "Are you a smoker?"), "assistant", [
            { label: t("نعم", "Yes"), value: "YES" }, { label: t("لا", "No"), value: "NO" },
          ]));
          pushState({ subIntent: "general", step: 1, draft: profile || {}, messages: next });
        }
        return;
      }

      if (intent === "edit" && step === 0.5) {
        if (val === "HOME") { goHome(); return; }
        const id = parseInt(val.replace("ITEM_", ""), 10);
        if (subIntent === "disease") {
          const item = diseases.find((d) => d.id === id);
          if (item) {
            next.push(addMsg(t(`اسم المرض الحالي: ${item.diseaseName}\nأدخل الاسم الجديد:`, `Current: ${item.diseaseName}\nNew name:`), "assistant", [{ label: item.diseaseName, value: item.diseaseName }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        } else if (subIntent === "med") {
          const item = meds.find((m) => m.id === id);
          if (item) {
            next.push(addMsg(t(`الدواء الحالي: ${item.medicationName}\nأدخل الاسم الجديد:`, `Current: ${item.medicationName}\nNew name:`), "assistant", [{ label: item.medicationName, value: item.medicationName }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
} else if (subIntent === "allergy") {
          const item = allergies.find((a) => a.id === id);
          if (item) {
            next.push(addMsg(t("نوع الحساسية:", "Allergy type:"), "assistant", [
              { label: t("دواء", "Drug"), value: "DRUG" },
              { label: t("طعام", "Food"), value: "FOOD" },
              { label: t("أخرى", "Other"), value: "OTHER" },
            ]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        } else if (subIntent === "surgery") {
          const item = surgeries.find((s) => s.id === id);
          if (item) {
            next.push(addMsg(t(`العملية الحالية: ${item.surgeryName}\nأدخل الاسم الجديد:`, `Current: ${item.surgeryName}\nNew name:`), "assistant", [{ label: item.surgeryName, value: item.surgeryName }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        } else if (subIntent === "vital") {
          const item = vitals.find((v) => v.id === id);
          if (item) {
            next.push(addMsg(t(`القياس الحالي: ${item.readingType}\nأدخل النوع الجديد:`, `Current: ${item.readingType}\nNew type:`), "assistant", [{ label: item.readingType, value: item.readingType }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        }
        return;
      }

      /* ── DELETE ── */
      if (intent === "delete" && step === 0) {
        let chips: { label: string; value: string }[] = [];
        if (val === "DISEASE") {
          chips = diseases.map((d) => ({ label: d.diseaseName, value: `DEL_${d.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر المرض لحذفه:", "Select disease to delete:"), "assistant", chips));
          pushState({ subIntent: "disease", step: 1, messages: next });
        } else if (val === "MED") {
          chips = meds.map((m) => ({ label: m.medicationName, value: `DEL_${m.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الدواء لحذفه:", "Select medication to delete:"), "assistant", chips));
          pushState({ subIntent: "med", step: 1, messages: next });
} else if (val === "ALLERGY") {
          chips = allergies.map((a) => ({ label: a.allergenName, value: `DEL_${a.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الحساسية لحذفها:", "Select allergy to delete:"), "assistant", chips));
          pushState({ subIntent: "allergy", step: 1, messages: next });
        } else if (val === "SURGERY") {
          chips = surgeries.map((s) => ({ label: s.surgeryName, value: `DEL_${s.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر العملية لحذفها:", "Select surgery to delete:"), "assistant", chips));
          pushState({ subIntent: "surgery", step: 1, messages: next });
        } else if (val === "VITAL") {
          chips = vitals.map((v) => ({ label: v.readingType, value: `DEL_${v.id}` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر القياس لحذفه:", "Select vital to delete:"), "assistant", chips));
          pushState({ subIntent: "vital", step: 1, messages: next });
        }
        return;
      }

      if (intent === "delete" && step === 1) {
        if (val === "HOME") { goHome(); return; }
        const id = parseInt(val.replace("DEL_", ""), 10);
        let name = label;
        next.push(addMsg(
          t(`هل تريد حذف "${name}"؟`, `Delete "${name}"?`), "assistant",
          [
            { label: t("نعم، احذف", "Yes, delete"), value: "CONFIRM_DEL" },
            { label: t("لا، تراجع", "No, cancel"), value: "CANCEL_DEL" },
          ]
        ));
        pushState({ step: 2, draft: { id }, messages: next });
        return;
      }

      if (intent === "delete" && step === 2) {
        if (val === "CONFIRM_DEL") {
          if (subIntent === "disease") await deleteChronicDisease(draft.id);
          if (subIntent === "med") await deleteMedication(draft.id);
          if (subIntent === "allergy") await deleteAllergy(draft.id);
          if (subIntent === "surgery") await deleteSurgery(draft.id);
          if (subIntent === "vital") await deleteVital(draft.id);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        } else { goHome(); }
        return;
      }

      /* ── MINI-FLOWS ── */
      const isSkip = val === "SKIP" || val === "NONE";

      // Disease flow
      if (subIntent === "disease") {
        if (step === 1) {
          if (isSkip && intent === "onboarding") {
            next.push(addMsg(t("ممتاز.\nهل تتناول أي أدوية حالياً؟", "Great.\nAre you taking any medications?"), "assistant", skipNoneChips()));
            pushState({ subIntent: "med", step: 1, onboardingSection: "med", draft: {}, messages: next });
            return;
          }
          draft.diseaseName = isSkip ? "Unknown" : label;
          next.push(addMsg(t("منذ متى؟ (مثال: 3 سنوات)", "Since when? (e.g. 3 years)"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.duration = isSkip ? "" : label;
          next.push(addMsg(t("هل تتلقى علاجاً له؟", "Are you receiving treatment?"), "assistant", [
            { label: t("نعم", "Yes"), value: "YES" }, { label: t("لا", "No"), value: "NO" },
          ]));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.treatment = val === "YES" ? "Yes" : val === "NO" ? "No" : label;
          const payload = { 
            diseaseName: draft.diseaseName, 
            doctorNotes: `Since: ${draft.duration}, Treatment: ${draft.treatment}`, 
            diseaseType: "Chronic", 
            severity: "Moderate", 
            monitoringFrequency: "Monthly",
            isActive: true 
          };
          if (intent === "edit") await updateChronicDisease(draft.id, payload);
          else await createChronicDisease(pid, payload);
          await loadData();
          if (intent === "onboarding") {
            next.push(addMsg(t("تم الحفظ.\nما اسم الدواء؟", "Saved.\nMedication name?"), "assistant", skipNoneChips()));
            pushState({ subIntent: "med", step: 1, onboardingSection: "med", draft: {}, messages: next });
          } else {
            next = askAfterAction(next);
            pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
          }
        }
      }

      // Medication flow
      else if (subIntent === "med") {
        if (step === 1) {
          if (isSkip && intent === "onboarding") {
            next.push(addMsg(t("حسناً.\nهل لديك حساسية تجاه شيء معين؟", "Alright.\nDo you have any allergies?"), "assistant", [
              { label: t("دواء", "Drug"), value: "DRUG" }, { label: t("طعام", "Food"), value: "FOOD" },
              { label: t("أخرى", "Other"), value: "OTHER" }, { label: t("لا يوجد", "None"), value: "NONE" },
            ]));
            pushState({ subIntent: "allergy", step: 1, onboardingSection: "allergy", draft: {}, messages: next });
            return;
          }
          draft.medicationName = isSkip ? "Unknown" : label;
          next.push(addMsg(t("ما الجرعة؟ (مثال: 500mg)", "Dosage? (e.g. 500mg)"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.dosage = isSkip ? "" : label;
          next.push(addMsg(t("كم مرة في اليوم؟", "How many times a day?"), "assistant", [
            { label: t("مرة يومياً", "Once daily"), value: t("مرة يومياً", "Once daily") },
            { label: t("مرتين يومياً", "Twice daily"), value: t("مرتين يومياً", "Twice daily") },
          ]));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.frequency = label;
          const isTwice = draft.frequency.includes("مرتين");
          const payload = { 
            medicationName: draft.medicationName, 
            dosage: draft.dosage, 
            frequency: draft.frequency, 
            form: "Tablet", 
            isChronic: true, 
            isActive: true, 
            startDate: new Date().toISOString().split("T")[0],
            timesPerDay: isTwice ? 2 : 1,
            doseTimes: isTwice ? "08:00, 20:00" : "08:00",
            daysOfWeek: "All",
            refillThreshold: 5
          };
          if (intent === "edit") await updateMedication(draft.id, payload);
          else await createMedication(pid, payload);
          await loadData();
          if (intent === "onboarding") {
            next.push(addMsg(t("تم الحفظ.\nهل لديك حساسية؟", "Saved.\nAny allergies?"), "assistant", [
              { label: t("دواء", "Drug"), value: "DRUG" }, { label: t("طعام", "Food"), value: "FOOD" },
              { label: t("أخرى", "Other"), value: "OTHER" }, { label: t("لا يوجد", "None"), value: "NONE" },
            ]));
            pushState({ subIntent: "allergy", step: 1, onboardingSection: "allergy", draft: {}, messages: next });
          } else {
            next = askAfterAction(next);
            pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
          }
        }
      }

      // Allergy flow
      else if (subIntent === "allergy") {
        if (step === 1) {
          if (isSkip && intent === "onboarding") {
            next.push(addMsg(t("تم إعداد ملفك الطبي بالكامل.", "Profile setup complete."), "assistant"));
            next = askAfterAction(next);
            pushState({ intent: "after_action", subIntent: null, step: 0, onboardingSection: "done", draft: {}, messages: next });
            return;
          }
          draft.allergyType = val === "DRUG" ? "Drug" : val === "FOOD" ? "Food" : "Other";
          next.push(addMsg(t("ما اسم المادة المسببة للحساسية؟", "Allergen name?"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.allergenName = isSkip ? "Unknown" : label;
          next.push(addMsg(t("ما رد فعل جسمك؟", "What is your reaction?"), "assistant", [
            { label: t("طفح جلدي", "Rash"), value: t("طفح جلدي", "Rash") },
            { label: t("ضيق تنفس", "Shortness of breath"), value: t("ضيق تنفس", "Shortness of breath") },
          ]));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.reactionDescription = label;
          const payload = { allergenName: draft.allergenName, allergyType: draft.allergyType, reactionDescription: draft.reactionDescription, severity: "Moderate", isActive: true };
          if (intent === "edit") await updateAllergy(draft.id, payload);
          else await createAllergy(pid, payload);
          await loadData();
          if (intent === "onboarding") {
            next.push(addMsg(t("تم إعداد ملفك بالكامل.", "Profile complete."), "assistant"));
          }
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, onboardingSection: "done", draft: {}, messages: next });
        }
      }

      // Surgery flow
      else if (subIntent === "surgery") {
        if (step === 1) {
          draft.surgeryName = isSkip ? "Unknown" : label;
          next.push(addMsg(t("متى أجريت هذه العملية؟", "When was this surgery?"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.surgeryDate = isSkip ? new Date().toISOString().split("T")[0] : label;
          const payload = { surgeryName: draft.surgeryName, surgeryDate: draft.surgeryDate, isActive: true };
          if (intent === "edit") await updateSurgery(draft.id, payload);
          else await createSurgery(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }

      // Vital flow
      else if (subIntent === "vital") {
        if (step === 1) {
          draft.readingType = isSkip ? "Blood Pressure" : label;
          next.push(addMsg(t("ما هي القيمة؟ (مثال: 120)", "What is the value? (e.g. 120)"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.value = parseFloat(label) || 120;
          next.push(addMsg(t("ما هي القيمة الثانية (إذا وجدت، للضغط)؟", "Second value (if any, for BP)?"), "assistant", skipNoneChips()));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.value2 = isSkip ? undefined : parseFloat(label) || 80;
          const payload = { readingType: draft.readingType, value: draft.value, value2: draft.value2, unit: "standard", isNormal: true };
          if (intent === "edit") await updateVital(draft.id, payload);
          else await createVital(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }

      // General profile flow
      else if (subIntent === "general") {
        if (step === 1) {
          draft.isSmoker = val === "YES";
          next.push(addMsg(t("فصيلة الدم؟", "Blood Type?"), "assistant", [
            { label: "A+", value: "A+" }, { label: "O+", value: "O+" }, { label: "B+", value: "B+" }, { label: "AB+", value: "AB+" }
          ]));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.bloodType = isSkip ? "" : label;
          next.push(addMsg(t("الوزن (كجم)؟", "Weight (kg)?"), "assistant", skipChips()));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.weightKg = parseFloat(label) || 70;
          next.push(addMsg(t("الطول (سم)؟", "Height (cm)?"), "assistant", skipChips()));
          pushState({ step: 4, draft, messages: next });
        } else if (step === 4) {
          draft.heightCm = parseFloat(label) || 170;
          const payload = { 
            isSmoker: draft.isSmoker, 
            bloodType: draft.bloodType, 
            weightKg: draft.weightKg, 
            heightCm: draft.heightCm 
          };
          await updateMedicalProfile(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }
    } catch (error: any) {
      console.error("[AI-Profile] Error:", error);
      next.push(addMsg(t("حدث خطأ. يرجى المحاولة مجدداً.", "An error occurred. Please try again."), "assistant"));
      pushState({ messages: next });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendText = () => {
    if (!inputText.trim()) return;
    const txt = inputText.trim();
    setInputText("");
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    processInput(txt);
  };

  const handleImagePick = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Toast.show({
          type: "error",
          text1: t("عذراً", "Sorry"),
          text2: t("نحتاج إذن الوصول للاستوديو لرفع الصورة", "We need media library permissions to pick an image"),
        });
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0].uri) {
        uploadImage(result.assets[0].uri);
      }
    } catch (error) {
      Toast.show({ type: "error", text1: t("خطأ في اختيار الصورة", "Error selecting image") });
    }
  };

  const uploadImage = async (uri: string) => {
    setIsLoading(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    // Add user message with image preview
    const userMsg = addMsg(t("تم رفع صورة للتحليل...", "Uploaded an image for analysis..."), "user", undefined, uri);
    const { messages } = state;
    const next = [...messages, userMsg];
    pushState({ messages: next });

    try {
      // Analyze the image using our AI service
      const res = await analyzeMedicalImage(uri, "prescription");
      const data = res as any;

      // Build readable response
      const analysis = isAr ? data.analysis_ar : (data.analysis_en || data.analysis_ar);
      const technical = data.technical_details ? `\n\n${t("التفاصيل التقنية:", "Technical Details:")}\n${data.technical_details}` : "";
      const disclaimer = data.disclaimer ? `\n\n${data.disclaimer}` : "";
      const assistantResponse = `${analysis}${technical}${disclaimer}`;

      const assistantMsg = addMsg(assistantResponse, "assistant");
      const updatedMessages = [...next, assistantMsg];
      
      // Update assistant state with results
      pushState({ messages: updatedMessages });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

      // Run silent AI parsing to extract diseases, meds, vitals etc. from the OCR text
      await runSilentAI(assistantResponse);
      await loadData();
      
    } catch (error) {
      console.error("[AI-Profile] Upload Error:", error);
      const errorMsg = addMsg(t("فشل تحليل الصورة. يرجى المحاولة مجدداً أو كتابة البيانات.", "Failed to analyze image. Please try again or write the data manually."), "assistant");
      pushState({ messages: [...next, errorMsg] });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setIsLoading(false);
    }
  };

  /* ──────────────── RENDER ──────────────── */
  const renderSummaryCard = () => {
    if (!diseases.length && !meds.length && !allergies.length && !surgeries.length && !vitals.length) return null;
    const items = [
      { icon: "heart", count: diseases.length, label: t("أمراض", "Diseases") },
      { icon: "medkit", count: meds.length, label: t("أدوية", "Meds") },
      { icon: "alert-circle", count: allergies.length, label: t("حساسية", "Allergies") },
      { icon: "cut", count: surgeries.length, label: t("جراحات", "Surgeries") },
      { icon: "pulse", count: vitals.length, label: t("قياسات", "Vitals") },
    ];
    return (
      <View style={[styles.summaryCard, { backgroundColor: isDark ? colors.surface : "#fff", borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)" }]}>
        <View style={styles.summaryRow}>
          {items.map((it, i) => (
            <View key={i} style={[styles.summaryItem, { backgroundColor: isDark ? "rgba(255,255,255,0.04)" : "#F8FAF9" }]}>
              <Ionicons name={it.icon as any} size={16} color={COLORS.primary} />
              <Text style={[styles.summaryCount, { color: colors.text }]}>{it.count}</Text>
              <Text style={[styles.summaryLabel, { color: colors.textMuted }]}>{it.label}</Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const renderMessage = ({ item, index }: { item: ChatMsg; index: number }) => {
    const isUser = item.role === "user";
    const isLast = index === state.messages.length - 1;

    return (
      <AnimatedMessage index={index}>
        <View style={{ marginBottom: 14 }}>
        {/* Bubble row */}
        <View style={[styles.bubbleRow, isUser ? styles.bubbleRowUser : styles.bubbleRowAI]}>
          {!isUser && (
            <View style={styles.aiAvatar}>
              <Ionicons name="sparkles" size={14} color="#fff" />
            </View>
          )}
          {isUser ? (
            <LinearGradient
              colors={[COLORS.primary, "#047857"]}
              style={[styles.bubble, styles.userBubble]}
            >
              {item.image && (
                <Image 
                  source={{ uri: item.image }} 
                  style={styles.bubbleImage} 
                  resizeMode="cover"
                />
              )}
              {item.content && <Text style={[styles.bubbleText, { color: "#fff" }]}>{item.content}</Text>}
              <Text style={[styles.bubbleTime, { color: "rgba(255,255,255,0.5)" }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </LinearGradient>
          ) : (
            <View style={[styles.bubble, styles.aiBubble, { backgroundColor: isDark ? "rgba(255,255,255,0.05)" : "#fff", borderColor: isDark ? "rgba(255,255,255,0.08)" : "#E2E8F0" }]}>
              {item.content && <Text style={[styles.bubbleText, { color: colors.text }]}>{item.content}</Text>}
              <Text style={[styles.bubbleTime, { color: colors.textMuted }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </View>
          )}
        </View>

        {/* Summary card on home */}
        {!isUser && (item.content.includes("مرحباً") || item.content.includes("Hi!") || item.content.includes("Welcome")) && (
          <View style={{ marginTop: 10, marginLeft: 34 }}>{renderSummaryCard()}</View>
        )}

        {/* Chips - rendered horizontally جنب بعض */}
        {isLast && !isUser && item.chips && item.chips.length > 0 && (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.chipsWrap}
            contentContainerStyle={styles.chipsScroll}
          >
            {item.chips.map((chip, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.chip, { 
                  backgroundColor: isDark ? "rgba(255,255,255,0.06)" : "#fff", 
                  borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.08)" 
                }]}
                onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); processInput(chip.label, chip.value); }}
                activeOpacity={0.6}
              >
                <Text style={[styles.chipLabel, { color: isDark ? "#D1D5DB" : "#374151" }]}>{chip.label}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </View>
      </AnimatedMessage>
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: isDark ? "#0F1419" : "#F5F7F6" }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor="transparent" translucent />

      {/* Background Blobs centered like in Home screen */}
      <View style={[styles.bgBubble, styles.bubbleTopLeft, { backgroundColor: isDark ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.08)' }]} />
      <View style={[styles.bgBubble, styles.bubbleBottomRight, { backgroundColor: isDark ? 'rgba(14, 165, 233, 0.15)' : 'rgba(14, 165, 233, 0.08)' }]} />
      <View style={[styles.bgBubble, styles.bubbleCenter, { backgroundColor: isDark ? 'rgba(139, 92, 246, 0.1)' : 'rgba(139, 92, 246, 0.05)' }]} />

      {/* ── Header ── */}
      <View style={styles.headerWrap}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={styles.headerGradient}>
          <View style={styles.headerContent}>
            <TouchableOpacity onPress={() => router.back()} style={styles.headerBtn} activeOpacity={0.7}>
              <Ionicons name="chevron-back" size={20} color="#fff" />
            </TouchableOpacity>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.headerTitle}>{t("المساعد الطبي", "Medical Assistant")}</Text>
              <View style={styles.statusRow}>
                <View style={styles.statusDot} />
                <Text style={styles.statusLabel}>{t("متصل • ذكاء اصطناعي", "Online • AI")}</Text>
              </View>
            </View>
            <TouchableOpacity
              onPress={() => { setLocalLang(isAr ? "en" : "ar"); Haptics.selectionAsync(); }}
              style={[styles.headerBtn, { marginRight: 6 }]}
              activeOpacity={0.7}
            >
              <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{isAr ? "EN" : "عربي"}</Text>
            </TouchableOpacity>
            {historyStack.length > 0 && (
              <TouchableOpacity onPress={handleUndo} style={styles.headerBtn} activeOpacity={0.7}>
                <Ionicons name="arrow-undo" size={16} color="#fff" />
              </TouchableOpacity>
            )}
          </View>
          <View style={[styles.blob, { top: -30, right: -30, width: 140, height: 140, backgroundColor: "#10B981", opacity: 0.08 }]} />
          <View style={[styles.blob, { bottom: -20, left: -20, width: 100, height: 100, backgroundColor: "#34D399", opacity: 0.06 }]} />
        </LinearGradient>
      </View>

      {/* ── Messages ── */}
      <FlatList
        ref={flatListRef}
        data={state.messages}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderMessage}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />

      {/* ── Typing indicator ── */}
      {isLoading && (
        <View style={styles.typingRow}>
          <TypingIndicator />
          <Text style={[styles.typingText, { color: colors.textMuted }]}>{t("جاري التفكير...", "Thinking...")}</Text>
        </View>
      )}

      {/* ── Input ── */}
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}>
        <View style={[styles.inputContainer, { backgroundColor: isDark ? "#1A1F2E" : "#fff", borderTopColor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)" }]}>
          <View style={[styles.inputRow, { backgroundColor: isDark ? "#252D3A" : "#F1F5F3" }]}>
            <TouchableOpacity
              style={styles.attachBtn}
              onPress={handleImagePick}
              disabled={isLoading}
              activeOpacity={0.7}
            >
              <Ionicons name="camera" size={20} color={isDark ? "#9CA3AF" : "#6B7280"} />
            </TouchableOpacity>
            <TextInput
              style={[styles.input, { color: colors.text, textAlign: isAr ? "right" : "left" }]}
              placeholder={t("اكتب هنا أو ارفع صورة...", "Type here or pick image...")}
              placeholderTextColor={isDark ? "#4B5563" : "#94A3B8"}
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={200}
            />
            <TouchableOpacity
              style={[styles.sendBtn, !inputText.trim() && { opacity: 0.35 }]}
              onPress={handleSendText}
              disabled={!inputText.trim() || isLoading}
              activeOpacity={0.7}
            >
              <Ionicons name="send" size={16} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
      <Toast />
    </View>
  );
}

/* ──────────────── STYLES ──────────────── */
const styles = StyleSheet.create({
  container: { flex: 1 },

  /* Background Blobs */
  bgBubble: { position: 'absolute', borderRadius: 300, filter: 'blur(40px)' } as any,
  bubbleTopLeft: { width: 350, height: 350, top: -100, left: -100 },
  bubbleBottomRight: { width: 400, height: 400, bottom: -150, right: -150 },
  bubbleCenter: { width: 250, height: 250, top: '40%', left: '20%' },

  /* Header */
  headerWrap: { borderBottomLeftRadius: 24, borderBottomRightRadius: 24, overflow: "hidden", elevation: 6, shadowColor: "#064E3B", shadowOpacity: 0.1, shadowRadius: 10, shadowOffset: { width: 0, height: 3 } },
  headerGradient: { paddingTop: Platform.OS === "ios" ? 54 : 44, paddingBottom: 16, paddingHorizontal: 16 },
  headerContent: { flexDirection: "row", alignItems: "center", zIndex: 10 },
  headerBtn: { width: 36, height: 36, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.15)", justifyContent: "center", alignItems: "center" },
  headerTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  statusRow: { flexDirection: "row", alignItems: "center", marginTop: 3, gap: 5 },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: "#34D399" },
  statusLabel: { color: "rgba(255,255,255,0.65)", fontSize: 11, fontWeight: "500" },
  blob: { position: "absolute", borderRadius: 200 },

  /* Messages */
  listContent: { paddingHorizontal: 14, paddingTop: 16, paddingBottom: 16 },
  bubbleRow: { flexDirection: "row", maxWidth: "85%" },
  bubbleRowUser: { alignSelf: "flex-end" },
  bubbleRowAI: { alignSelf: "flex-start" },
  aiAvatar: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center", marginRight: 8, marginTop: 2 },
  bubble: { padding: 12, borderRadius: 16, maxWidth: width * 0.74 },
  userBubble: { borderBottomRightRadius: 4 },
  aiBubble: { borderBottomLeftRadius: 4, borderWidth: 1 },
  bubbleText: { fontSize: 14.5, lineHeight: 22 },
  bubbleTime: { fontSize: 10, marginTop: 4, alignSelf: "flex-end" },
  bubbleImage: { width: 200, height: 140, borderRadius: 12, marginBottom: 6 },

  /* Chips */
  chipsWrap: { marginTop: 8, paddingLeft: 14, flexDirection: "row" },
  chipsScroll: { paddingRight: 16, flexDirection: "row", gap: 8 },
  chip: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 20, borderWidth: 1 },
  chipLabel: { fontSize: 13.5, fontWeight: "600" },

  /* Summary card */
  summaryCard: { padding: 14, borderRadius: 20, borderWidth: 1 },
  summaryRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, justifyContent: "center" },
  summaryItem: { minWidth: 58, flex: 1, paddingVertical: 10, borderRadius: 14, alignItems: "center" },
  summaryCount: { fontSize: 18, fontWeight: "800", marginTop: 4 },
  summaryLabel: { fontSize: 10, fontWeight: "600", marginTop: 2 },
  summarySubtext: { fontSize: 10, color: "#888", marginTop: 2, textAlign: "center" },
  chatContainer: { flex: 1, padding: 14 },

  /* Typing */
  typingRow: { flexDirection: "row", alignItems: "center", paddingHorizontal: 14, paddingBottom: 8, gap: 8, marginLeft: 14 },
  typingText: { fontSize: 12, fontStyle: "italic" },
  typingDot: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: COLORS.primary },

  /* Input */
  inputContainer: { paddingHorizontal: 12, paddingTop: 10, paddingBottom: Platform.OS === "ios" ? 28 : 10, borderTopWidth: 1 },
  inputRow: { flexDirection: "row", alignItems: "center", borderRadius: 24, paddingLeft: 10, paddingRight: 5, paddingVertical: 4 },
  input: { flex: 1, minHeight: 38, maxHeight: 100, fontSize: 15, paddingHorizontal: 6, paddingTop: Platform.OS === "ios" ? 10 : 7 },
  sendBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center", marginLeft: 6 },
  attachBtn: { width: 36, height: 36, borderRadius: 18, justifyContent: "center", alignItems: "center", marginRight: 4 },
});
