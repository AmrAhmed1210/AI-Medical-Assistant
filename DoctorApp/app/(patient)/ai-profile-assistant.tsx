import React, { useState, useRef, useCallback } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity, StatusBar,
  TextInput, FlatList, KeyboardAvoidingView, Platform,
  ActivityIndicator, Keyboard, Animated, Dimensions, ScrollView
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import {
  Sparkles, Send, HeartPulse, Pill, AlertTriangle,
  CheckCircle2, ChevronRight, Bot, User, Stethoscope,
  ShieldCheck, Zap, MessageCircle
} from "lucide-react-native";
import * as Haptics from "expo-haptics";
import Toast from "react-native-toast-message";
import { useTheme } from "../../context/ThemeContext";
import { useLanguage } from "../../context/LanguageContext";
import { getMyPatientId } from "../../services/authService";
import { parseMedicalProfile, ParsedMedicalProfile } from "../../services/aiService";
import { createAllergy, createChronicDisease, createMedication } from "../../services/medicalRecordService";

const { width } = Dimensions.get("window");

interface ChatMsg {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
  parsedData?: ParsedMedicalProfile;
  timestamp: string;
}

const SUGGESTION_CHIPS_AR = [
  "عندي سكر وضغط عالي",
  "باخد أدوية للقلب",
  "عندي حساسية من البنسلين",
  "عملت عملية في الركبة",
];

const SUGGESTION_CHIPS_EN = [
  "I have diabetes and hypertension",
  "I take heart medication",
  "I'm allergic to Penicillin",
  "I had knee surgery",
];

const SAVE_COMMANDS = ["save", "saved", "done", "finish", "finished", "حفظ", "احفظ", "تمام", "سيف"];

const countProfileData = (data?: ParsedMedicalProfile | null) => {
  if (!data) return 0;
  return data.chronic_diseases.length + data.medications.length + data.allergies.length;
};

const collectParsedData = (items: ChatMsg[]): ParsedMedicalProfile => {
  const collected: ParsedMedicalProfile = {
    chronic_diseases: [],
    medications: [],
    allergies: [],
    summary_ar: "",
    summary_en: "",
    follow_up_ar: "",
    follow_up_en: "",
  };

  items.forEach((item) => {
    if (!item.parsedData) return;
    collected.chronic_diseases.push(...item.parsedData.chronic_diseases);
    collected.medications.push(...item.parsedData.medications);
    collected.allergies.push(...item.parsedData.allergies);
  });

  return collected;
};

const buildNextPrompt = (data: ParsedMedicalProfile, isAr: boolean) => {
  const hasDiseases = data.chronic_diseases.length > 0;
  const hasMeds = data.medications.length > 0;
  const hasAllergies = data.allergies.length > 0;

  if (!hasDiseases) {
    return isAr
      ? "هل عندك امراض مزمنة مثل السكر، الضغط، القلب، الحساسية الصدرية، او اي تشخيص مستمر؟ اكتبها بطريقتك."
      : "Do you have any chronic conditions such as diabetes, hypertension, heart disease, asthma, or any long-term diagnosis? Type them freely.";
  }

  if (!hasMeds) {
    return isAr
      ? "هل تستخدم ادوية حاليا؟ اكتب اسم الدواء والجرعة وعدد المرات لو تعرفهم."
      : "Are you taking any medications now? Send the medication name, dose, and frequency if you know them.";
  }

  if (!hasAllergies) {
    return isAr
      ? "هل عندك حساسية من دواء، اكل، او مادة معينة؟ لو لا، اكتب لا يوجد حساسية."
      : "Do you have any allergy to medications, food, or anything else? If not, write no allergies.";
  }

  return isAr
    ? "جمعت البيانات الاساسية. لو في تفاصيل اضافية اكتبها، او اكتب Save واضغط حفظ الكل لتسجيلها في السجل الطبي."
    : "I collected the main details. Add anything else, or type Save / tap Save All to record them in your medical profile.";
};

const saveCollectedData = async (patientId: number, data: ParsedMedicalProfile) => {
  const jobs: Promise<unknown>[] = [];

  data.chronic_diseases.forEach((d) => {
    if (!d.diseaseName?.trim()) return;
    jobs.push(createChronicDisease(patientId, {
      diseaseName: d.diseaseName.trim(),
      diseaseType: d.diseaseType || "Chronic",
      severity: d.severity || "Moderate",
      monitoringFrequency: "Monthly",
      isActive: true,
      doctorNotes: d.notes || undefined,
    }));
  });

  data.medications.forEach((m) => {
    if (!m.medicationName?.trim()) return;
    jobs.push(createMedication(patientId, {
      medicationName: m.medicationName.trim(),
      genericName: m.genericName,
      dosage: m.dosage || "",
      form: m.form || "Tablet",
      frequency: m.frequency || "Once daily",
      startDate: new Date().toISOString().split("T")[0],
      instructions: m.instructions,
      isChronic: m.isChronic ?? true,
      isActive: true,
    }));
  });

  data.allergies.forEach((a) => {
    if (!a.allergenName?.trim()) return;
    jobs.push(createAllergy(patientId, {
      allergenName: a.allergenName.trim(),
      allergyType: a.allergyType || "Other",
      severity: a.severity || "Moderate",
      reactionDescription: a.reactionDescription,
      isActive: true,
    }));
  });

  await Promise.all(jobs);
};

export default function AIProfileAssistantScreen() {
  const router = useRouter();
  const { colors, isDark } = useTheme();
  const { lang } = useLanguage();
  const isAr = lang === "ar";

  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [pendingData, setPendingData] = useState<ParsedMedicalProfile | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Pulse animation for the AI avatar
  const startPulse = useCallback(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, [pulseAnim]);

  const handleNewChat = () => {
    setMessages([]);
    setPendingData(null);
    setSavedSuccess(false);
    setInputText("");
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleSend = async (text?: string) => {
    const msg = (text || inputText).trim();
    if (!msg || isLoading) return;

    setInputText("");
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    const userMsg: ChatMsg = {
      id: Date.now(),
      role: "user",
      content: msg,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMsg]);

    if (SAVE_COMMANDS.includes(msg.toLowerCase()) && countProfileData(collectParsedData(messages)) > 0) {
      await handleSaveAll(messages);
      return;
    }

    setIsLoading(true);
    startPulse();

    try {
      // First call: parse only (don't save yet)
      const result = await parseMedicalProfile(msg, false);

      const hasData =
        result.chronic_diseases.length > 0 ||
        result.medications.length > 0 ||
        result.allergies.length > 0;

      const summary = result.summary_ar || result.summary_en;
      const followUp = result.follow_up_ar || result.follow_up_en;

      const assistantMsg: ChatMsg = {
        id: Date.now() + 1,
        role: "assistant",
        content: hasData ? summary : "لم أتمكن من استخراج بيانات طبية من النص. حاول وصف حالتك بشكل أوضح 😊",
        parsedData: hasData ? result : undefined,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMsg]);

      if (hasData) {
        const conversationWithParsedData = [...messages, userMsg, assistantMsg];
        const collectedData = collectParsedData(conversationWithParsedData);
        setPendingData(collectedData);

        // Add follow-up question or final save prompt.
        const followUpMsg: ChatMsg = {
          id: Date.now() + 2,
          role: "assistant",
          content: buildNextPrompt(collectedData, isAr) || followUp,
          timestamp: new Date().toISOString(),
        };
        setTimeout(() => setMessages(prev => [...prev, followUpMsg]), 500);
      }

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error: any) {
      const errorMsg: ChatMsg = {
        id: Date.now() + 1,
        role: "assistant",
        content: "عذراً، حصل خطأ. حاول تاني. 🙏",
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMsg]);
      Toast.show({ type: "error", text1: "Error", text2: error?.message || "AI service error" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAll = async (sourceMessages: ChatMsg[] = messages) => {
    const collected = collectParsedData(sourceMessages);
    if (countProfileData(collected) === 0 || isSaving) return;

    setIsSaving(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

    try {
      const patientId = await getMyPatientId();
      if (patientId <= 0) {
        throw new Error("Patient profile was not found. Please login again.");
      }

      await saveCollectedData(patientId, collected);

      setSavedSuccess(true);
      setPendingData(null);

      const successMsg: ChatMsg = {
        id: Date.now(),
        role: "system",
        content: isAr
          ? "✅ تم حفظ جميع بياناتك الصحية بنجاح! يمكنك مراجعتها من صفحة السجل الطبي."
          : "✅ All your health data has been saved successfully! You can review it from your Medical Records page.",
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, successMsg]);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Toast.show({
        type: "success",
        text1: isAr ? "تم الحفظ بنجاح! 🎉" : "Saved Successfully! 🎉",
        text2: isAr ? "بياناتك الصحية اتسجلت" : "Your health data has been recorded",
      });
    } catch (error: any) {
      Toast.show({ type: "error", text1: "Error", text2: error?.message || "Failed to save" });
    } finally {
      setIsSaving(false);
    }
  };

  const totalExtracted = countProfileData(pendingData);

  const renderExtractedCards = (data: ParsedMedicalProfile) => (
    <View style={styles.cardsContainer}>
      {data.chronic_diseases.map((d, i) => (
        <View key={`d-${i}`} style={[styles.extractedCard, { backgroundColor: isDark ? "#1A1A2E" : "#FFF5F5", borderColor: isDark ? "#2D1B2E" : "#FECACA" }]}>
          <View style={[styles.cardIconBox, { backgroundColor: "#FEE2E2" }]}>
            <HeartPulse size={16} color="#EF4444" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[styles.cardLabel, { color: isDark ? "#FCA5A5" : "#991B1B" }]}>
              {isAr ? "مرض مزمن" : "Chronic Disease"}
            </Text>
            <Text style={[styles.cardValue, { color: colors.text }]}>
              {d.diseaseName}
            </Text>
            {d.diseaseNameAr && !isAr ? null : d.diseaseNameAr ? (
              <Text style={[styles.cardSub, { color: colors.textMuted }]}>{d.diseaseNameAr}</Text>
            ) : null}
            <Text style={[styles.cardMeta, { color: colors.textMuted }]}>
              {d.severity} • {d.diseaseType}
            </Text>
          </View>
        </View>
      ))}

      {data.medications.map((m, i) => (
        <View key={`m-${i}`} style={[styles.extractedCard, { backgroundColor: isDark ? "#1A1A2E" : "#F0F9FF", borderColor: isDark ? "#1B2D3E" : "#BAE6FD" }]}>
          <View style={[styles.cardIconBox, { backgroundColor: "#DBEAFE" }]}>
            <Pill size={16} color="#3B82F6" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[styles.cardLabel, { color: isDark ? "#93C5FD" : "#1E40AF" }]}>
              {isAr ? "دواء" : "Medication"}
            </Text>
            <Text style={[styles.cardValue, { color: colors.text }]}>
              {m.medicationName} {m.dosage ? `(${m.dosage})` : ""}
            </Text>
            <Text style={[styles.cardMeta, { color: colors.textMuted }]}>
              {m.frequency} • {m.form}
            </Text>
            {m.doseTimes ? (
              <Text style={[styles.cardMeta, { color: "#059669", marginTop: 2, fontWeight: "600" }]}>
                {isAr ? "المواعيد: " : "Times: "}{m.doseTimes}
              </Text>
            ) : null}
          </View>
        </View>
      ))}

      {data.allergies.map((a, i) => (
        <View key={`a-${i}`} style={[styles.extractedCard, { backgroundColor: isDark ? "#1A1A2E" : "#FFFBEB", borderColor: isDark ? "#2E2D1B" : "#FDE68A" }]}>
          <View style={[styles.cardIconBox, { backgroundColor: "#FEF3C7" }]}>
            <AlertTriangle size={16} color="#F59E0B" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[styles.cardLabel, { color: isDark ? "#FCD34D" : "#92400E" }]}>
              {isAr ? "حساسية" : "Allergy"}
            </Text>
            <Text style={[styles.cardValue, { color: colors.text }]}>
              {a.allergenName}
            </Text>
            <Text style={[styles.cardMeta, { color: colors.textMuted }]}>
              {a.severity} • {a.allergyType}
            </Text>
          </View>
        </View>
      ))}
    </View>
  );

  const renderMessage = ({ item }: { item: ChatMsg }) => {
    const isUser = item.role === "user";
    const isSystem = item.role === "system";

    if (isSystem) {
      return (
        <View style={styles.systemMsgWrap}>
          <LinearGradient colors={["#059669", "#047857"]} style={styles.systemMsgBox}>
            <CheckCircle2 size={20} color="#fff" />
            <Text style={styles.systemMsgText}>{item.content}</Text>
          </LinearGradient>
        </View>
      );
    }

    return (
      <View style={[styles.msgWrap, isUser ? styles.userWrap : styles.aiWrap]}>
        {!isUser && (
          <Animated.View style={[styles.aiAvatarBox, { transform: [{ scale: isLoading ? pulseAnim : 1 }] }]}>
            <LinearGradient colors={["#059669", "#047857"]} style={styles.aiAvatarGradient}>
              <Sparkles size={14} color="#fff" />
            </LinearGradient>
          </Animated.View>
        )}
        <View style={[
          styles.msgBubble,
          isUser
            ? styles.userBubble
            : [styles.aiBubble, { backgroundColor: isDark ? "#1E293B" : "#fff", borderColor: isDark ? "#334155" : "#E2E8F0" }]
        ]}>
          <Text style={[
            styles.msgText,
            isUser ? styles.userText : { color: colors.text }
          ]}>
            {item.content}
          </Text>

          {/* Show extracted data cards */}
          {item.parsedData && renderExtractedCards(item.parsedData)}

          <Text style={[styles.msgTime, isUser ? styles.userTime : { color: colors.textMuted }]}>
            {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </Text>
        </View>
        {isUser && (
          <View style={styles.userAvatarBox}>
            <User size={14} color="#fff" />
          </View>
        )}
      </View>
    );
  };

  const chips = isAr ? SUGGESTION_CHIPS_AR : SUGGESTION_CHIPS_EN;

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor="transparent" translucent />

      {/* Header */}
      <LinearGradient colors={["#064E3B", "#059669"]} style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={22} color="#fff" />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <View style={styles.headerAiIcon}>
            <Sparkles size={16} color="#FBBF24" />
          </View>
          <View>
            <Text style={styles.headerTitle}>
              {isAr ? "مساعد الملف الصحي" : "Health Profile Assistant"}
            </Text>
            <Text style={styles.headerSub}>
              {isAr ? "أخبرني عن حالتك الصحية ✨" : "Tell me about your health ✨"}
            </Text>
          </View>
        </View>
        
        {messages.length > 0 ? (
          <TouchableOpacity onPress={handleNewChat} style={styles.newChatBtn}>
            <Ionicons name="refresh" size={18} color="#fff" />
            <Text style={styles.newChatText}>{isAr ? "جديد" : "New"}</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.headerBadge}>
            <Zap size={14} color="#FBBF24" />
            <Text style={styles.headerBadgeText}>AI</Text>
          </View>
        )}
      </LinearGradient>

      {/* SAVED BADGE */}
      {savedSuccess && (
        <View style={styles.savedBanner}>
          <CheckCircle2 size={18} color="#fff" />
          <Text style={styles.savedBannerText}>
            {isAr ? "تم حفظ البيانات بنجاح SAVED ✅" : "Data SAVED Successfully ✅"}
          </Text>
        </View>
      )}

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={item => item.id.toString()}
        renderItem={renderMessage}
        contentContainerStyle={styles.listContent}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <View style={styles.emptyIconWrap}>
              <LinearGradient colors={["#059669", "#047857"]} style={styles.emptyIconGradient}>
                <Stethoscope size={40} color="#fff" />
              </LinearGradient>
              <View style={styles.emptyIconGlow} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.text }]}>
              {isAr ? "مرحباً! أنا مساعدك الصحي 🩺" : "Hello! I'm your Health Assistant 🩺"}
            </Text>
            <Text style={[styles.emptySub, { color: colors.textMuted }]}>
              {isAr
                ? "اكتب بياناتك الصحية بأي شكل وأنا هرتبها وأسجلها لك تلقائياً"
                : "Describe your health conditions in any way and I'll organize and save them automatically"}
            </Text>

            {/* Suggestion chips */}
            <View style={styles.chipsWrap}>
              <Text style={[styles.chipsTitle, { color: colors.textMuted }]}>
                {isAr ? "جرّب تقول:" : "Try saying:"}
              </Text>
              {chips.map((chip, i) => (
                <TouchableOpacity
                  key={i}
                  style={[styles.chip, { backgroundColor: isDark ? "#1E293B" : "#F0FDF4", borderColor: isDark ? "#334155" : "#BBF7D0" }]}
                  onPress={() => handleSend(chip)}
                  activeOpacity={0.7}
                >
                  <MessageCircle size={14} color="#059669" />
                  <Text style={[styles.chipText, { color: isDark ? "#10B981" : "#047857" }]}>{chip}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        }
      />

      {/* Loading */}
      {isLoading && (
        <View style={[styles.typingBar, { backgroundColor: isDark ? "#1E293B" : "#F0FDF4" }]}>
          <ActivityIndicator size="small" color="#059669" />
          <Text style={[styles.typingText, { color: colors.textMuted }]}>
            {isAr ? "الذكاء الاصطناعي بيحلل بياناتك..." : "AI is analyzing your data..."}
          </Text>
        </View>
      )}

      {/* Save Bar */}
      {pendingData && totalExtracted > 0 && !savedSuccess && (
        <View style={[styles.saveBar, { backgroundColor: isDark ? "#1E293B" : "#fff" }]}>
          <View style={styles.saveBarInfo}>
            <ShieldCheck size={20} color="#059669" />
            <Text style={[styles.saveBarText, { color: colors.text }]}>
              {isAr
                ? `${totalExtracted} عنصر جاهز للحفظ`
                : `${totalExtracted} items ready to save`}
            </Text>
          </View>
          <TouchableOpacity onPress={() => handleSaveAll()} disabled={isSaving} activeOpacity={0.8}>
            <LinearGradient colors={["#059669", "#047857"]} style={styles.saveBtn}>
              {isSaving ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <>
                  <CheckCircle2 size={18} color="#fff" />
                  <Text style={styles.saveBtnText}>
                    {isAr ? "حفظ الكل" : "Save All"}
                  </Text>
                </>
              )}
            </LinearGradient>
          </TouchableOpacity>
        </View>
      )}

      {/* Input */}
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
      >
        <View style={[styles.inputBar, { backgroundColor: isDark ? "#0F172A" : "#fff", borderTopColor: isDark ? "#1E293B" : "#E2E8F0" }]}>
          <View style={[styles.inputBox, { backgroundColor: isDark ? "#1E293B" : "#F1F5F9" }]}>
            <TextInput
              style={[styles.input, { color: colors.text, textAlign: isAr ? "right" : "left" }]}
              placeholder={isAr ? "اكتب حالتك الصحية هنا..." : "Describe your health here..."}
              placeholderTextColor={isDark ? "#4B5563" : "#94A3B8"}
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={1000}
            />
            <TouchableOpacity
              style={[styles.sendBtn, !inputText.trim() && { opacity: 0.4 }]}
              onPress={() => handleSend()}
              disabled={!inputText.trim() || isLoading}
            >
              <LinearGradient colors={["#059669", "#047857"]} style={styles.sendBtnGradient}>
                <Send size={18} color="#fff" />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
      <Toast />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },

  // Header
  header: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingTop: Platform.OS === "ios" ? 56 : 44, paddingBottom: 14,
    paddingHorizontal: 16, elevation: 8,
    shadowColor: "#064E3B", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 12,
  },
  backBtn: { width: 38, height: 38, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.15)", justifyContent: "center", alignItems: "center" },
  headerCenter: { flex: 1, flexDirection: "row", alignItems: "center", marginLeft: 12 },
  headerAiIcon: {
    width: 36, height: 36, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.15)",
    justifyContent: "center", alignItems: "center", marginRight: 10,
  },
  headerTitle: { color: "#fff", fontSize: 17, fontWeight: "700" },
  headerSub: { color: "rgba(255,255,255,0.7)", fontSize: 12, fontWeight: "500", marginTop: 1 },
  headerBadge: {
    flexDirection: "row", alignItems: "center", backgroundColor: "rgba(251,191,36,0.2)",
    paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20, gap: 4,
  },
  headerBadgeText: { color: "#FBBF24", fontSize: 12, fontWeight: "800" },
  newChatBtn: {
    flexDirection: "row", alignItems: "center", backgroundColor: "rgba(255,255,255,0.2)",
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, gap: 4,
  },
  newChatText: { color: "#fff", fontSize: 12, fontWeight: "700" },
  savedBanner: {
    backgroundColor: "#059669", paddingVertical: 10, paddingHorizontal: 16,
    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8,
  },
  savedBannerText: { color: "#fff", fontSize: 14, fontWeight: "700" },

  // Messages
  listContent: { padding: 16, paddingBottom: 24 },
  msgWrap: { flexDirection: "row", marginBottom: 16, maxWidth: "88%" },
  userWrap: { alignSelf: "flex-end" },
  aiWrap: { alignSelf: "flex-start" },
  aiAvatarBox: { marginRight: 8, marginTop: 4 },
  aiAvatarGradient: { width: 30, height: 30, borderRadius: 10, justifyContent: "center", alignItems: "center" },
  userAvatarBox: {
    width: 30, height: 30, borderRadius: 10, backgroundColor: "#6366F1",
    justifyContent: "center", alignItems: "center", marginLeft: 8, marginTop: 4,
  },
  msgBubble: { padding: 14, borderRadius: 18, maxWidth: width * 0.72 },
  userBubble: { backgroundColor: "#059669", borderBottomRightRadius: 4 },
  aiBubble: { borderBottomLeftRadius: 4, borderWidth: 1 },
  msgText: { fontSize: 15, lineHeight: 23 },
  userText: { color: "#fff" },
  msgTime: { fontSize: 10, marginTop: 6, alignSelf: "flex-end" },
  userTime: { color: "rgba(255,255,255,0.6)" },

  // System messages
  systemMsgWrap: { alignItems: "center", marginVertical: 12, paddingHorizontal: 20 },
  systemMsgBox: {
    flexDirection: "row", alignItems: "center", gap: 10,
    paddingHorizontal: 20, paddingVertical: 14, borderRadius: 16,
  },
  systemMsgText: { color: "#fff", fontSize: 14, fontWeight: "600", flex: 1, lineHeight: 20 },

  // Extracted data cards
  cardsContainer: { marginTop: 12, gap: 8 },
  extractedCard: {
    flexDirection: "row", alignItems: "center", padding: 12, borderRadius: 14,
    borderWidth: 1, gap: 10,
  },
  cardIconBox: { width: 36, height: 36, borderRadius: 10, justifyContent: "center", alignItems: "center" },
  cardLabel: { fontSize: 11, fontWeight: "700", textTransform: "uppercase", letterSpacing: 0.5 },
  cardValue: { fontSize: 14, fontWeight: "600", marginTop: 2 },
  cardSub: { fontSize: 12, marginTop: 1 },
  cardMeta: { fontSize: 11, marginTop: 3 },

  // Empty state
  emptyContainer: { alignItems: "center", paddingTop: 60, paddingHorizontal: 30 },
  emptyIconWrap: { position: "relative", marginBottom: 24 },
  emptyIconGradient: { width: 88, height: 88, borderRadius: 28, justifyContent: "center", alignItems: "center" },
  emptyIconGlow: {
    position: "absolute", width: 110, height: 110, borderRadius: 55,
    backgroundColor: "rgba(5,150,105,0.1)", top: -11, left: -11,
  },
  emptyTitle: { fontSize: 22, fontWeight: "800", textAlign: "center", marginBottom: 10 },
  emptySub: { fontSize: 14, textAlign: "center", lineHeight: 22, marginBottom: 28 },

  // Suggestion chips
  chipsWrap: { width: "100%", gap: 8 },
  chipsTitle: { fontSize: 13, fontWeight: "600", marginBottom: 4 },
  chip: {
    flexDirection: "row", alignItems: "center", paddingHorizontal: 16, paddingVertical: 12,
    borderRadius: 14, borderWidth: 1, gap: 8,
  },
  chipText: { fontSize: 14, fontWeight: "500", flex: 1 },

  // Typing indicator
  typingBar: {
    flexDirection: "row", alignItems: "center", paddingHorizontal: 20, paddingVertical: 10, gap: 10,
    marginHorizontal: 16, borderRadius: 12, marginBottom: 4,
  },
  typingText: { fontSize: 13, fontStyle: "italic" },

  // Save bar
  saveBar: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingHorizontal: 16, paddingVertical: 12, borderTopWidth: 1, borderTopColor: "#E2E8F0",
    elevation: 4, shadowColor: "#000", shadowOffset: { width: 0, height: -2 }, shadowOpacity: 0.08, shadowRadius: 8,
  },
  saveBarInfo: { flexDirection: "row", alignItems: "center", gap: 8 },
  saveBarText: { fontSize: 14, fontWeight: "600" },
  saveBtn: {
    flexDirection: "row", alignItems: "center", gap: 6,
    paddingHorizontal: 20, paddingVertical: 12, borderRadius: 14,
  },
  saveBtnText: { color: "#fff", fontSize: 14, fontWeight: "700" },

  // Input bar
  inputBar: { padding: 12, borderTopWidth: 1, paddingBottom: Platform.OS === "ios" ? 30 : 12 },
  inputBox: { flexDirection: "row", alignItems: "flex-end", borderRadius: 20, paddingHorizontal: 14, paddingVertical: 6 },
  input: { flex: 1, minHeight: 40, maxHeight: 100, fontSize: 15, paddingHorizontal: 8, paddingTop: 10 },
  sendBtn: { marginLeft: 8, marginBottom: 4 },
  sendBtnGradient: { width: 40, height: 40, borderRadius: 14, justifyContent: "center", alignItems: "center" },
});
