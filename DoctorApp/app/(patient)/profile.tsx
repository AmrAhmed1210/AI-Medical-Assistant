import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, TextInput, Modal, Alert, ActivityIndicator, Image,
} from "react-native";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useState, useCallback } from "react";
import * as ImagePicker from "expo-image-picker";
import { useLanguage } from "../../hooks/useLanguage";

interface Booking {
  id: string; doctorName: string; specialty: string;
  date: string; time: string; paymentMethod: "visa"|"cash"; bookedAt: string;
}
interface UserData { firstName: string; lastName: string; email: string; phone: string; }
interface Medicine { name: string; dose: string; frequency: string; duration: string; }
interface MedicalHistory {
  chronicDiseases: string[];
  surgeries: string[];
  scannedMeds: Medicine[];
}

// ── Rule-based parser (no API needed) ──────────────────────────────────────
const DRUG_NAMES = [
  "paracetamol","amoxicillin","ibuprofen","omeprazole","metformin",
  "aspirin","cetirizine","atorvastatin","metronidazole","doxycycline",
  "vitamin d","calcium","iron","folic acid","zinc","pantoprazole",
  "clarithromycin","levothyroxine","amlodipine","omega-3",
];

function parsePrescriptionText(text: string): Medicine[] {
  const lower = text.toLowerCase();
  const name  = DRUG_NAMES.find((d) => lower.includes(d))?.replace(/\b\w/g, (c) => c.toUpperCase()) ?? "Unknown";
  const doseM = text.match(/(\d+\.?\d*)\s*(mg|mcg|ml|g|iu|%)/i);
  const dose  = doseM ? doseM[0] : "N/A";
  const freqMap: [RegExp, string][] = [
    [/once\s+daily|1\s*x\s*daily|od/i,       "Once daily"],
    [/twice\s+daily|2\s*x\s*daily|bd/i,       "Twice daily"],
    [/3\s+times\s+daily|tds|tid|3x/i,         "3 times daily"],
    [/every\s+8\s+hours|q8h/i,                "Every 8 hours"],
    [/every\s+12\s+hours|q12h/i,              "Every 12 hours"],
    [/at\s+night|before\s+sleep|bedtime/i,    "At night"],
    [/when\s+needed|as\s+needed|prn/i,        "When needed"],
  ];
  const freq  = freqMap.find(([re]) => re.test(text))?.[1] ?? "N/A";
  const durM  = text.match(/(\d+)\s*(day|days|week|weeks|month|months)/i);
  const dur   = durM ? durM[0] : "N/A";
  return [{ name, dose, frequency: freq, duration: dur }];
}
// ───────────────────────────────────────────────────────────────────────────

export default function ProfileScreen() {
  const router = useRouter();
  const { tr, isRTL, lang, switchLanguage } = useLanguage();
  const [userData,  setUserData]  = useState<UserData>({ firstName:"", lastName:"", email:"", phone:"" });
  const [bookings,  setBookings]  = useState<Booking[]>([]);
  const [editModal, setEditModal] = useState(false);
  const [editData,  setEditData]  = useState<UserData>({ firstName:"", lastName:"", email:"", phone:"" });
  const [saving,    setSaving]    = useState(false);
  const [errors,    setErrors]    = useState<any>({});
  const [activeTab, setActiveTab] = useState<"info"|"bookings"|"scan"|"history">("info");

  // Medical History state
  const [history,       setHistory]       = useState<MedicalHistory>({ chronicDiseases: [], surgeries: [], scannedMeds: [] });
  const [historyModal,  setHistoryModal]  = useState<null|"chronic"|"surgery">(null);
  const [historyInput,  setHistoryInput]  = useState("");

  // Scan state
  const [scanImage,   setScanImage]   = useState<string|null>(null);
  const [scanResult,  setScanResult]  = useState<Medicine[]|null>(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [rawOCR,      setRawOCR]      = useState<string>("");

  useFocusEffect(useCallback(() => { loadAll(); }, []));

  const loadAll = async () => {
    try {
      const raw = await AsyncStorage.getItem("user");
      if (raw) {
        const data      = JSON.parse(raw);
        const firstName = data.firstName ?? data.name?.split(" ")[0] ?? "";
        const lastName  = data.lastName  ?? data.name?.split(" ").slice(1).join(" ") ?? "";
        setUserData({ firstName, lastName, email: data.email ?? "", phone: data.phone ?? "" });
      }
      const userEmail  = (raw ? JSON.parse(raw).email : null) ?? "guest";
      const bookingKey = `bookings_${userEmail}`;
      const bRaw = await AsyncStorage.getItem(bookingKey);
      setBookings(bRaw ? JSON.parse(bRaw) : []);

      // Load medical history
      const hRaw = await AsyncStorage.getItem(`history_${userEmail}`);
      setHistory(hRaw ? JSON.parse(hRaw) : { chronicDiseases: [], surgeries: [], scannedMeds: [] });
    } catch {}
  };

  const openEdit  = () => { setEditData({ ...userData }); setErrors({}); setEditModal(true); };
  const validate  = () => {
    const e: any = {};
    if (!editData.firstName.trim()) e.firstName = "Required";
    if (!editData.email.includes("@")) e.email = "Invalid email";
    setErrors(e);
    return Object.keys(e).length === 0;
  };
  const saveProfile = async () => {
    if (!validate()) return;
    setSaving(true);
    try {
      const raw     = await AsyncStorage.getItem("user");
      const base    = raw ? JSON.parse(raw) : {};
      const updated = { ...base, ...editData, name: `${editData.firstName} ${editData.lastName}`.trim() };
      await AsyncStorage.setItem("user",     JSON.stringify(updated));
      await AsyncStorage.setItem("userName", updated.name);
      setUserData(editData);
      setEditModal(false);
    } finally { setSaving(false); }
  };

  const cancelBooking = (id: string) => {
    Alert.alert(tr("cancel_booking"), tr("cancel_booking_confirm"), [
      { text: tr("no"), style: "cancel" },
      { text: tr("yes_cancel"), style: "destructive", onPress: async () => {
        const updated = bookings.filter((b) => b.id !== id);
        setBookings(updated);
        const userRaw    = await AsyncStorage.getItem("user");
        const userEmail  = userRaw ? (JSON.parse(userRaw).email ?? "guest") : "guest";
        await AsyncStorage.setItem(`bookings_${userEmail}`, JSON.stringify(updated));
      }},
    ]);
  };

  const handleLogout = () => {
    Alert.alert(tr("logout"), tr("logout_confirm"), [
      { text: tr("cancel"), style: "cancel" },
      { text: tr("logout"), style: "destructive", onPress: async () => {
        await AsyncStorage.multiRemove(["isLoggedIn","token","userToken","user","userName","userRole"]);
        router.replace("/(auth)");
      }},
    ]);
  };

  // ── Scan handlers ──────────────────────────────────────────────────────────
  const pickFromGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") { Alert.alert("Permission needed", "Please allow access to your photo library."); return; }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.8 });
    if (!result.canceled) { setScanImage(result.assets[0].uri); setScanResult(null); setRawOCR(""); }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") { Alert.alert("Permission needed", "Please allow camera access."); return; }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) { setScanImage(result.assets[0].uri); setScanResult(null); setRawOCR(""); }
  };

  const analyzeImage = async () => {
    if (!scanImage) return;
    setScanLoading(true);
    try {
      // TODO: replace with real TrOCR API call when backend is ready
      // const form = new FormData();
      // form.append("file", { uri: scanImage, type: "image/jpeg", name: "presc.jpg" } as any);
      // const res = await axios.post(`${API_URL}/ocr/scan`, form, { headers: { "Content-Type": "multipart/form-data" } });
      // setRawOCR(res.data.text);
      // setScanResult(res.data.medicines);

      await new Promise((r) => setTimeout(r, 1500));
      const mockOCR = "Paracetamol 500mg twice daily for 7 days";
      const parsed = parsePrescriptionText(mockOCR);
      setRawOCR(mockOCR);
      setScanResult(parsed);
      // Save to medical history
      const raw2 = await AsyncStorage.getItem("user");
      const email = raw2 ? (JSON.parse(raw2).email ?? "guest") : "guest";
      const hRaw2 = await AsyncStorage.getItem(`history_${email}`);
      const hist2: MedicalHistory = hRaw2 ? JSON.parse(hRaw2) : { chronicDiseases: [], surgeries: [], scannedMeds: [] };
      const updatedHist = { ...hist2, scannedMeds: [...parsed, ...hist2.scannedMeds] };
      await AsyncStorage.setItem(`history_${email}`, JSON.stringify(updatedHist));
      setHistory(updatedHist);
    } catch {
      Alert.alert("Error", "Could not analyze image. Please try again.");
    } finally {
      setScanLoading(false);
    }
  };

  const saveHistory = async (updated: MedicalHistory) => {
    const raw = await AsyncStorage.getItem("user");
    const email = raw ? (JSON.parse(raw).email ?? "guest") : "guest";
    await AsyncStorage.setItem(`history_${email}`, JSON.stringify(updated));
    setHistory(updated);
  };

  const addHistoryItem = async () => {
    if (!historyInput.trim() || !historyModal) return;
    const updated = { ...history };
    if (historyModal === "chronic") updated.chronicDiseases = [...history.chronicDiseases, historyInput.trim()];
    else updated.surgeries = [...history.surgeries, historyInput.trim()];
    await saveHistory(updated);
    setHistoryInput(""); setHistoryModal(null);
  };

  const removeHistoryItem = async (type: "chronic"|"surgery"|"med", index: number) => {
    const updated = { ...history };
    if (type === "chronic") updated.chronicDiseases = history.chronicDiseases.filter((_, i) => i !== index);
    else if (type === "surgery") updated.surgeries = history.surgeries.filter((_, i) => i !== index);
    else updated.scannedMeds = history.scannedMeds.filter((_, i) => i !== index);
    await saveHistory(updated);
  };

  const clearScan = () => { setScanImage(null); setScanResult(null); setRawOCR(""); };
  // ───────────────────────────────────────────────────────────────────────────

  const fullName = [userData.firstName, userData.lastName].filter(Boolean).join(" ") || "User";
  const initials = fullName.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Hero */}
      <View style={styles.hero}>
        <View style={styles.heroTop}>
          <TouchableOpacity onPress={() => router.back()} style={styles.heroBtn}>
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.heroTitle}>{tr("my_profile")}</Text>
          <TouchableOpacity onPress={openEdit} style={styles.heroBtn}>
            <Ionicons name="pencil-outline" size={20} color="#fff" />
          </TouchableOpacity>
        </View>
        {/* Language toggle */}
        <TouchableOpacity
          onPress={() => switchLanguage(lang === "en" ? "ar" : "en")}
          style={styles.langToggle}
        >
          <Ionicons name="language-outline" size={14} color="#fff" />
          <Text style={styles.langToggleTxt}>{lang === "en" ? "العربية" : "English"}</Text>
        </TouchableOpacity>
        <View style={styles.avatarCircle}>
          <Text style={styles.avatarTxt}>{initials || "U"}</Text>
        </View>
        <Text style={styles.heroName}>{fullName}</Text>
        <Text style={styles.heroEmail}>{userData.email || "—"}</Text>
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{bookings.length}</Text>
            <Text style={styles.statLbl}>Bookings</Text>
          </View>
          <View style={styles.statDiv} />
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{bookings.filter(b => b.paymentMethod === "visa").length}</Text>
            <Text style={styles.statLbl}>Paid Online</Text>
          </View>
          <View style={styles.statDiv} />
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{bookings.filter(b => b.paymentMethod === "cash").length}</Text>
            <Text style={styles.statLbl}>Pay on Arrival</Text>
          </View>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {(["info","bookings","scan","history"] as const).map((tab) => (
          <TouchableOpacity
            key={tab}
            style={[styles.tab, activeTab === tab && styles.tabActive]}
            onPress={() => setActiveTab(tab)}
          >
            <Ionicons
              name={tab === "info" ? "person-outline" : tab === "bookings" ? "calendar-outline" : tab === "scan" ? "scan-outline" : "document-text-outline"}
              size={15}
              color={activeTab === tab ? COLORS.primary : "#AAA"}
            />
            <Text style={[styles.tabTxt, activeTab === tab && styles.tabTxtActive]}>
              {tab === "info" ? tr("info") : tab === "bookings" ? `${tr("bookings")}${bookings.length > 0 ? ` (${bookings.length})` : ""}` : tab === "scan" ? tr("scan_rx") : tr("history")}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40, paddingHorizontal: 18 }}>

        {/* Info Tab */}
        {activeTab === "info" && (
          <View style={{ marginTop: 16 }}>
            <View style={styles.card}>
              <InfoRow icon="person-outline" label={tr("first_name")} value={userData.firstName || "—"} isRTL={isRTL} />
              <Divider />
              <InfoRow icon="person-outline" label={tr("last_name")}  value={userData.lastName  || "—"} isRTL={isRTL} />
              <Divider />
              <InfoRow icon="mail-outline"   label={tr("email")}      value={userData.email     || "—"} isRTL={isRTL} />
              <Divider />
              <InfoRow icon="call-outline"   label={tr("phone")}      value={userData.phone     || "—"} isRTL={isRTL} />
            </View>
            <TouchableOpacity style={styles.editFullBtn} onPress={openEdit}>
              <Ionicons name="pencil" size={16} color="#fff" />
              <Text style={styles.editFullBtnTxt}>{tr("edit_profile")}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
              <Ionicons name="log-out-outline" size={18} color="#FF4444" />
              <Text style={styles.logoutTxt}>{tr("logout")}</Text>
            </TouchableOpacity>
            <Text style={styles.version}>{tr("version")}</Text>
          </View>
        )}

        {/* Bookings Tab */}
        {activeTab === "bookings" && (
          <View style={{ marginTop: 16 }}>
            {bookings.length === 0 ? (
              <View style={styles.emptyWrap}>
                <Ionicons name="calendar-outline" size={60} color="#DDD" />
                <Text style={styles.emptyTxt}>{tr("no_bookings")}</Text>
                <TouchableOpacity style={styles.browseBtn} onPress={() => router.push("/(patient)/doctors")}>
                  <Text style={styles.browseBtnTxt}>{tr("browse_doctors")}</Text>
                </TouchableOpacity>
              </View>
            ) : bookings.map((b) => (
              <View key={b.id} style={styles.bookingCard}>
                <View style={styles.bookingAccent} />
                <View style={{ flex: 1 }}>
                  <Text style={[styles.bookingDoc, isRTL && { textAlign: "right" }]}>{b.doctorName}</Text>
                  <Text style={[styles.bookingSpec, isRTL && { textAlign: "right" }]}>{b.specialty}</Text>
                  <View style={styles.metaRow}>
                    <MetaChip icon="calendar-outline" label={b.date} />
                    <MetaChip icon="time-outline"     label={b.time} />
                    <MetaChip icon={b.paymentMethod === "visa" ? "card-outline" : "cash-outline"} label={b.paymentMethod === "visa" ? tr("paid_online") : tr("pay_on_arrival")} />
                  </View>
                </View>
                <TouchableOpacity style={styles.cancelBtn} onPress={() => cancelBooking(b.id)}>
                  <Text style={styles.cancelTxt}>{tr("cancel")}</Text>
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}

        {/* Scan Tab */}
        {activeTab === "scan" && (
          <View style={{ marginTop: 16 }}>
            <View style={styles.scanHeaderCard}>
              <View style={styles.scanHeaderIcon}>
                <Ionicons name="scan" size={28} color={COLORS.primary} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.scanHeaderTitle, isRTL && { textAlign: "right" }]}>{tr("prescription_scanner")}</Text>
                <Text style={[styles.scanHeaderSub, isRTL && { textAlign: "right" }]}>{tr("scan_subtitle")}</Text>
              </View>
            </View>

            {!scanImage && (
              <View style={styles.scanPickRow}>
                <TouchableOpacity style={styles.scanPickBtn} onPress={takePhoto}>
                  <Ionicons name="camera" size={26} color={COLORS.primary} />
                  <Text style={styles.scanPickTxt}>{tr("camera")}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.scanPickBtn} onPress={pickFromGallery}>
                  <Ionicons name="images" size={26} color={COLORS.primary} />
                  <Text style={styles.scanPickTxt}>{tr("gallery")}</Text>
                </TouchableOpacity>
              </View>
            )}

            {scanImage && (
              <View style={styles.scanPreviewWrap}>
                <Image source={{ uri: scanImage }} style={styles.scanPreviewImg} resizeMode="contain" />
                <TouchableOpacity style={styles.scanClearBtn} onPress={clearScan}>
                  <Ionicons name="close-circle" size={28} color="#FF4444" />
                </TouchableOpacity>
              </View>
            )}

            {scanImage && !scanResult && (
              <TouchableOpacity style={[styles.analyzeBtn, scanLoading && { opacity: 0.7 }]} onPress={analyzeImage} disabled={scanLoading}>
                {scanLoading
                  ? <><ActivityIndicator color="#fff" size="small" /><Text style={styles.analyzeBtnTxt}>{tr("analyzing")}</Text></>
                  : <><Ionicons name="flash" size={18} color="#fff" /><Text style={styles.analyzeBtnTxt}>{tr("analyze")}</Text></>
                }
              </TouchableOpacity>
            )}

            {rawOCR !== "" && (
              <View style={styles.rawOcrCard}>
                <Text style={styles.rawOcrLabel}>{tr("raw_ocr")}</Text>
                <Text style={styles.rawOcrText}>{rawOCR}</Text>
              </View>
            )}

            {scanResult && (
              <View style={{ marginTop: 4 }}>
                <Text style={styles.resultTitle}>{tr("detected_medicines")}</Text>
                {scanResult.map((med, i) => (
                  <View key={i} style={styles.medCard}>
                    <View style={styles.medIconWrap}>
                      <Ionicons name="medical" size={20} color={COLORS.primary} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.medName}>{med.name}</Text>
                      <View style={styles.medMetaRow}>
                        <MedChip icon="flask-outline"    label={med.dose} />
                        <MedChip icon="time-outline"     label={med.frequency} />
                        <MedChip icon="calendar-outline" label={med.duration} />
                      </View>
                    </View>
                  </View>
                ))}
                <TouchableOpacity style={styles.scanAgainBtn} onPress={clearScan}>
                  <Ionicons name="refresh" size={16} color={COLORS.primary} />
                  <Text style={styles.scanAgainTxt}>{tr("scan_another")}</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {/* Medical History Tab */}
        {activeTab === "history" && (
          <View style={{ marginTop: 16, gap: 16 }}>

            {/* Chronic Diseases */}
            <View style={styles.histSection}>
              <View style={styles.histSectionHeader}>
                <View style={styles.histIconWrap}>
                  <Ionicons name="fitness-outline" size={18} color={COLORS.primary} />
                </View>
                <Text style={styles.histSectionTitle}>{tr("chronic_diseases")}</Text>
                <TouchableOpacity style={styles.histAddBtn} onPress={() => { setHistoryInput(""); setHistoryModal("chronic"); }}>
                  <Ionicons name="add" size={18} color="#fff" />
                </TouchableOpacity>
              </View>
              {history.chronicDiseases.length === 0 ? (
                <Text style={styles.histEmpty}>{tr("no_chronic")}</Text>
              ) : history.chronicDiseases.map((item, i) => (
                <View key={i} style={styles.histItem}>
                  <Ionicons name="ellipse" size={8} color={COLORS.primary} style={{ marginTop: 4 }} />
                  <Text style={styles.histItemTxt}>{item}</Text>
                  <TouchableOpacity onPress={() => removeHistoryItem("chronic", i)}>
                    <Ionicons name="close-circle-outline" size={18} color="#FF4444" />
                  </TouchableOpacity>
                </View>
              ))}
            </View>

            {/* Surgeries & Diagnoses */}
            <View style={styles.histSection}>
              <View style={styles.histSectionHeader}>
                <View style={styles.histIconWrap}>
                  <Ionicons name="medkit-outline" size={18} color="#FF6B6B" />
                </View>
                <Text style={styles.histSectionTitle}>{tr("surgeries_diagnoses")}</Text>
                <TouchableOpacity style={[styles.histAddBtn, { backgroundColor: "#FF6B6B" }]} onPress={() => { setHistoryInput(""); setHistoryModal("surgery"); }}>
                  <Ionicons name="add" size={18} color="#fff" />
                </TouchableOpacity>
              </View>
              {history.surgeries.length === 0 ? (
                <Text style={styles.histEmpty}>{tr("no_surgeries")}</Text>
              ) : history.surgeries.map((item, i) => (
                <View key={i} style={styles.histItem}>
                  <Ionicons name="ellipse" size={8} color="#FF6B6B" style={{ marginTop: 4 }} />
                  <Text style={styles.histItemTxt}>{item}</Text>
                  <TouchableOpacity onPress={() => removeHistoryItem("surgery", i)}>
                    <Ionicons name="close-circle-outline" size={18} color="#FF4444" />
                  </TouchableOpacity>
                </View>
              ))}
            </View>

            {/* Scanned Medications */}
            <View style={styles.histSection}>
              <View style={styles.histSectionHeader}>
                <View style={[styles.histIconWrap, { backgroundColor: "#E8F5E9" }]}>
                  <Ionicons name="medical-outline" size={18} color="#4CAF50" />
                </View>
                <Text style={styles.histSectionTitle}>{tr("scanned_medications")}</Text>
                <TouchableOpacity style={[styles.histAddBtn, { backgroundColor: "#4CAF50" }]} onPress={() => setActiveTab("scan")}>
                  <Ionicons name="scan-outline" size={16} color="#fff" />
                </TouchableOpacity>
              </View>
              {history.scannedMeds.length === 0 ? (
                <Text style={styles.histEmpty}>{tr("no_scanned_meds")}</Text>
              ) : history.scannedMeds.map((med, i) => (
                <View key={i} style={styles.histMedCard}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.histMedName}>{med.name}</Text>
                    <Text style={styles.histMedMeta}>{med.dose}  ·  {med.frequency}  ·  {med.duration}</Text>
                  </View>
                  <TouchableOpacity onPress={() => removeHistoryItem("med", i)}>
                    <Ionicons name="close-circle-outline" size={18} color="#FF4444" />
                  </TouchableOpacity>
                </View>
              ))}
            </View>

          </View>
        )}
      </ScrollView>

      {/* Add History Item Modal */}
      <Modal visible={!!historyModal} transparent animationType="fade">
        <View style={styles.overlay}>
          <View style={[styles.editSheet, { paddingBottom: 30 }]}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>
              {historyModal === "chronic" ? tr("add_chronic") : tr("add_surgery")}
            </Text>
            <TextInput
              style={styles.input}
              placeholder={historyModal === "chronic" ? tr("chronic_placeholder") : tr("surgery_placeholder")}
              value={historyInput}
              onChangeText={setHistoryInput}
              autoFocus
            />
            <TouchableOpacity style={styles.saveBtn} onPress={addHistoryItem}>
              <Text style={styles.saveBtnTxt}>{tr("add")}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.cancelModalBtn} onPress={() => setHistoryModal(null)}>
              <Text style={styles.cancelModalTxt}>{tr("cancel")}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Edit Modal */}
      <Modal visible={editModal} transparent animationType="slide">
        <View style={styles.overlay}>
          <View style={styles.editSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Edit Profile</Text>
            <View style={{ flexDirection: "row", gap: 10 }}>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLbl}>First Name</Text>
                <TextInput style={[styles.input, errors.firstName && styles.inputErr]} value={editData.firstName} onChangeText={(v) => setEditData({ ...editData, firstName: v })} placeholder="First name" placeholderTextColor="#BBB" />
                {errors.firstName && <Text style={styles.errTxt}>{errors.firstName}</Text>}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLbl}>Last Name</Text>
                <TextInput style={styles.input} value={editData.lastName} onChangeText={(v) => setEditData({ ...editData, lastName: v })} placeholder="Last name" placeholderTextColor="#BBB" />
              </View>
            </View>
            <Text style={styles.fieldLbl}>Email</Text>
            <TextInput style={[styles.input, errors.email && styles.inputErr]} value={editData.email} onChangeText={(v) => setEditData({ ...editData, email: v })} placeholder="Email" placeholderTextColor="#BBB" keyboardType="email-address" autoCapitalize="none" />
            {errors.email && <Text style={styles.errTxt}>{errors.email}</Text>}
            <Text style={styles.fieldLbl}>Phone</Text>
            <TextInput style={styles.input} value={editData.phone} onChangeText={(v) => setEditData({ ...editData, phone: v })} placeholder="Phone number" placeholderTextColor="#BBB" keyboardType="phone-pad" />
            <View style={{ flexDirection: "row", gap: 10, marginTop: 16 }}>
              <TouchableOpacity style={[styles.saveBtn, { backgroundColor: "#F0F0F0", flex: 1 }]} onPress={() => setEditModal(false)}>
                <Text style={[styles.saveBtnTxt, { color: "#555" }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.saveBtn, { flex: 2 }]} onPress={saveProfile} disabled={saving}>
                {saving ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.saveBtnTxt}>Save Changes</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

function InfoRow({ icon, label, value, isRTL }: { icon: any; label: string; value: string; isRTL?: boolean }) {
  return (
    <View style={[styles.infoRow, isRTL && { flexDirection: "row-reverse" }]}>
      <View style={styles.infoIcon}><Ionicons name={icon} size={15} color={COLORS.primary} /></View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.infoLbl, isRTL && { textAlign: "right" }]}>{label}</Text>
        <Text style={[styles.infoVal, isRTL && { textAlign: "right" }]}>{value}</Text>
      </View>
    </View>
  );
}
function MetaChip({ icon, label }: { icon: any; label: string }) {
  return (
    <View style={styles.metaChip}>
      <Ionicons name={icon} size={11} color={COLORS.primary} />
      <Text style={styles.metaChipTxt}>{label}</Text>
    </View>
  );
}
function MedChip({ icon, label }: { icon: any; label: string }) {
  return (
    <View style={styles.medChip}>
      <Ionicons name={icon} size={10} color="#666" />
      <Text style={styles.medChipTxt}>{label}</Text>
    </View>
  );
}
function Divider() {
  return <View style={{ height: 1, backgroundColor: "#F5F5F5", marginLeft: 48 }} />;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F4F6FA" },
  hero: { backgroundColor: COLORS.primary, alignItems: "center", paddingBottom: 24, borderBottomLeftRadius: 28, borderBottomRightRadius: 28 },
  heroTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", width: "100%", paddingHorizontal: 18, paddingTop: 52, paddingBottom: 16 },
  heroBtn:   { width: 38, height: 38, justifyContent: "center", alignItems: "center" },
  heroTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  langToggle: { flexDirection: "row", alignItems: "center", gap: 5, alignSelf: "center", backgroundColor: "rgba(255,255,255,0.2)", paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20, marginTop: -8, marginBottom: 8 },
  langToggleTxt: { fontSize: 12, color: "#fff", fontWeight: "600" },
  avatarCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: "#fff", justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: "rgba(255,255,255,0.35)", marginBottom: 10 },
  avatarTxt: { fontSize: 28, fontWeight: "800", color: COLORS.primary },
  heroName:  { fontSize: 20, fontWeight: "800", color: "#fff" },
  heroEmail: { fontSize: 12, color: "rgba(255,255,255,0.7)", marginTop: 2, marginBottom: 18 },
  statsRow: { flexDirection: "row", backgroundColor: "rgba(255,255,255,0.15)", borderRadius: 16, paddingVertical: 12, paddingHorizontal: 24, width: "88%" },
  statItem: { flex: 1, alignItems: "center" },
  statDiv:  { width: 1, backgroundColor: "rgba(255,255,255,0.25)" },
  statNum:  { fontSize: 18, fontWeight: "800", color: "#fff" },
  statLbl:  { fontSize: 10, color: "rgba(255,255,255,0.7)", marginTop: 2, textAlign: "center" },
  tabs: { flexDirection: "row", marginHorizontal: 18, marginTop: 16, backgroundColor: "#fff", borderRadius: 14, padding: 4, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 4, elevation: 2 },
  tab:         { flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 5, paddingVertical: 10, borderRadius: 11 },
  tabActive:   { backgroundColor: COLORS.primary + "15" },
  tabTxt:      { fontSize: 11, color: "#AAA", fontWeight: "500" },
  tabTxtActive:{ color: COLORS.primary, fontWeight: "700" },
  card: { backgroundColor: "#fff", borderRadius: 18, overflow: "hidden", shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2 },
  infoRow:  { flexDirection: "row", alignItems: "center", gap: 12, padding: 14 },
  infoIcon: { width: 34, height: 34, borderRadius: 10, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center" },
  infoLbl:  { fontSize: 10, color: "#AAA", marginBottom: 2 },
  infoVal:  { fontSize: 14, color: "#1A1A1A", fontWeight: "600" },
  editFullBtn:    { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 14, marginTop: 14 },
  editFullBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  logoutBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, borderRadius: 16, paddingVertical: 14, marginTop: 10, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#FFE5E5" },
  logoutTxt: { color: "#FF4444", fontSize: 15, fontWeight: "700" },
  version:   { textAlign: "center", color: "#CCC", fontSize: 11, marginTop: 18 },
  emptyWrap:    { alignItems: "center", paddingVertical: 48, gap: 10 },
  emptyTxt:     { fontSize: 14, color: "#BBB" },
  browseBtn:    { backgroundColor: COLORS.primary, paddingHorizontal: 22, paddingVertical: 10, borderRadius: 20, marginTop: 4 },
  browseBtnTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },
  bookingCard:   { backgroundColor: "#fff", borderRadius: 16, padding: 14, marginBottom: 10, flexDirection: "row", alignItems: "center", shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2, overflow: "hidden" },
  bookingAccent: { position: "absolute", left: 0, top: 0, bottom: 0, width: 4, backgroundColor: COLORS.primary },
  bookingDoc:    { fontSize: 14, fontWeight: "700", color: "#1A1A1A", marginLeft: 10 },
  bookingSpec:   { fontSize: 11, color: COLORS.primary, fontWeight: "500", marginTop: 2, marginLeft: 10 },
  metaRow:       { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 8, marginLeft: 10 },
  metaChip:      { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: COLORS.primary + "12", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 20 },
  metaChipTxt:   { fontSize: 10, color: "#555", fontWeight: "500" },
  cancelBtn:     { backgroundColor: "#FFF0F0", paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  cancelTxt:     { color: "#FF4444", fontSize: 11, fontWeight: "700" },
  // Scan
  scanHeaderCard:  { flexDirection: "row", alignItems: "flex-start", gap: 14, backgroundColor: COLORS.primary + "12", borderRadius: 18, padding: 16, marginBottom: 18 },
  scanHeaderIcon:  { width: 50, height: 50, borderRadius: 14, backgroundColor: "#fff", justifyContent: "center", alignItems: "center" },
  scanHeaderTitle: { fontSize: 15, fontWeight: "800", color: "#1A1A1A", marginBottom: 4 },
  scanHeaderSub:   { fontSize: 11, color: "#666", lineHeight: 16 },
  scanPickRow: { flexDirection: "row", gap: 14, marginBottom: 6 },
  scanPickBtn: { flex: 1, backgroundColor: "#fff", borderRadius: 18, paddingVertical: 22, alignItems: "center", gap: 8, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 8, elevation: 2 },
  scanPickTxt: { fontSize: 13, fontWeight: "700", color: COLORS.primary },
  scanPreviewWrap: { position: "relative", borderRadius: 18, overflow: "hidden", marginBottom: 14, backgroundColor: "#fff" },
  scanPreviewImg:  { width: "100%", height: 200, borderRadius: 18 },
  scanClearBtn:    { position: "absolute", top: 10, right: 10 },
  analyzeBtn:    { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 15, marginBottom: 14 },
  analyzeBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  rawOcrCard:  { backgroundColor: "#F7F7F7", borderRadius: 14, padding: 14, marginBottom: 14 },
  rawOcrLabel: { fontSize: 10, fontWeight: "700", color: "#999", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.8 },
  rawOcrText:  { fontSize: 13, color: "#444", lineHeight: 20 },
  resultTitle: { fontSize: 15, fontWeight: "800", color: "#1A1A1A", marginBottom: 10 },
  medCard:     { backgroundColor: "#fff", borderRadius: 16, padding: 14, marginBottom: 10, flexDirection: "row", alignItems: "flex-start", gap: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2 },
  medIconWrap: { width: 40, height: 40, borderRadius: 12, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center" },
  medName:     { fontSize: 15, fontWeight: "700", color: "#1A1A1A", marginBottom: 8 },
  medMetaRow:  { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  medChip:     { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "#F4F4F4", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 20 },
  medChipTxt:  { fontSize: 10, color: "#555" },
  scanAgainBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, borderWidth: 1.5, borderColor: COLORS.primary, borderRadius: 16, paddingVertical: 12, marginTop: 8 },
  scanAgainTxt: { color: COLORS.primary, fontSize: 14, fontWeight: "700" },
  // Modal
  overlay:    { flex: 1, backgroundColor: "rgba(0,0,0,0.45)", justifyContent: "flex-end" },
  editSheet:  { backgroundColor: "#fff", borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 22, paddingTop: 14, paddingBottom: 36 },
  sheetHandle:{ width: 40, height: 4, borderRadius: 2, backgroundColor: "#DDD", alignSelf: "center", marginBottom: 18 },
  sheetTitle: { fontSize: 19, fontWeight: "800", color: "#1A1A1A", marginBottom: 14 },
  fieldLbl:   { fontSize: 12, fontWeight: "600", color: "#555", marginBottom: 6, marginTop: 10 },
  input:      { backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 14, color: "#1A1A1A", borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2 },
  inputErr:   { borderColor: "#e53935" },
  errTxt:     { fontSize: 11, color: "#e53935", marginTop: 2 },
  saveBtn:    { backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 14, alignItems: "center", justifyContent: "center", marginBottom: 10 },
  saveBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  cancelModalBtn: { borderRadius: 16, paddingVertical: 12, alignItems: "center", justifyContent: "center" },
  cancelModalTxt: { color: "#AAA", fontSize: 14, fontWeight: "600" },
  // Medical History
  histSection:      { backgroundColor: "#fff", borderRadius: 18, padding: 16, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2 },
  histSectionHeader:{ flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 12 },
  histIconWrap:     { width: 34, height: 34, borderRadius: 10, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center" },
  histSectionTitle: { flex: 1, fontSize: 15, fontWeight: "700", color: "#1A1A1A" },
  histAddBtn:       { width: 30, height: 30, borderRadius: 15, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center" },
  histEmpty:        { fontSize: 13, color: "#AAA", textAlign: "center", paddingVertical: 10 },
  histItem:         { flexDirection: "row", alignItems: "flex-start", gap: 10, paddingVertical: 8, borderTopWidth: 1, borderTopColor: "#F4F4F4" },
  histItemTxt:      { flex: 1, fontSize: 14, color: "#333", lineHeight: 20 },
  histMedCard:      { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 10, borderTopWidth: 1, borderTopColor: "#F4F4F4" },
  histMedName:      { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  histMedMeta:      { fontSize: 12, color: "#888", marginTop: 2 },
});