import React, { useState, useCallback } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, TextInput, Modal, Alert, ActivityIndicator, Image,
  RefreshControl,
} from "react-native";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as ImagePicker from "expo-image-picker";
import Toast from "react-native-toast-message";
import { COLORS } from "../../constants/colors";
import { useLanguage } from "../../context/LanguageContext";

// Import services
import { getMyProfile, updateMyProfile, Profile, uploadProfilePhoto } from "../../services/profileService";
import { getMyAppointments, cancelAppointment, Appointment } from "../../services/appointmentService";
import { getMyVisits, PatientVisit } from "../../services/visitService";
import { logout } from "../../services/authService";
import { getDoctorById } from "../../services/doctorService";
import { getFollowedDoctorIds } from "../../services/followService";
import { startSupportSession } from "../../services/sessionService";

// Types
interface Medicine {
  name: string;
  dose: string;
  frequency: string;
  duration: string;
}

interface MedicalHistory {
  chronicDiseases: string[];
  surgeries: string[];
  scannedMeds: Medicine[];
}

interface FollowedDoctor {
  id: number;
  name: string;
  specialty: string;
  photoUrl?: string;
}

// ── Rule-based parser ──────────────────────────────────────────────────────
const DRUG_NAMES = [
  "paracetamol", "amoxicillin", "ibuprofen", "omeprazole", "metformin",
  "aspirin", "cetirizine", "atorvastatin", "metronidazole", "doxycycline",
  "vitamin d", "calcium", "iron", "folic acid", "zinc", "pantoprazole",
  "clarithromycin", "levothyroxine", "amlodipine", "omega-3",
];

function parsePrescriptionText(text: string): Medicine[] {
  const lower = text.toLowerCase();
  const name = DRUG_NAMES.find((d) => lower.includes(d))?.replace(/\b\w/g, (c) => c.toUpperCase()) ?? "Unknown";
  const doseM = text.match(/(\d+\.?\d*)\s*(mg|mcg|ml|g|iu|%)/i);
  const dose = doseM ? doseM[0] : "N/A";
  const freqMap: [RegExp, string][] = [
    [/once\s+daily|1\s*x\s*daily|od/i, "Once daily"],
    [/twice\s+daily|2\s*x\s*daily|bd/i, "Twice daily"],
    [/3\s+times\s+daily|tds|tid|3x/i, "3 times daily"],
    [/every\s+8\s+hours|q8h/i, "Every 8 hours"],
    [/every\s+12\s+hours|q12h/i, "Every 12 hours"],
    [/at\s+night|before\s+sleep|bedtime/i, "At night"],
    [/when\s+needed|as\s+needed|prn/i, "When needed"],
  ];
  const freq = freqMap.find(([re]) => re.test(text))?.[1] ?? "N/A";
  const durM = text.match(/(\d+)\s*(day|days|week|weeks|month|months)/i);
  const dur = durM ? durM[0] : "N/A";
  return [{ name, dose, frequency: freq, duration: dur }];
}

export default function ProfileScreen() {
  const router = useRouter();
  const { tr, isRTL, lang, switchLanguage } = useLanguage();

  const [profile, setProfile] = useState<Profile | null>(null);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [followedDoctors, setFollowedDoctors] = useState<FollowedDoctor[]>([]);
  const [loadingFollowed, setLoadingFollowed] = useState(false);
  const [loading, setLoading] = useState(true);

  const [editModal, setEditModal] = useState(false);
  const [editName, setEditName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editPhone, setEditPhone] = useState("");
  const [editDateOfBirth, setEditDateOfBirth] = useState("");
  const [saving, setSaving] = useState(false);

  const [activeTab, setActiveTab] = useState<"info" | "bookings" | "visits" | "scan" | "history">("info");
  const [visits, setVisits] = useState<PatientVisit[]>([]);
  const [loadingVisits, setLoadingVisits] = useState(false);
  const [history, setHistory] = useState<MedicalHistory>({ chronicDiseases: [], surgeries: [], scannedMeds: [] });
  const [historyModal, setHistoryModal] = useState<null | "chronic" | "surgery">(null);
  const [historyInput, setHistoryInput] = useState("");

  const [scanImage, setScanImage] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<Medicine[] | null>(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [rawOCR, setRawOCR] = useState<string>("");

  const fetchFollowedDoctors = async () => {
    try {
      setLoadingFollowed(true);
      const ids = await getFollowedDoctorIds();
      if (ids.length === 0) {
        setFollowedDoctors([]);
        return;
      }
      const details = await Promise.all(
        ids.map(async (id) => {
          try {
            const doc: any = await getDoctorById(id);
            return {
              id: Number(doc.id ?? id),
              name: String(doc.name ?? "Doctor"),
              specialty: String(doc.specialty ?? "General"),
              photoUrl: doc.imageUrl || doc.photoUrl,
            } as FollowedDoctor;
          } catch { return null; }
        })
      );
      setFollowedDoctors(details.filter((d): d is FollowedDoctor => d !== null));
    } finally {
      setLoadingFollowed(false);
    }
  };

  const fetchVisits = async () => {
    try {
      setLoadingVisits(true);
      const data = await getMyVisits();
      setVisits(data);
    } catch {
      setVisits([]);
    } finally {
      setLoadingVisits(false);
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      const [prof, appts] = await Promise.all([
        getMyProfile(),
        getMyAppointments(),
        fetchFollowedDoctors(),
      ]);
      setProfile(prof);
      setAppointments(appts);
      setEditName(prof.name);
      setEditEmail(prof.email || "");
      setEditPhone(prof.phone || "");
      setEditDateOfBirth(prof.dateOfBirth || "");

      const userEmail = prof.email;
      const hRaw = await AsyncStorage.getItem(`history_${userEmail}`);
      setHistory(hRaw ? JSON.parse(hRaw) : { chronicDiseases: [], surgeries: [], scannedMeds: [] });

      // Load visits
      fetchVisits();
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to load data" });
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(useCallback(() => { fetchData(); }, []));

  const saveProfile = async () => {
    if (!editName.trim()) {
      Alert.alert("Error", "Name is required");
      return;
    }
    setSaving(true);
    try {
      await updateMyProfile(editName, editEmail, editPhone, editDateOfBirth);
      await AsyncStorage.setItem("userName", editName);
      await AsyncStorage.setItem("userEmail", editEmail);
      setProfile(prev => prev ? { ...prev, name: editName, email: editEmail, phone: editPhone, dateOfBirth: editDateOfBirth } : prev);
      setEditModal(false);
      Toast.show({ type: "success", text1: "Profile updated!" });
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message });
    } finally {
      setSaving(false);
    }
  };

  const cancelBooking = (id: number) => {
    Alert.alert(tr("cancel_booking"), tr("cancel_booking_confirm"), [
      { text: tr("no"), style: "cancel" },
      {
        text: tr("yes_cancel"), style: "destructive", onPress: async () => {
          try {
            await cancelAppointment(id);
            setAppointments(prev => prev.filter(a => a.id !== id));
            Toast.show({ type: "success", text1: "Appointment cancelled" });
          } catch (e: any) {
            Toast.show({ type: "error", text1: e.message });
          }
        }
      },
    ]);
  };

  const handleLogout = () => {
    Alert.alert(tr("logout"), tr("logout_confirm"), [
      { text: tr("cancel"), style: "cancel" },
      {
        text: tr("logout"), style: "destructive", onPress: async () => {
          await logout();
          router.replace("/(auth)");
        }
      },
    ]);
  };

  const handleContactSupport = async () => {
    setLoading(true);
    try {
      const session = await startSupportSession();
      router.push({
        pathname: "/(patient)/messages",
        params: {
          sessionId: String(session.id),
          doctorName: "Technical Support",
        }
      });
    } catch (e: any) {
      Toast.show({ type: "error", text1: "Failed to open support chat" });
    } finally {
      setLoading(false);
    }
  };

  const pickFromGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") return;
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.8 });
    if (!result.canceled) {
      setScanImage(result.assets[0].uri);
      setScanResult(null);
      setRawOCR("");
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") return;
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) {
      setScanImage(result.assets[0].uri);
      setScanResult(null);
      setRawOCR("");
    }
  };

  const handleUploadPhoto = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission Required", "Please allow gallery access to upload a photo.");
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled) {
        setLoading(true);
        const url = await uploadProfilePhoto(result.assets[0].uri);
        setProfile(prev => prev ? { ...prev, photoUrl: url } : prev);
        Toast.show({ type: "success", text1: "Photo uploaded!" });
      }
    } catch (e: any) {
      Toast.show({ type: "error", text1: "Upload failed" });
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!scanImage) return;
    setScanLoading(true);
    try {
      const OCR_URL = "http://192.168.1.6:8000/scan";
      const form = new FormData();
      form.append("file", { uri: scanImage, type: "image/jpeg", name: "presc.jpg" } as any);
      const res = await fetch(OCR_URL, { method: "POST", body: form, headers: { "Content-Type": "multipart/form-data" } });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      const rawText: string = data.text ?? "";
      const medicines: Medicine[] = data.medicines ?? parsePrescriptionText(rawText);
      setRawOCR(rawText);
      setScanResult(medicines);
      const email = profile?.email ?? "guest";
      const hRaw2 = await AsyncStorage.getItem(`history_${email}`);
      const hist2: MedicalHistory = hRaw2 ? JSON.parse(hRaw2) : { chronicDiseases: [], surgeries: [], scannedMeds: [] };
      const updatedHist = { ...hist2, scannedMeds: [...medicines, ...hist2.scannedMeds] };
      await AsyncStorage.setItem(`history_${email}`, JSON.stringify(updatedHist));
      setHistory(updatedHist);
    } catch {
      Alert.alert("Error", "Could not analyze image. Please try again.");
    } finally {
      setScanLoading(false);
    }
  };

  const saveHistory = async (updated: MedicalHistory) => {
    const email = profile?.email ?? "guest";
    await AsyncStorage.setItem(`history_${email}`, JSON.stringify(updated));
    setHistory(updated);
  };

  const addHistoryItem = async () => {
    if (!historyInput.trim() || !historyModal) return;
    const updated = { ...history };
    if (historyModal === "chronic") updated.chronicDiseases = [...history.chronicDiseases, historyInput.trim()];
    else updated.surgeries = [...history.surgeries, historyInput.trim()];
    await saveHistory(updated);
    setHistoryInput("");
    setHistoryModal(null);
  };

  const removeHistoryItem = async (type: "chronic" | "surgery" | "med", index: number) => {
    const updated = { ...history };
    if (type === "chronic") updated.chronicDiseases = history.chronicDiseases.filter((_, i) => i !== index);
    else if (type === "surgery") updated.surgeries = history.surgeries.filter((_, i) => i !== index);
    else updated.scannedMeds = history.scannedMeds.filter((_, i) => i !== index);
    await saveHistory(updated);
  };

  const clearScan = () => {
    setScanImage(null);
    setScanResult(null);
    setRawOCR("");
  };

  const fullName = profile?.name || "User";
  const initials = fullName.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color={COLORS.primary} /></View>;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Profile Header */}
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{tr("my_profile")}</Text>
          <TouchableOpacity onPress={() => setEditModal(true)}>
            <Ionicons name="settings-outline" size={20} color="#fff" />
          </TouchableOpacity>
        </View>

        <View style={styles.headerMain}>
          <TouchableOpacity style={styles.avatarCircle} onPress={handleUploadPhoto}>
            {profile?.photoUrl ? (
              <Image source={{ uri: profile.photoUrl }} style={styles.avatarImg} />
            ) : (
              <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png' }} style={styles.avatarImg} />
            )}
            <View style={styles.cameraBadge}>
              <Ionicons name="camera" size={12} color="#fff" />
            </View>
          </TouchableOpacity>
          <View style={styles.headerInfo}>
            <Text style={styles.heroName}>{fullName}</Text>
            <TouchableOpacity onPress={() => switchLanguage(lang === "en" ? "ar" : "en")} style={styles.langBadge}>
              <Text style={styles.langBadgeText}>{lang === "en" ? "العربية" : "English"}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statNum}>
              {appointments.filter(a => {
                const s = a.status?.toLowerCase();
                return s === "pending" || s === "confirmed";
              }).length}
            </Text>
            <Text style={styles.statLbl}>Active Bookings</Text>
          </View>
          <View style={styles.statLine} />
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{followedDoctors.length}</Text>
            <Text style={styles.statLbl}>Following</Text>
          </View>
          <View style={styles.statLine} />
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{history.chronicDiseases.length + history.surgeries.length}</Text>
            <Text style={styles.statLbl}>Medical Records</Text>
          </View>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabsWrap}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabsScroll}>
          {[
            { id: "info", icon: "person-outline", label: "Profile" },
            { id: "bookings", icon: "calendar-outline", label: "Bookings" },
            { id: "visits", icon: "medical-outline", label: "Visits" },
            { id: "scan", icon: "scan-outline", label: "Scan Rx" },
            { id: "history", icon: "document-text-outline", label: "Medical History" },
          ].map((tab) => (
            <TouchableOpacity
              key={tab.id}
              style={[styles.tabBtn, activeTab === tab.id && styles.tabBtnActive]}
              onPress={() => setActiveTab(tab.id as any)}
            >
              <Ionicons name={tab.icon as any} size={16} color={activeTab === tab.id ? COLORS.primary : "#94A3B8"} />
              <Text style={[styles.tabBtnTxt, activeTab === tab.id && styles.tabBtnTxtActive]}>{tab.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchData} colors={[COLORS.primary]} />}
      >
        {/* Info Tab */}
        {activeTab === "info" && (
          <View>
            {/* Followed Doctors - Highlighted */}
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Followed Doctors</Text>
              <TouchableOpacity onPress={() => router.push("/(patient)/doctors")}>
                <Text style={styles.seeAll}>Find more</Text>
              </TouchableOpacity>
            </View>

            {loadingFollowed ? (
              <ActivityIndicator color={COLORS.primary} style={{ marginVertical: 10 }} />
            ) : followedDoctors.length === 0 ? (
              <View style={styles.followedEmptyCard}>
                <Text style={styles.followedEmptyText}>Keep track of your favorite doctors here.</Text>
              </View>
            ) : (
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.followedList}>
                {followedDoctors.map(doc => (
                  <TouchableOpacity key={doc.id} style={styles.circleDoc} onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doc.id) } })}>
                    <View style={styles.circleAvatar}>
                      {doc.photoUrl ? (
                        <Image source={{ uri: doc.photoUrl }} style={styles.avatarImg} />
                      ) : (
                        <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.avatarImg} />
                      )}
                    </View>
                    <Text style={styles.circleName} numberOfLines={1}>{doc.name.split(" ")[0]}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            )}

            {/* Health Hub */}
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Health Hub</Text>
            </View>
            <View style={{ flexDirection: "row", gap: 14 }}>
              <TouchableOpacity
                style={[styles.healthCard, { flex: 1, backgroundColor: "#EEF2FF", borderColor: "#C7D2FE", borderWidth: 1 }]}
                onPress={() => router.push("/(patient)/vitals" as any)}
              >
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                  <View style={[styles.healthIcon, { backgroundColor: "#6366F1" }]}>
                    <Ionicons name="pulse" size={24} color="#fff" />
                  </View>
                  <Ionicons name="chevron-forward-circle" size={22} color="#6366F1" />
                </View>
                <Text style={[styles.healthTitle, { color: "#3730A3", marginTop: 14 }]}>My Vitals</Text>
                <Text style={styles.healthSub}>Blood pressure, sugar & more</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.healthCard, { flex: 1, backgroundColor: "#F0FDF4", borderColor: "#BBF7D0", borderWidth: 1 }]}
                onPress={() => router.push("/(patient)/medications" as any)}
              >
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                  <View style={[styles.healthIcon, { backgroundColor: "#10B981" }]}>
                    <Ionicons name="medkit" size={24} color="#fff" />
                  </View>
                  <Ionicons name="chevron-forward-circle" size={22} color="#10B981" />
                </View>
                <Text style={[styles.healthTitle, { color: "#065F46", marginTop: 14 }]}>Medications</Text>
                <Text style={styles.healthSub}>Pills schedule & reminders</Text>
              </TouchableOpacity>
            </View>

            <View style={[styles.infoCard, { marginTop: 24 }]}>
              <Text style={styles.cardHeader}>Personal Details</Text>
              <InfoRow icon="person-outline" label="Full Name" value={profile?.name || "—"} isRTL={isRTL} />
              <InfoRow icon="mail-outline" label="Email" value={profile?.email || "—"} isRTL={isRTL} />
              <InfoRow icon="call-outline" label="Phone" value={profile?.phone || "—"} isRTL={isRTL} />
              <InfoRow icon="calendar-outline" label="Date of Birth" value={profile?.dateOfBirth || "Not set"} isRTL={isRTL} />
            </View>

            <TouchableOpacity style={styles.secondaryBtn} onPress={() => setEditModal(true)}>
              <Ionicons name="create-outline" size={18} color={COLORS.primary} />
              <Text style={styles.secondaryBtnTxt}>Edit Profile Information</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.secondaryBtn, { marginTop: 12 }]} onPress={handleContactSupport}>
              <Ionicons name="headset-outline" size={18} color="#475569" />
              <Text style={styles.secondaryBtnTxt}>Contact Technical Support</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
              <Ionicons name="log-out-outline" size={18} color="#FF4444" />
              <Text style={styles.logoutBtnTxt}>Sign Out</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Bookings Tab */}
        {activeTab === "bookings" && (
          <View>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Active Bookings</Text>
            </View>
            {appointments.filter(a => {
              const status = String(a.status ?? "").toLowerCase();
              return status === "pending" || status === "confirmed";
            }).length === 0 ? (
              <View style={styles.emptyState}>
                <View style={styles.emptyCircle}>
                  <Ionicons name="calendar-outline" size={40} color={COLORS.primary} />
                </View>
                <Text style={styles.emptyStateTitle}>No Active Bookings</Text>
                <Text style={styles.emptyStateSub}>Pending and confirmed bookings will appear here.</Text>
                <TouchableOpacity style={styles.primaryBtnSmall} onPress={() => router.push("/(patient)/doctors")}>
                  <Text style={styles.primaryBtnSmallTxt}>Book Now</Text>
                </TouchableOpacity>
              </View>
            ) : (
              appointments
                .filter(a => {
                  const status = String(a.status ?? "").toLowerCase();
                  return status === "pending" || status === "confirmed";
                })
                .map(b => (
                  <TouchableOpacity 
                    key={b.id} 
                    style={styles.bookingCard}
                    onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(b.doctorId) } })}
                  >
                    <View style={styles.bookingTop}>
                      <View style={styles.docInfo}>
                        <Text style={styles.bookingDocName}>{b.doctorName}</Text>
                        <Text style={styles.bookingSpec}>{b.specialty}</Text>
                      </View>
                      <View style={[
                        styles.statusBadge,
                        String(b.status ?? "").toLowerCase() === "confirmed" && { backgroundColor: "#DCFCE7" }
                      ]}>
                        <Text style={[
                          styles.statusTxt,
                          String(b.status ?? "").toLowerCase() === "confirmed" && { color: "#166534" }
                        ]}>{b.status}</Text>
                      </View>
                    </View>
                    <View style={styles.bookingMeta}>
                      <View style={styles.metaIcon}>
                        <Ionicons name="calendar-outline" size={12} color="#64748B" />
                        <Text style={styles.metaTxt}>{b.date}</Text>
                      </View>
                      <View style={styles.metaIcon}>
                        <Ionicons name="time-outline" size={12} color="#64748B" />
                        <Text style={styles.metaTxt}>{b.time}</Text>
                      </View>
                    </View>
                    {(String(b.status ?? "").toLowerCase() === "pending" || String(b.status ?? "").toLowerCase() === "confirmed") && (
                      <TouchableOpacity 
                        style={styles.cancelBookBtn} 
                        onPress={(e) => {
                          e.stopPropagation();
                          cancelBooking(b.id);
                        }}
                      >
                        <Text style={styles.cancelBookBtnTxt}>Cancel Request</Text>
                      </TouchableOpacity>
                    )}
                  </TouchableOpacity>
              ))
            )}
          </View>
        )}

        {/* Visits Tab */}
        {activeTab === "visits" && (
          <View>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>My Visits</Text>
            </View>
            {loadingVisits ? (
              <ActivityIndicator color={COLORS.primary} style={{ marginVertical: 20 }} />
            ) : visits.length === 0 ? (
              <View style={styles.emptyState}>
                <View style={styles.emptyCircle}>
                  <Ionicons name="medical-outline" size={40} color={COLORS.primary} />
                </View>
                <Text style={styles.emptyStateTitle}>No Visits Found</Text>
                <Text style={styles.emptyStateSub}>Your completed doctor visits will appear here.</Text>
              </View>
            ) : (
              visits.map(v => (
                <TouchableOpacity key={v.id} style={styles.bookingCard} onPress={() => router.push({ pathname: "/(patient)/visit-summary" as any, params: { visitId: String(v.id) } })}>
                  <View style={styles.bookingTop}>
                    <View style={styles.docInfo}>
                      <Text style={styles.bookingDocName}>Visit #{v.id}</Text>
                      <Text style={styles.bookingSpec}>{v.chiefComplaint}</Text>
                    </View>
                    <View style={[styles.statusBadge, v.status === "closed" && { backgroundColor: "#DCFCE7" }]}>
                      <Text style={[styles.statusTxt, v.status === "closed" && { color: "#166534" }]}>{v.status}</Text>
                    </View>
                  </View>
                  <View style={styles.bookingMeta}>
                    <View style={styles.metaIcon}>
                      <Ionicons name="calendar-outline" size={12} color="#64748B" />
                      <Text style={styles.metaTxt}>{v.visitDate}</Text>
                    </View>
                    {v.closedAt && (
                      <View style={styles.metaIcon}>
                        <Ionicons name="checkmark-circle-outline" size={12} color="#64748B" />
                        <Text style={styles.metaTxt}>Closed</Text>
                      </View>
                    )}
                  </View>
                </TouchableOpacity>
              ))
            )}
          </View>
        )}

        {/* Scan placeholder */}
        {activeTab === "scan" && (
          <View style={styles.emptyState}>
            <Ionicons name="scan" size={50} color={COLORS.primary} />
            <Text style={styles.emptyStateTitle}>Features coming soon</Text>
          </View>
        )}

        {/* History Tab */}
        {activeTab === "history" && (
          <View>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Booking History</Text>
            </View>
            {appointments.filter(a => {
              const status = String(a.status ?? "").toLowerCase();
              return status === "completed" || status === "cancelled";
            }).length === 0 ? (
              <View style={styles.emptyState}>
                <View style={styles.emptyCircle}>
                  <Ionicons name="time-outline" size={40} color={COLORS.primary} />
                </View>
                <Text style={styles.emptyStateTitle}>No History Found</Text>
                <Text style={styles.emptyStateSub}>Your completed and cancelled bookings will appear here.</Text>
              </View>
            ) : (
              appointments
                .filter(a => {
                  const status = String(a.status ?? "").toLowerCase();
                  return status === "completed" || status === "cancelled";
                })
                .map(b => (
                <View key={b.id} style={[styles.bookingCard, { opacity: 0.8 }]}>
                  <View style={styles.bookingTop}>
                    <View style={styles.docInfo}>
                      <Text style={styles.bookingDocName}>{b.doctorName}</Text>
                      <Text style={styles.bookingSpec}>{b.specialty}</Text>
                    </View>
                    <View style={[
                      styles.statusBadge,
                      String(b.status ?? "").toLowerCase() === "completed" && { backgroundColor: "#DCFCE7" },
                      String(b.status ?? "").toLowerCase() === "cancelled" && { backgroundColor: "#FEE2E2" }
                    ]}>
                      <Text style={[
                        styles.statusTxt,
                        String(b.status ?? "").toLowerCase() === "completed" && { color: "#166534" },
                        String(b.status ?? "").toLowerCase() === "cancelled" && { color: "#991B1B" }
                      ]}>{b.status}</Text>
                    </View>
                  </View>
                  <View style={styles.bookingMeta}>
                    <View style={styles.metaIcon}>
                      <Ionicons name="calendar-outline" size={12} color="#64748B" />
                      <Text style={styles.metaTxt}>{b.date}</Text>
                    </View>
                    <View style={styles.metaIcon}>
                      <Ionicons name="time-outline" size={12} color="#64748B" />
                      <Text style={styles.metaTxt}>{b.time}</Text>
                    </View>
                  </View>
                </View>
              ))
            )}
            
            {/* Displaying Medical History Items */}
            <View style={[styles.sectionHeader, { marginTop: 20 }]}>
              <Text style={styles.sectionTitle}>My Medical Records</Text>
              <TouchableOpacity onPress={() => router.push("/(patient)/medical-records" as any)}>
                <Text style={styles.seeAll}>View All</Text>
              </TouchableOpacity>
            </View>
            <TouchableOpacity style={styles.infoCard} onPress={() => router.push("/(patient)/medical-records" as any)}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
                <View style={[styles.infoIconBox, { backgroundColor: COLORS.primary + "15" }]}>
                  <Ionicons name="document-text-outline" size={20} color={COLORS.primary} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 14, fontWeight: "700", color: "#1E293B" }}>View Full Medical Records</Text>
                  <Text style={{ fontSize: 12, color: "#64748B", marginTop: 2 }}>Allergies, chronic diseases, medications & vitals</Text>
                </View>
                <Ionicons name="chevron-forward" size={18} color="#94A3B8" />
              </View>
            </TouchableOpacity>

            {history.chronicDiseases.length === 0 && history.surgeries.length === 0 ? (
                <Text style={styles.emptyStateSub}>No local medical history added yet.</Text>
            ) : (
                <>
                  {history.chronicDiseases.map((d, i) => (
                      <Text key={`chronic-${i}`} style={{ color: "#475569", marginBottom: 4 }}>• {d}</Text>
                  ))}
                  {history.surgeries.map((s, i) => (
                      <Text key={`surg-${i}`} style={{ color: "#475569", marginBottom: 4 }}>• {s} (Surgery)</Text>
                  ))}
                </>
            )}
          </View>
        )}
      </ScrollView>

      {/* Edit Profile Modal */}
      <Modal visible={editModal} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.bottomSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Update Profile</Text>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Full Name</Text>
              <TextInput style={styles.textInput} value={editName} onChangeText={setEditName} placeholder="Name" />
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Email</Text>
              <TextInput style={styles.textInput} value={editEmail} onChangeText={setEditEmail} keyboardType="email-address" />
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Phone</Text>
              <TextInput style={styles.textInput} value={editPhone} onChangeText={setEditPhone} keyboardType="phone-pad" />
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Date of Birth (YYYY-MM-DD)</Text>
              <TextInput style={styles.textInput} value={editDateOfBirth} onChangeText={setEditDateOfBirth} placeholder="1990-01-01" />
            </View>

            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.modalCancelBtn} onPress={() => setEditModal(false)}>
                <Text style={styles.modalCancelTxt}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.modalSaveBtn} onPress={saveProfile} disabled={saving}>
                {saving ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.modalSaveTxt}>Save Updates</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Toast />
    </View>
  );
}

function InfoRow({ icon, label, value, isRTL }: { icon: any; label: string; value: string; isRTL?: boolean }) {
  return (
    <View style={[styles.infoRow, isRTL && { flexDirection: "row-reverse" }]}>
      <View style={styles.infoIconBox}>
        <Ionicons name={icon} size={16} color={COLORS.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.infoLabel, isRTL && { textAlign: "right" }]}>{label}</Text>
        <Text style={[styles.infoValue, isRTL && { textAlign: "right" }]}>{value}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  header: { backgroundColor: COLORS.primary, paddingBottom: 20, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 20, paddingTop: 60, paddingBottom: 20 },
  headerTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },
  headerMain: { flexDirection: "row", alignItems: "center", paddingHorizontal: 20, marginBottom: 20 },
  avatarCircle: { width: 72, height: 72, borderRadius: 36, backgroundColor: "rgba(255,255,255,0.25)", justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: "rgba(255,255,255,0.8)", position: "relative", overflow: "visible", shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 5 },
  avatarImg: { width: "100%", height: "100%", borderRadius: 36 },
  cameraBadge: { position: "absolute", bottom: -2, right: -2, backgroundColor: "#fff", width: 24, height: 24, borderRadius: 12, justifyContent: "center", alignItems: "center", borderWidth: 1.5, borderColor: COLORS.primary },
  avatarTxt: { fontSize: 24, fontWeight: "800", color: "#fff" },
  headerInfo: { marginLeft: 16 },
  heroName: { fontSize: 22, fontWeight: "800", color: "#fff" },
  langBadge: { backgroundColor: "rgba(255,255,255,0.2)", borderRadius: 12, paddingHorizontal: 12, paddingVertical: 4, alignSelf: "flex-start", marginTop: 6 },
  langBadgeText: { color: "#fff", fontSize: 11, fontWeight: "700" },
  statsRow: { flexDirection: "row", justifyContent: "space-around", paddingHorizontal: 20, marginTop: 15 },
  statItem: { alignItems: "center", backgroundColor: "rgba(255,255,255,0.1)", paddingVertical: 8, paddingHorizontal: 12, borderRadius: 12, minWidth: 90 },
  statNum: { fontSize: 18, fontWeight: "800", color: "#fff" },
  statLbl: { fontSize: 11, color: "rgba(255,255,255,0.8)", marginTop: 2, fontWeight: "500" },
  statLine: { width: 1, height: "60%", backgroundColor: "rgba(255,255,255,0.15)", alignSelf: "center" },
  tabsWrap: { backgroundColor: "#fff", paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  tabsScroll: { paddingHorizontal: 20 },
  tabBtn: { flexDirection: "row", alignItems: "center", gap: 6, paddingHorizontal: 18, paddingVertical: 10, borderRadius: 20, marginRight: 10, backgroundColor: "#F8FAFC" },
  tabBtnActive: { backgroundColor: COLORS.primary + "15" },
  tabBtnTxt: { fontSize: 13, fontWeight: "600", color: "#64748B" },
  tabBtnTxtActive: { color: COLORS.primary, fontWeight: "700" },
  scrollContent: { padding: 20, paddingBottom: 100 },
  sectionHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 15 },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B" },
  seeAll: { fontSize: 12, color: COLORS.primary, fontWeight: "600" },
  followedEmptyCard: { backgroundColor: "#F1F5F9", borderRadius: 16, padding: 20, alignItems: "center", borderStyle: "dashed", borderWidth: 1, borderColor: "#CBD5E1" },
  followedEmptyText: { fontSize: 13, color: "#64748B", textAlign: "center" },
  followedList: { marginBottom: 20 },
  circleDoc: { alignItems: "center", marginRight: 20, width: 60 },
  circleAvatar: { width: 56, height: 56, borderRadius: 28, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center", marginBottom: 6, borderWidth: 1, borderColor: COLORS.primary + "30" },
  circleAvatarTxt: { fontSize: 20, fontWeight: "700", color: COLORS.primary },
  circleName: { fontSize: 11, color: "#475569", fontWeight: "600", textAlign: "center" },
  infoCard: { backgroundColor: "#fff", borderRadius: 20, padding: 15, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  cardHeader: { fontSize: 13, fontWeight: "700", color: "#64748B", marginBottom: 15, marginLeft: 5 },
  infoRow: { flexDirection: "row", alignItems: "center", gap: 15, paddingVertical: 12 },
  infoIconBox: { width: 36, height: 36, borderRadius: 10, backgroundColor: "#F1F5F9", justifyContent: "center", alignItems: "center" },
  infoLabel: { fontSize: 10, color: "#94A3B8", fontWeight: "600", marginBottom: 2 },
  infoValue: { fontSize: 14, color: "#1E293B", fontWeight: "600" },
  logoutBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, paddingVertical: 15, marginTop: 12, borderRadius: 20, backgroundColor: "#FEF2F2" },
  logoutBtnTxt: { fontSize: 14, fontWeight: "700", color: "#EF4444" },
  emptyState: { alignItems: "center", paddingVertical: 40 },
  emptyCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: COLORS.primary + "10", justifyContent: "center", alignItems: "center", marginBottom: 15 },
  emptyStateTitle: { fontSize: 16, fontWeight: "700", color: "#1E293B", marginBottom: 5 },
  emptyStateSub: { fontSize: 13, color: "#64748B", textAlign: "center", marginBottom: 20 },
  primaryBtnSmall: { backgroundColor: COLORS.primary, paddingHorizontal: 25, paddingVertical: 10, borderRadius: 20 },
  primaryBtnSmallTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },
  bookingCard: { backgroundColor: "#fff", borderRadius: 20, padding: 15, marginBottom: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2, position: "relative", overflow: "hidden" },
  bookingAccent: { position: "absolute", left: 0, top: 0, bottom: 0, width: 4, backgroundColor: COLORS.primary },
  bookingTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 },
  docInfo: { flex: 1 },
  bookingDocName: { fontSize: 15, fontWeight: "700", color: "#1E293B" },
  bookingSpec: { fontSize: 11, color: COLORS.primary, fontWeight: "600", marginTop: 2 },
  statusBadge: { backgroundColor: "#F1F5F9", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
  statusTxt: { fontSize: 10, fontWeight: "700", color: "#64748B", textTransform: "uppercase" },
  bookingMeta: { flexDirection: "row", gap: 15 },
  metaIcon: { flexDirection: "row", alignItems: "center", gap: 5 },
  metaTxt: { fontSize: 12, color: "#64748B" },
  cancelBookBtn: { borderTopWidth: 1, borderTopColor: "#F1F5F9", marginTop: 12, paddingTop: 10, alignItems: "center" },
  cancelBookBtnTxt: { fontSize: 12, fontWeight: "700", color: "#E11D48" },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "flex-end" },
  bottomSheet: { backgroundColor: "#fff", borderTopLeftRadius: 32, borderTopRightRadius: 32, padding: 25 },
  sheetHandle: { width: 40, height: 4, backgroundColor: "#E2E8F0", borderRadius: 2, alignSelf: "center", marginBottom: 20 },
  sheetTitle: { fontSize: 20, fontWeight: "800", color: "#1E293B", marginBottom: 20 },
  inputGroup: { marginBottom: 15 },
  inputLabel: { fontSize: 12, fontWeight: "600", color: "#64748B", marginBottom: 6 },
  textInput: { backgroundColor: "#F8FAFC", borderRadius: 12, paddingHorizontal: 15, paddingVertical: 12, fontSize: 14, color: "#1E293B", borderWidth: 1, borderColor: "#E2E8F0" },
  modalActions: { flexDirection: "row", gap: 12, marginTop: 20 },
  modalCancelBtn: { flex: 1, paddingVertical: 15, alignItems: "center", borderRadius: 16, backgroundColor: "#F1F5F9" },
  modalCancelTxt: { fontSize: 14, fontWeight: "700", color: "#64748B" },
  modalSaveBtn: { flex: 2, paddingVertical: 15, alignItems: "center", borderRadius: 16, backgroundColor: COLORS.primary },
  modalSaveTxt: { fontSize: 14, fontWeight: "700", color: "#fff" },
  secondaryBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, paddingVertical: 15, marginTop: 20, borderRadius: 20, backgroundColor: "#F8FAFC", borderColor: "#E2E8F0", borderWidth: 1 },
  secondaryBtnTxt: { fontSize: 14, fontWeight: "600", color: "#475569" },
  healthCard: { borderRadius: 24, padding: 18, shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 12, elevation: 4 },
  healthIcon: { width: 48, height: 48, borderRadius: 16, justifyContent: "center", alignItems: "center", marginBottom: 12 },
  healthTitle: { fontSize: 15, fontWeight: "800", marginBottom: 4 },
  healthSub: { fontSize: 12, color: "#64748B", fontWeight: "500" },
});
