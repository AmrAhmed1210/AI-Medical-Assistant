import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, TextInput, Modal, Alert, ActivityIndicator, Image,
  RefreshControl, Dimensions, KeyboardAvoidingView, Platform, Animated
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import {
  User, Mail, Phone, Calendar, Globe, Settings,
  ChevronRight, LogOut, MessageSquare,
  HeartPulse, Pill, Edit3, Headphones,
  Activity, Sparkles, ShieldCheck, CreditCard, Bell, MapPin, Star,
  ClipboardList, Plus, PlusCircle, AlertCircle, History, FileText, XCircle, Trash2, Camera, FolderOpen, Clock, CheckCircle2,
  Users, Droplet, Scale, Ruler, Cigarette
} from "lucide-react-native";
import { useRouter, useFocusEffect, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import Toast from "react-native-toast-message";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { COLORS } from "../../constants/colors";
import { useLanguage } from "../../context/LanguageContext";

// Import services
import { getMyProfile, updateMyProfile, Profile, uploadProfilePhoto } from "../../services/profileService";
import { getMyAppointments, Appointment, cancelAppointment, deleteAppointment } from "../../services/appointmentService";
import { scheduleAppointmentReminders } from "../../services/appointmentReminders";
import { getMyVisits, PatientVisit } from "../../services/visitService";
import { getDoctorById, DoctorDetails } from "../../services/doctorService";
import { getFollowedDoctorIds } from "../../services/followService";
import { logout, getMyPatientId } from "../../services/authService";
import { addNotification } from "../../services/notificationService";
import { startSupportSession } from "../../services/sessionService";
import { getChronicDiseases, getSurgeries, getAllergies, AllergyRecord, ChronicDisease, SurgeryRecord } from "../../services/medicalRecordService";

const { width, height: SCREEN_HEIGHT } = Dimensions.get("window");

const normalizeStatus = (status?: string) => (status || "").trim().toLowerCase();

const getAppointmentDateTime = (appt: Appointment) => {
  const raw = `${appt.date || ""} ${appt.time || ""}`.trim();
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const isActiveBooking = (appt: Appointment) => {
  const status = normalizeStatus(appt.status);
  return (status === "pending" || status === "confirmed") && !(appt as any).isAutoCancelled;
};

const isPastBooking = (appt: Appointment) => {
  const status = normalizeStatus(appt.status);
  return status === "cancelled" || status === "completed" || status === "noshow" || !!(appt as any).isAutoCancelled;
};

const formatBookingDate = (appt: Appointment) => {
  const date = getAppointmentDateTime(appt);
  if (!date) return appt.date || "No date";
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};

export default function ProfileScreen() {
  const router = useRouter();
  const { tr, isRTL, lang, switchLanguage } = useLanguage();
  const { tab } = useLocalSearchParams<{ tab?: string }>();

  const [profile, setProfile] = useState<Profile | null>(null);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [visits, setVisits] = useState<PatientVisit[]>([]);
  const [followedDoctors, setFollowedDoctors] = useState<DoctorDetails[]>([]);

  const [chronicDiseases, setChronicDiseases] = useState<ChronicDisease[]>([]);
  const [surgeries, setSurgeries] = useState<SurgeryRecord[]>([]);
  const [allergies, setAllergies] = useState<AllergyRecord[]>([]);

  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"info" | "activity" | "history" | "visits">("info");
  const [refreshing, setRefreshing] = useState(false);

  const [isEditModalVisible, setEditModalVisible] = useState(false);
  const [editData, setEditData] = useState({ 
    name: "", phone: "", dateOfBirth: "", 
    gender: "", bloodType: "", weight: "", height: "", smokingStatus: "" 
  });
  const [updating, setUpdating] = useState(false);
  const [photoUploading, setPhotoUploading] = useState(false);
  const [isQuickAddOpen, setQuickAddOpen] = useState(false);

  const scrollY = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (tab === "activity") setActiveTab("activity");
    else if (tab === "history") setActiveTab("history");
    else if (tab === "visits") setActiveTab("visits");
  }, [tab]);

  const DOCUMENT_FOLDERS = [
    { id: "Blood Test", label: "Blood", icon: "water", color: "#EF4444", bg: "#FEF2F2" },
    { id: "X-Ray", label: "Scans", icon: "scan", color: "#6366F1", bg: "#EEF2FF" },
    { id: "MRI", label: "MRI/CT", icon: "layers", color: "#0EA5E9", bg: "#EFF6FF" },
    { id: "Prescription", label: "Scripts", icon: "receipt", color: "#10B981", bg: "#ECFDF5" },
    { id: "Other", label: "Other", icon: "document-text", color: "#64748B", bg: "#F1F5F9" },
  ];

  const fetchAll = async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      const [p, appts, v, followed] = await Promise.all([
        getMyProfile().catch(() => null),
        getMyAppointments().catch(() => []),
        getMyVisits().catch(() => []),
        getFollowedDoctorIds()
          .then((ids) => Promise.all(ids.map((id) => getDoctorById(id).catch(() => null))))
          .then((items) => items.filter((item): item is DoctorDetails => item !== null))
          .catch(() => []),
      ]);

      if (p) {
        setProfile(p);
        setEditData({ 
          name: p.name || "", 
          phone: p.phone || "", 
          dateOfBirth: p.dateOfBirth || "",
          gender: p.gender || "Male",
          bloodType: p.bloodType || "O+",
          weight: p.weight?.toString() || "",
          height: p.height?.toString() || "",
          smokingStatus: p.smokingStatus || "Non-Smoker"
        });
      }
      
      const now = new Date();
      const processedAppts = (appts || []).map(a => {
        const status = a.status?.toLowerCase() || "";
        if (status === "pending") {
          try {
            const apptDate = new Date(a.date);
            let [h, m] = (a.time || "0:0").split(':').map(val => parseInt(val, 10));
            if (a.time?.toLowerCase().includes("pm") && h < 12) h += 12;
            if (a.time?.toLowerCase().includes("am") && h === 12) h = 0;
            
            if (!isNaN(apptDate.getTime())) {
              apptDate.setHours(h, isNaN(m) ? 0 : m, 0, 0);
              if (apptDate < now) {
                return { ...a, status: "Cancelled", isAutoCancelled: true };
              }
            }
          } catch (e) {
            console.log("Date parsing failed for appt", a.id, e);
          }
        }
        return a;
      });

      setAppointments(processedAppts);
      setVisits(v || []);
      setFollowedDoctors(followed || []);

      const futureOnly = processedAppts.filter(a => 
        (a.status?.toLowerCase() === "pending" || a.status?.toLowerCase() === "confirmed") && 
        !((a as any).isAutoCancelled)
      );
      scheduleAppointmentReminders(futureOnly.slice(0, 5));

      const anyAutoCancelled = processedAppts.some((a: any) => a.isAutoCancelled);
      const lastNotifiedKey = "@last_auto_cancel_notif";
      const lastNotified = await AsyncStorage.getItem(lastNotifiedKey);
      const currentCancelIds = processedAppts.filter((a: any) => a.isAutoCancelled).map(a => a.id).join(',');

      if (anyAutoCancelled && lastNotified !== currentCancelIds) {
        addNotification({
          id: `cancel_${Date.now()}`,
          type: 'message',
          icon: '⚠️',
          title: 'Booking Update',
          message: 'Overdue pending bookings have been moved to history.',
          timestamp: Date.now()
        });
        await AsyncStorage.setItem(lastNotifiedKey, currentCancelIds);
      }

      if (pid > 0) {
        const [cd, sur, alg] = await Promise.all([
          getChronicDiseases(pid).catch(() => []),
          getSurgeries(pid).catch(() => []),
          getAllergies(pid).catch(() => []),
        ]);
        setChronicDiseases(cd);
        setSurgeries(sur);
        setAllergies(alg);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchAll();
  }, []);

  const activeBookings = useMemo(
    () => appointments
      .filter(isActiveBooking)
      .sort((a, b) => (getAppointmentDateTime(a)?.getTime() || 0) - (getAppointmentDateTime(b)?.getTime() || 0)),
    [appointments]
  );

  const pastBookings = useMemo(
    () => appointments
      .filter(isPastBooking)
      .sort((a, b) => (getAppointmentDateTime(b)?.getTime() || 0) - (getAppointmentDateTime(a)?.getTime() || 0)),
    [appointments]
  );

  const handleUpdateProfile = async () => {
    try {
      if (!profile) return;
      setUpdating(true);
      await updateMyProfile(
        editData.name, 
        profile.email, 
        editData.phone, 
        editData.dateOfBirth,
        editData.gender,
        editData.bloodType,
        Number(editData.weight) || 0,
        Number(editData.height) || 0,
        editData.smokingStatus
      );
      setEditModalVisible(false);
      Toast.show({ type: "success", text1: "Success", text2: "Profile updated successfully" });
      fetchAll();
    } catch (err: any) {
      Alert.alert("Error", err.message || "Failed to update profile.");
    } finally {
      setUpdating(false);
    }
  };

  const handlePickImage = async () => {
    Alert.alert(
      "Update Profile Photo",
      "Would you like to take a new photo or choose from your gallery?",
      [
        { text: "Take Photo", onPress: () => pickImage(true) },
        { text: "Choose from Gallery", onPress: () => pickImage(false) },
        { text: "Cancel", style: "cancel" }
      ]
    );
  };

  const pickImage = async (useCamera: boolean) => {
    try {
      if (useCamera) {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert("Permission Needed", "Camera access is required to take photos.");
          return;
        }
      }
      const options: ImagePicker.ImagePickerOptions = {
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.7,
      };
      const result = useCamera 
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);
      if (!result.canceled && result.assets && result.assets[0].uri) {
        setPhotoUploading(true);
        await uploadProfilePhoto(result.assets[0].uri);
        Toast.show({ type: "success", text1: "Success", text2: "Profile photo updated!" });
        fetchAll();
      }
    } catch (err: any) {
      Alert.alert("Error", "Failed to upload photo: " + (err.message || "Unknown error"));
    } finally {
      setPhotoUploading(false);
    }
  };

  const handleSupport = async () => {
    try {
      const session = await startSupportSession();
      router.push({ pathname: "/(patient)/messages", params: { sessionId: session.id, title: "Technical Support" } } as any);
    } catch {
      Alert.alert("Error", "Could not start support session.");
    }
  };

  const handleLogout = () => {
    Alert.alert(tr("logout"), tr("logout_confirm"), [
      { text: tr("cancel"), style: "cancel" },
      {
        text: tr("logout"), style: "destructive", onPress: async () => {
          await logout();
          router.replace("/(auth)/login");
        }
      }
    ]);
  };

  const removeBookingLocally = async (id: number) => {
    try {
      await deleteAppointment(id);
      setAppointments(prev => prev.filter(a => a.id !== id));
      Toast.show({ type: "success", text1: "Booking Deleted", text2: "Removed from the server" });
    } catch (error) {
      console.error("Failed to delete booking", error);
      Toast.show({ type: "error", text1: "Delete Failed", text2: "Could not remove from server" });
    }
  };

  const cancelBookingLocally = async (id: number) => {
    try {
      const updated = await cancelAppointment(id);
      setAppointments(prev => prev.map(a => a.id === id ? { ...a, ...updated, status: updated.status || "Cancelled" } : a));
      Toast.show({ type: "info", text1: "Booking Cancelled", text2: "Moved to past bookings" });
    } catch (error) {
      console.error("Failed to cancel booking", error);
      Toast.show({ type: "error", text1: "Cancel Failed", text2: "Could not cancel booking" });
    }
  };

  if (loading && !refreshing) {
    return <View style={styles.center}><ActivityIndicator size="large" color="#059669" /></View>;
  }

  const headerTranslateY = scrollY.interpolate({
    inputRange: [0, 200],
    outputRange: [0, -80],
    extrapolate: 'clamp',
  });

  const avatarScale = scrollY.interpolate({
    inputRange: [-100, 0, 150],
    outputRange: [1.2, 1, 0.8],
    extrapolate: 'clamp',
  });

  const avatarTranslateY = scrollY.interpolate({
    inputRange: [0, 150],
    outputRange: [0, 10],
    extrapolate: 'clamp',
  });

  const blobTranslateY = scrollY.interpolate({
    inputRange: [0, 300],
    outputRange: [0, -50],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 200],
    outputRange: [1, 0.9],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.main}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* 1. FIXED BUTTONS LAYER (Z-INDEX 1000) */}
      <View style={styles.fixedHeaderTop}>
        <View style={styles.headerLeft}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
            <Ionicons name="chevron-back" size={22} color="#fff" />
          </TouchableOpacity>
        </View>
        <Text style={styles.headerTitle} numberOfLines={1}>{tr("my_profile")}</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity onPress={() => router.push("/(patient)/vitals" as any)} style={styles.headerActionBtn}>
            <Activity size={18} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => switchLanguage(lang === 'ar' ? 'en' : 'ar')} style={styles.headerActionBtn}>
            <Globe size={18} color="#fff" />
          </TouchableOpacity>
        </View>
      </View>

      {/* 2. ANIMATED BACKGROUND HEADER */}
      <Animated.View style={[styles.magicHeader, { transform: [{ translateY: headerTranslateY }], opacity: headerOpacity }]}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={StyleSheet.absoluteFill}>
          <Animated.View style={[styles.liquidBlob, { top: 0, left: -20, width: 250, height: 250, backgroundColor: '#10B981', transform: [{ translateY: blobTranslateY }] }]} />
          <Animated.View style={[styles.liquidBlob, { bottom: -50, right: -50, width: 200, height: 200, backgroundColor: '#34D399', transform: [{ translateY: Animated.multiply(blobTranslateY, 1.5) }] }]} />
          <View style={styles.goldDustContainer}>
            {[...Array(8)].map((_, i) => <View key={i} style={[styles.goldParticle, { top: `${Math.random() * 90}%`, left: `${Math.random() * 95}%` }]} />)}
          </View>
          <View style={styles.emeraldWave} />
        </LinearGradient>
      </Animated.View>

      <Animated.ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: true }
        )}
        scrollEventThrottle={16}
        decelerationRate="fast"
        overScrollMode="always"
        bounces={true}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#059669" />}
      >
        <View style={styles.contentOverlap}>
          <View style={styles.profileCardWrap}>
            <Animated.View style={[styles.glassProfileCard, { transform: [{ scale: avatarScale }, { translateY: avatarTranslateY }] }]}>
              <LinearGradient colors={["rgba(255,255,255,1)", "rgba(252,255,254,0.98)"]} style={styles.cardGlassOverlay} />
              <LinearGradient colors={["#FBBF24", "#D97706", "#FBBF24"]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.goldHairline} />
              <View style={styles.profileMain}>
                <View style={styles.avatarGlowContainer}>
                  <View style={styles.avatarPulse} />
                  <View style={styles.avatarGlow}>
                    {photoUploading ? (
                      <ActivityIndicator color="#059669" style={styles.profileImg} />
                    ) : (
                      <Image source={{ uri: profile?.photoUrl || "https://via.placeholder.com/150" }} style={styles.profileImg} />
                    )}
                    <TouchableOpacity style={styles.editPhotoBtn} onPress={handlePickImage} activeOpacity={0.8}>
                      <LinearGradient colors={["#FBBF24", "#D97706"]} style={styles.editPhotoGradient}>
                        <Camera size={12} color="#fff" />
                      </LinearGradient>
                    </TouchableOpacity>
                  </View>
                </View>
                <View style={styles.profileInfo}>
                  <View style={styles.nameRow}>
                    <Text style={styles.profileName} numberOfLines={1}>{profile?.name || "Patient Name"}</Text>
                    <LinearGradient colors={["#FBBF24", "#D97706"]} style={styles.eliteStarBadge}><Star size={10} color="#fff" fill="#fff" /></LinearGradient>
                  </View>
                  <Text style={styles.profileEmail} numberOfLines={1}>{profile?.email || "patient@example.com"}</Text>
                  <View style={styles.badgeRow}>
                    <LinearGradient colors={["#ECFDF5", "#D1FAE5"]} style={styles.premiumBadge}><ShieldCheck size={12} color="#059669" /><Text style={styles.badgeText}>{tr("verified_patient")}</Text></LinearGradient>
                  </View>
                </View>
              </View>
            </Animated.View>
          </View>

          <View style={styles.tabsContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.tabsScroll}>
              <TabItem label={tr("info")} active={activeTab === "info"} onPress={() => setActiveTab("info")} icon={User} />
              <TabItem label={tr("active_bookings")} active={activeTab === "activity"} onPress={() => setActiveTab("activity")} icon={Calendar} />
              <TabItem label={tr("history")} active={activeTab === "history"} onPress={() => setActiveTab("history")} icon={History} />
              <TabItem label={tr("my_visits")} active={activeTab === "visits"} onPress={() => setActiveTab("visits")} icon={FileText} />
            </ScrollView>
          </View>

          <View style={styles.contentArea}>
            {activeTab === "info" && (
              <View style={styles.section}>
                <View style={styles.infoQuickActions}>
                  <TouchableOpacity style={styles.actionCard} onPress={() => router.push("/(patient)/vitals")} activeOpacity={0.8}>
                    <LinearGradient colors={["#0EA5E9", "#0284C7"]} style={styles.actionIconBox}><HeartPulse size={22} color="#fff" /></LinearGradient>
                    <Text style={styles.actionLabel}>{tr("vitals")}</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.actionCard} onPress={() => router.push("/(patient)/medications")} activeOpacity={0.8}>
                    <LinearGradient colors={["#6366F1", "#4F46E5"]} style={styles.actionIconBox}><Pill size={22} color="#fff" /></LinearGradient>
                    <Text style={styles.actionLabel}>{tr("medications")}</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.luxuryInfoBox}>
                  <InfoRow icon={Mail} label={tr("email")} value={profile?.email || "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Phone} label={tr("phone")} value={profile?.phone || "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Calendar} label={tr("date_of_birth")} value={profile?.dateOfBirth ? new Date(profile.dateOfBirth).toLocaleDateString() : "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Users} label="Gender" value={profile?.gender || "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Droplet} label="Blood Type" value={profile?.bloodType || "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Scale} label="Weight" value={profile?.weight ? `${profile.weight} kg` : "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Ruler} label="Height" value={profile?.height ? `${profile.height} cm` : "—"} />
                  <View style={styles.boxDivider} /><InfoRow icon={Cigarette} label="Smoking" value={profile?.smokingStatus || "—"} />
                </View>
                <TouchableOpacity style={styles.supportBtn} onPress={handleSupport} activeOpacity={0.7}><View style={styles.supportIcon}><Headphones size={20} color="#F59E0B" /></View><Text style={styles.supportText}>{tr("contact_support")}</Text><ChevronRight size={18} color="#CBD5E1" /></TouchableOpacity>
                <TouchableOpacity style={styles.luxuryBtn} onPress={() => setEditModalVisible(true)} activeOpacity={0.8}><LinearGradient colors={["#059669", "#047857"]} style={styles.btnGradient}><Edit3 size={18} color="#fff" /><Text style={styles.btnText}>{tr("edit_profile")}</Text></LinearGradient></TouchableOpacity>
              </View>
            )}

            {activeTab === "activity" && (
              <View style={styles.section}>
                <View style={styles.activitySubHeader}>
                  <Calendar size={18} color="#059669" />
                  <Text style={styles.activitySubTitle}>Upcoming Appointments</Text>
                </View>
                {activeBookings.length === 0 ? 
                  <EmptyState icon={Calendar} text={tr("no_active_bookings")} /> :
                  activeBookings.map(appt => (
                    <BookingCard
                      key={appt.id}
                      appt={appt}
                      tr={tr}
                      variant="active"
                      onCancel={cancelBookingLocally}
                      onDelete={removeBookingLocally}
                    />
                  ))}

                <View style={[styles.activitySubHeader, { marginTop: 20 }]}>
                  <History size={18} color="#94A3B8" />
                  <Text style={[styles.activitySubTitle, { color: '#64748B' }]}>Past & Expired Bookings</Text>
                </View>
                {pastBookings.length === 0 ?
                  <Text style={styles.emptyHistoryTxt}>No past booking activity</Text> :
                  pastBookings.map(appt => (
                    <BookingCard
                      key={appt.id}
                      appt={appt}
                      tr={tr}
                      variant="past"
                      onCancel={cancelBookingLocally}
                      onDelete={removeBookingLocally}
                    />
                  ))}
              </View>
            )}

            {activeTab === "history" && (
              <View style={styles.section}>
                <View style={styles.activitySubHeader}>
                  <FolderOpen size={18} color="#059669" />
                  <Text style={styles.activitySubTitle}>Medical Records & Files</Text>
                </View>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.foldersHorizontalScroll}>
                  {DOCUMENT_FOLDERS.map(f => (
                    <TouchableOpacity 
                      key={f.id} 
                      style={styles.profileFolderCard} 
                      onPress={() => router.push({ pathname: "/(patient)/medical-records/documents", params: { folder: f.id } } as any)}
                    >
                      <LinearGradient colors={[f.bg, '#fff']} style={styles.profileFolderGradient}>
                        <Ionicons name={f.icon as any} size={24} color={f.color} />
                        <Text style={[styles.profileFolderLabel, { color: f.color }]}>{f.label}</Text>
                      </LinearGradient>
                    </TouchableOpacity>
                  ))}
                </ScrollView>

                <View style={[styles.activitySubHeader, { marginTop: 20 }]}>
                  <ClipboardList size={18} color="#059669" />
                  <Text style={styles.activitySubTitle}>General Health History</Text>
                </View>
                <HistoryCard icon={HeartPulse} color="#EF4444" title={tr("chronic_diseases")} items={chronicDiseases.map(d => d.diseaseName)} onAdd={() => router.push({ pathname: "/(patient)/medical-records/[category]", params: { category: "chronic" } } as any)} />
                <HistoryCard icon={ClipboardList} color="#0EA5E9" title={tr("surgeries")} items={surgeries.map(s => s.surgeryName)} onAdd={() => router.push({ pathname: "/(patient)/medical-records/[category]", params: { category: "surgeries" } } as any)} />
                <HistoryCard icon={AlertCircle} color="#F59E0B" title={tr("allergies")} items={allergies.map(a => a.allergenName)} onAdd={() => router.push({ pathname: "/(patient)/medical-records/[category]", params: { category: "allergies" } } as any)} />
              </View>
            )}

            {activeTab === "visits" && (
              <View style={styles.section}>
                <View style={styles.activitySubHeader}>
                  <HeartPulse size={18} color="#059669" />
                  <Text style={styles.activitySubTitle}>Followed Doctors</Text>
                </View>
                {followedDoctors.length === 0 ? (
                  <EmptyState icon={FileText} text="No followed doctors yet" />
                ) : (
                  followedDoctors.map((doctor) => (
                    <FollowedDoctorCard key={doctor.id} doctor={doctor} />
                  ))
                )}
              </View>
            )}

            <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout} activeOpacity={0.7}><LogOut size={20} color="#EF4444" /><Text style={styles.logoutText}>{tr("logout")}</Text></TouchableOpacity>
          </View>
        </View>
      </Animated.ScrollView>

      {/* FAB */}
      <View style={styles.fabContainer}>
        {isQuickAddOpen && (
          <View style={styles.quickAddMenu}>
            <QuickAddOption icon={HeartPulse} label={tr("vitals")} color="#0EA5E9" onPress={() => { setQuickAddOpen(false); router.push("/(patient)/vitals"); }} />
            <QuickAddOption icon={Pill} label={tr("medications")} color="#6366F1" onPress={() => { setQuickAddOpen(false); router.push("/(patient)/medications"); }} />
            <QuickAddOption icon={ClipboardList} label={tr("history")} color="#059669" onPress={() => { setQuickAddOpen(false); setActiveTab("history"); }} />
          </View>
        )}
        <TouchableOpacity style={styles.shinyFab} onPress={() => setQuickAddOpen(!isQuickAddOpen)} activeOpacity={0.9}>
          <LinearGradient colors={["#059669", "#064E3B"]} style={styles.fabGradient}><Plus size={30} color="#fff" /><View style={styles.fabShine} /></LinearGradient>
        </TouchableOpacity>
      </View>

      <Modal visible={isEditModalVisible} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.modalContent}>
            <View style={styles.modalHeader}><Text style={styles.modalTitle}>{tr("edit_profile")}</Text><TouchableOpacity onPress={() => setEditModalVisible(false)}><Ionicons name="close" size={24} color="#64748B" /></TouchableOpacity></View>
            <ScrollView style={styles.modalBody} showsVerticalScrollIndicator={false}>
              <InputBox label={tr("full_name")} value={editData.name} onChange={(t) => setEditData({ ...editData, name: t })} icon={User} />
              <InputBox label={tr("phone")} value={editData.phone} onChange={(t) => setEditData({ ...editData, phone: t })} icon={Phone} />
              <InputBox label={tr("date_of_birth")} value={editData.dateOfBirth} onChange={(t) => setEditData({ ...editData, dateOfBirth: t })} icon={Calendar} />
              
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}><InputBox label="Weight (kg)" value={editData.weight} onChange={(t) => setEditData({ ...editData, weight: t })} icon={Scale} /></View>
                <View style={{ flex: 1 }}><InputBox label="Height (cm)" value={editData.height} onChange={(t) => setEditData({ ...editData, height: t })} icon={Ruler} /></View>
              </View>

              <InputBox label="Gender" value={editData.gender} onChange={(t) => setEditData({ ...editData, gender: t })} icon={Users} />
              <InputBox label="Blood Type" value={editData.bloodType} onChange={(t) => setEditData({ ...editData, bloodType: t })} icon={Droplet} />
              <InputBox label="Smoking Status" value={editData.smokingStatus} onChange={(t) => setEditData({ ...editData, smokingStatus: t })} icon={Cigarette} />

              <TouchableOpacity style={styles.saveBtn} onPress={handleUpdateProfile} disabled={updating} activeOpacity={0.8}><LinearGradient colors={["#059669", "#047857"]} style={styles.saveBtnGradient}>{updating ? <ActivityIndicator color="#fff" /> : <><CheckCircle2 size={20} color="#fff" /><Text style={styles.saveBtnText}>{tr("save")}</Text></>}</LinearGradient></TouchableOpacity>
            </ScrollView>
          </KeyboardAvoidingView>
        </View>
      </Modal>
      <Toast />
    </View>
  );
}

function TabItem({ label, active, onPress, icon: Icon }: any) {
  return (
    <TouchableOpacity style={[styles.tabItem, active && styles.tabItemActive]} onPress={onPress} activeOpacity={0.9}>
      <Icon size={16} color={active ? "#fff" : "#64748B"} /><Text style={[styles.tabLabel, active && styles.tabLabelActive]}>{label}</Text>
      {active && <View style={styles.activeIndicator} />}
    </TouchableOpacity>
  );
}

function InfoRow({ icon: Icon, label, value }: any) {
  return (
    <View style={styles.infoRow}><View style={styles.infoIconBox}><Icon size={18} color="#059669" /></View><View style={{ flex: 1 }}><Text style={styles.infoLabel}>{label}</Text><Text style={styles.infoValue}>{value}</Text></View><ChevronRight size={14} color="#CBD5E1" /></View>
  );
}

function HistoryCard({ icon: Icon, color, title, items, onAdd }: any) {
  return (
    <View style={styles.historyCard}>
      <View style={styles.historyHeader}>
        <View style={[styles.historyIconCircle, { backgroundColor: color + '15' }]}>
          <Icon size={18} color={color} />
        </View>
        <Text style={styles.historyTitle}>{title}</Text>
        <TouchableOpacity style={styles.miniAddBtn} onPress={onAdd} activeOpacity={0.7}>
          <Plus size={16} color="#fff" />
        </TouchableOpacity>
      </View>
      <View style={styles.historyItemsList}>
        {items.length === 0 ? 
          <Text style={styles.historyEmpty}>No records found</Text> :
          items.map((it: any, i: number) => (
            <View key={i} style={styles.historyItemRow}>
              <View style={[styles.luxuryBullet, { backgroundColor: color }]} />
              <Text style={styles.historyTextContent}>{it}</Text>
            </View>
          ))
        }
      </View>
    </View>
  );
}

function InputBox({ label, value, onChange, icon: Icon }: { label: string, value: string, onChange: (t: string) => void, icon: any }) {
  return (
    <View style={styles.inputBox}><Text style={styles.inputLabel}>{label}</Text><View style={styles.inputWrapper}><Icon size={18} color="#94A3B8" /><TextInput style={styles.input} value={value} onChangeText={onChange} placeholderTextColor="#CBD5E1" /></View></View>
  );
}

function FollowedDoctorCard({ doctor }: { doctor: DoctorDetails }) {
  const router = useRouter();
  const photoUrl = doctor.imageUrl || (doctor as any).photoUrl || "https://cdn-icons-png.flaticon.com/512/3774/3774299.png";
  const rating = Number(doctor.rating || 0).toFixed(1);

  return (
    <TouchableOpacity
      style={styles.followedDoctorCard}
      activeOpacity={0.82}
      onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { id: doctor.id } } as any)}
    >
      <Image source={{ uri: photoUrl }} style={styles.followedDoctorAvatar} />
      <View style={{ flex: 1 }}>
        <Text style={styles.followedDoctorName} numberOfLines={1}>Dr. {doctor.name}</Text>
        <Text style={styles.followedDoctorMeta} numberOfLines={1}>{doctor.specialty || "Specialist"}</Text>
        <View style={styles.followedDoctorStats}>
          <Star size={12} color="#F59E0B" fill="#F59E0B" />
          <Text style={styles.followedDoctorStatText}>{rating}</Text>
          <Text style={styles.followedDoctorDot}>•</Text>
          <Text style={styles.followedDoctorStatText}>{doctor.yearsExperience || doctor.experience || 0} yrs</Text>
        </View>
      </View>
      <View style={styles.followedDoctorChevron}>
        <ChevronRight size={18} color="#059669" />
      </View>
    </TouchableOpacity>
  );
}

function BookingCard({ appt, variant, onCancel, onDelete }: any) {
  const router = useRouter();
  const isConfirmed = appt.status?.toLowerCase() === "confirmed";
  const isPending = appt.status?.toLowerCase() === "pending";
  const isCancelled = appt.status?.toLowerCase() === "cancelled" || (appt as any).isAutoCancelled;
  const isCompleted = appt.status?.toLowerCase() === "completed";
  const isActive = variant === "active";

  let cardBg = ["#F8FAFC", "#fff"];
  let iconBg = "#F1F5F9";
  let iconColor = "#64748B";
  let statusBg = "#F1F5F9";
  let statusText = "#64748B";

  if (isConfirmed) {
    cardBg = ["#ECFDF5", "#fff"];
    iconBg = "#DCFCE7";
    iconColor = "#059669";
    statusBg = "#059669";
    statusText = "#fff";
  } else if (isPending) {
    cardBg = ["#FFF7ED", "#fff"];
    iconBg = "#FFEDD5";
    iconColor = "#D97706";
    statusBg = "#F59E0B";
    statusText = "#fff";
  } else if (isCancelled) {
    cardBg = ["#FEF2F2", "#fff"];
    iconBg = "#FEE2E2";
    iconColor = "#EF4444";
    statusBg = "#EF4444";
    statusText = "#fff";
  }

  return (
    <View style={[styles.bookingCardModern, isActive && styles.bookingCardActive]}>
      <LinearGradient colors={cardBg as any} style={styles.bookingCardGradient}>
        <View style={styles.cardHeader}>
          <View style={[styles.cardIconBox, { backgroundColor: iconBg }]}>
            <Calendar size={20} color={iconColor} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.cardDoctorName}>Dr. {appt.doctorName}</Text>
            <Text style={styles.cardSpecialtyText}>{appt.specialty}</Text>
          </View>
          <View style={[styles.statusTagModern, { backgroundColor: statusBg }]}>
            <Text style={styles.statusTagText}>{appt.status?.toUpperCase()}</Text>
          </View>
        </View>

        <View style={styles.bookingInfoRow}>
          <View style={styles.infoPill}>
            <Ionicons name="calendar" size={14} color="#64748B" />
            <Text style={styles.infoPillText}>{formatBookingDate(appt)}</Text>
          </View>
          <View style={styles.infoPill}>
            <Ionicons name="time" size={14} color="#64748B" />
            <Text style={styles.infoPillText}>{appt.time}</Text>
          </View>
          {appt.paymentMethod ? (
            <View style={styles.infoPill}>
              <CreditCard size={14} color="#64748B" />
              <Text style={styles.infoPillText}>{appt.paymentMethod}</Text>
            </View>
          ) : null}
        </View>

        <View style={styles.bookingActionRowModern}>
          {!isCancelled && !isCompleted && (isPending || isConfirmed) ? (
            <TouchableOpacity 
              style={styles.cancelBookingActionBtn}
              onPress={() => {
                Alert.alert("Cancel Booking", "Are you sure you want to cancel this appointment?", [
                  { text: "No", style: "cancel" },
                  { text: "Yes, Cancel", style: "destructive", onPress: () => {
                    onCancel(appt.id);
                  }}
                ]);
              }}
            >
              <XCircle size={14} color="#EF4444" />
              <Text style={styles.cancelBtnText}>Cancel Booking</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity 
              style={styles.manageBookingBtn}
              onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { id: appt.doctorId } } as any)}
            >
              <Text style={styles.manageBtnText}>View Doctor</Text>
              <ChevronRight size={14} color="#059669" />
            </TouchableOpacity>
          )}
          
          <TouchableOpacity 
            style={styles.deleteBookingBtn} 
            onPress={() => {
              Alert.alert(
                "Delete Booking",
                "This will permanently delete this booking from the server.",
                [
                  { text: "Cancel", style: "cancel" },
                  { text: "Delete", style: "destructive", onPress: () => onDelete(appt.id) }
                ]
              );
            }}
          >
            <Trash2 size={16} color="#94A3B8" />
          </TouchableOpacity>
        </View>
      </LinearGradient>
    </View>
  );
}

function EmptyState({ icon: Icon, text }: any) {
  return (
    <View style={styles.emptyBox}><Icon size={40} color="#E2E8F0" /><Text style={styles.emptyText}>{text}</Text></View>
  );
}

function QuickAddOption({ icon: Icon, label, color, onPress }: any) {
  return (
    <TouchableOpacity style={styles.quickOption} onPress={onPress} activeOpacity={0.8}><Text style={styles.quickOptionLabel}>{label}</Text><View style={[styles.quickOptionIcon, { backgroundColor: color }]}><Icon size={20} color="#fff" /></View></TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  main: { flex: 1, backgroundColor: "#fff" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  scroll: { paddingBottom: 120, zIndex: 10, paddingTop: 260 },
  fixedHeaderTop: { position: 'absolute', top: 0, left: 0, right: 0, height: 120, paddingTop: 60, paddingHorizontal: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', zIndex: 9999 },
  headerLeft: { width: 50, alignItems: 'flex-start' },
  headerRight: { width: 90, flexDirection: 'row', gap: 8, justifyContent: 'flex-end' },
  backBtn: { width: 40, height: 40, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  headerTitle: { flex: 1, fontSize: 14, fontWeight: '700', color: '#fff', letterSpacing: 0.5, textAlign: 'center', textShadowColor: 'rgba(0,0,0,0.2)', textShadowRadius: 5 },
  headerActionBtn: { width: 38, height: 38, borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  magicHeader: { height: 320, borderBottomLeftRadius: 50, borderBottomRightRadius: 50, position: 'absolute', top: 0, left: 0, right: 0, zIndex: 0, overflow: 'hidden' },
  liquidBlob: { position: 'absolute', borderRadius: 125, opacity: 0.15 },
  goldDustContainer: { ...StyleSheet.absoluteFillObject },
  goldParticle: { position: 'absolute', width: 3, height: 3, borderRadius: 1.5, backgroundColor: '#FBBF24', opacity: 0.4 },
  emeraldWave: { position: 'absolute', bottom: -10, left: 0, right: 0, height: 40, backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 100, transform: [{ scaleX: 2 }] },
  contentOverlap: { backgroundColor: '#fff', borderTopLeftRadius: 40, borderTopRightRadius: 40, minHeight: SCREEN_HEIGHT, paddingTop: 20, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 30, elevation: 15 },
  profileCardWrap: { paddingHorizontal: 20, marginTop: -80, zIndex: 20, marginBottom: 15 },
  glassProfileCard: { backgroundColor: '#fff', borderRadius: 24, padding: 18, elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.15, shadowRadius: 25, position: 'relative', overflow: 'hidden', borderWidth: 1, borderColor: '#F1F5F9' },
  cardGlassOverlay: { ...StyleSheet.absoluteFillObject, opacity: 0.6 },
  goldHairline: { position: 'absolute', top: 0, left: '15%', right: '15%', height: 4, borderBottomLeftRadius: 10, borderBottomRightRadius: 10 },
  profileMain: { flexDirection: 'row', alignItems: 'center', gap: 18 },
  avatarGlowContainer: { position: 'relative', width: 72, height: 72, justifyContent: 'center', alignItems: 'center' },
  avatarPulse: { position: 'absolute', width: 78, height: 78, borderRadius: 39, backgroundColor: '#059669', opacity: 0.12 },
  avatarGlow: { width: 64, height: 64, borderRadius: 32, borderWidth: 3, borderColor: '#fff', elevation: 12, shadowColor: '#059669', shadowOpacity: 0.35, shadowRadius: 15 },
  profileImg: { width: '100%', height: '100%', borderRadius: 32 },
  editPhotoBtn: { position: 'absolute', bottom: 0, right: 0, width: 28, height: 28, borderRadius: 14, overflow: 'hidden', borderWidth: 3, borderColor: '#fff', elevation: 5 },
  editPhotoGradient: { width: '100%', height: '100%', justifyContent: 'center', alignItems: 'center' },
  profileInfo: { flex: 1 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  profileName: { fontSize: 16, fontWeight: '700', color: '#064E3B', letterSpacing: -0.5 },
  eliteStarBadge: { width: 20, height: 20, borderRadius: 10, justifyContent: 'center', alignItems: 'center', elevation: 3 },
  profileEmail: { fontSize: 11, color: '#64748B', marginTop: 3, fontWeight: '600' },
  badgeRow: { flexDirection: 'row', marginTop: 10 },
  premiumBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 4, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(5, 150, 105, 0.1)' },
  badgeText: { fontSize: 10, fontWeight: '800', color: '#059669', textTransform: 'uppercase', letterSpacing: 0.5 },
  tabsContainer: { marginTop: 10 },
  tabsScroll: { paddingHorizontal: 20, gap: 12 },
  tabItem: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#F8FAFC', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 20, borderWidth: 1, borderColor: '#F1F5F9' },
  tabItemActive: { backgroundColor: '#064E3B', borderColor: '#064E3B', elevation: 5 },
  tabLabel: { fontSize: 12, fontWeight: '800', color: '#64748B' },
  tabLabelActive: { color: '#fff' },
  activeIndicator: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: '#FBBF24', marginLeft: 4 },
  infoQuickActions: { flexDirection: 'row', gap: 15, marginBottom: 12, paddingVertical: 8, paddingHorizontal: 4 },
  actionCard: { flex: 1, backgroundColor: '#fff', padding: 15, borderRadius: 24, alignItems: 'center', elevation: 12, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 12, borderWidth: 1, borderColor: '#F8FAFC' },
  actionIconBox: { width: 44, height: 44, borderRadius: 15, justifyContent: 'center', alignItems: 'center', marginBottom: 10, elevation: 5, shadowOpacity: 0.15 },
  actionLabel: { fontSize: 11, fontWeight: '700', color: '#1E293B' },
  contentArea: { paddingHorizontal: 20, marginTop: 10, paddingBottom: 40 },
  section: { gap: 16 },
  luxuryInfoBox: { backgroundColor: '#fff', borderRadius: 28, padding: 8, elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15, borderWidth: 1, borderColor: '#F1F5F9', marginVertical: 4 },
  boxDivider: { height: 1, backgroundColor: '#F8FAFC', marginHorizontal: 20 },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 15, padding: 14 },
  infoIconBox: { width: 40, height: 40, borderRadius: 14, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center' },
  infoLabel: { fontSize: 10, color: '#94A3B8', fontWeight: '700', textTransform: 'uppercase' },
  infoValue: { fontSize: 12, fontWeight: '700', color: '#1E293B', marginTop: 1 },
  supportBtn: { flexDirection: 'row', alignItems: 'center', gap: 14, backgroundColor: '#FFFBEB', padding: 16, borderRadius: 24, borderWidth: 1, borderColor: '#FEF3C7' },
  supportIcon: { width: 42, height: 42, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', elevation: 3 },
  supportText: { fontSize: 13, fontWeight: '700', color: '#92400E', flex: 1 },
  luxuryBtn: { marginTop: 10, borderRadius: 24, overflow: 'hidden', elevation: 8, shadowColor: '#059669', shadowOpacity: 0.2, shadowRadius: 15 },
  btnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, paddingVertical: 16 },
  btnText: { color: '#fff', fontSize: 15, fontWeight: '800', letterSpacing: 0.5 },
  activitySubHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 5 },
  activitySubTitle: { fontSize: 13, fontWeight: '800', color: '#064E3B', letterSpacing: -0.2 },
  emptyBox: { alignItems: 'center', paddingVertical: 40, gap: 10 },
  emptyText: { color: '#94A3B8', fontSize: 13, fontWeight: '600' },
  emptyHistoryTxt: { color: '#94A3B8', fontSize: 12, textAlign: 'center', paddingVertical: 20, fontStyle: 'italic' },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, marginTop: 30, paddingVertical: 15, borderRadius: 20, backgroundColor: '#FEF2F2', borderWidth: 1, borderColor: '#FEE2E2' },
  logoutText: { color: '#EF4444', fontSize: 14, fontWeight: '800' },
  fabContainer: { position: 'absolute', bottom: 30, right: 25, alignItems: 'flex-end', gap: 15, zIndex: 9999 },
  quickAddMenu: { backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 24, padding: 8, gap: 4, elevation: 20, shadowColor: '#000', shadowOpacity: 0.2, shadowRadius: 20, borderWidth: 1, borderColor: '#F1F5F9' },
  quickOption: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 10, paddingRight: 4 },
  quickOptionLabel: { fontSize: 12, fontWeight: '700', color: '#1E293B' },
  quickOptionIcon: { width: 36, height: 36, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  shinyFab: { width: 64, height: 64, borderRadius: 32, elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.3, shadowRadius: 20, overflow: 'hidden' },
  fabGradient: { width: '100%', height: '100%', justifyContent: 'center', alignItems: 'center' },
  fabShine: { position: 'absolute', top: -30, left: -30, width: 60, height: 120, backgroundColor: 'rgba(255,255,255,0.2)', transform: [{ rotate: '45deg' }] },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(6, 78, 59, 0.4)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#fff', borderTopLeftRadius: 40, borderTopRightRadius: 40, height: '85%', padding: 25 },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25 },
  modalTitle: { fontSize: 20, fontWeight: '800', color: '#064E3B' },
  modalBody: { flex: 1 },
  inputBox: { marginBottom: 18 },
  inputLabel: { fontSize: 11, fontWeight: '800', color: '#64748B', marginBottom: 8, textTransform: 'uppercase', marginLeft: 4 },
  inputWrapper: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#F8FAFC', borderRadius: 18, paddingHorizontal: 16, height: 56, borderWidth: 1, borderColor: '#F1F5F9' },
  input: { flex: 1, color: '#1E293B', fontSize: 14, fontWeight: '600' },
  saveBtn: { marginTop: 25, borderRadius: 24, overflow: 'hidden', elevation: 8 },
  saveBtnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12, height: 64 },
  saveBtnText: { color: '#fff', fontSize: 16, fontWeight: '800' },
  followedDoctorCard: { flexDirection: 'row', alignItems: 'center', gap: 15, backgroundColor: '#fff', padding: 12, borderRadius: 24, marginBottom: 12, elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15, borderWidth: 1, borderColor: '#F8FAFC' },
  followedDoctorAvatar: { width: 60, height: 60, borderRadius: 20, backgroundColor: '#F1F5F9' },
  followedDoctorName: { fontSize: 15, fontWeight: '800', color: '#1E293B' },
  followedDoctorMeta: { fontSize: 12, color: '#64748B', fontWeight: '600', marginTop: 2 },
  followedDoctorStats: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 6 },
  followedDoctorStatText: { fontSize: 11, color: '#059669', fontWeight: '700' },
  followedDoctorDot: { fontSize: 11, color: '#CBD5E1' },
  followedDoctorChevron: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center' },
  bookingCardModern: { marginBottom: 16, borderRadius: 28, overflow: 'hidden', elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15, borderWidth: 1, borderColor: '#F1F5F9' },
  bookingCardActive: { elevation: 12, shadowOpacity: 0.1, borderColor: 'rgba(5, 150, 105, 0.2)' },
  bookingCardGradient: { padding: 18 },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 15, marginBottom: 15 },
  cardIconBox: { width: 48, height: 48, borderRadius: 16, justifyContent: 'center', alignItems: 'center' },
  cardDoctorName: { fontSize: 16, fontWeight: '800', color: '#1E293B' },
  cardSpecialtyText: { fontSize: 12, color: '#64748B', fontWeight: '600', marginTop: 2 },
  statusTagModern: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  statusTagText: { fontSize: 10, fontWeight: '800', color: '#fff' },
  bookingInfoRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 15 },
  infoPill: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: '#F1F5F9', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  infoPillText: { fontSize: 11, fontWeight: '700', color: '#64748B' },
  bookingActionRowModern: { flexDirection: 'row', alignItems: 'center', gap: 12, borderTopWidth: 1, borderTopColor: '#F1F5F9', paddingTop: 15 },
  cancelBookingActionBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 44, borderRadius: 14, backgroundColor: '#FEF2F2', borderWidth: 1, borderColor: '#FEE2E2' },
  cancelBtnText: { color: '#EF4444', fontSize: 12, fontWeight: '800' },
  manageBookingBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 44, borderRadius: 14, backgroundColor: '#F0FDF4', borderWidth: 1, borderColor: '#DCFCE7' },
  manageBtnText: { color: '#059669', fontSize: 12, fontWeight: '800' },
  deleteBookingBtn: { width: 44, height: 44, borderRadius: 14, backgroundColor: '#F8FAFC', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#F1F5F9' },
  historyCard: { backgroundColor: '#fff', borderRadius: 28, padding: 18, marginBottom: 16, elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15, borderWidth: 1, borderColor: '#F1F5F9' },
  historyHeader: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 15 },
  historyIconCircle: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  historyTitle: { fontSize: 14, fontWeight: '800', color: '#1E293B', flex: 1 },
  miniAddBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#059669', justifyContent: 'center', alignItems: 'center', elevation: 3 },
  historyItemsList: { gap: 8 },
  historyItemRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 4 },
  luxuryBullet: { width: 6, height: 6, borderRadius: 3 },
  historyTextContent: { fontSize: 12, color: '#475569', fontWeight: '600' },
  historyEmpty: { fontSize: 12, color: '#94A3B8', fontStyle: 'italic', marginLeft: 4 },
  foldersHorizontalScroll: { paddingVertical: 5 },
  profileFolderCard: { width: 110, height: 110, marginRight: 15, borderRadius: 24, overflow: 'hidden', elevation: 8, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 10, borderWidth: 1, borderColor: '#F1F5F9' },
  profileFolderGradient: { flex: 1, padding: 15, justifyContent: 'center', alignItems: 'center', gap: 10 },
  profileFolderLabel: { fontSize: 12, fontWeight: '800' },
});
