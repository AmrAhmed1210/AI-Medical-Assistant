import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, TextInput, Modal, Alert, ActivityIndicator,
} from "react-native";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useState, useCallback } from "react";

interface Booking {
  id: string; doctorName: string; specialty: string;
  date: string; time: string; paymentMethod: "visa"|"cash"; bookedAt: string;
}
interface UserData { firstName: string; lastName: string; email: string; phone: string; }

export default function ProfileScreen() {
  const router = useRouter();
  const [userData,   setUserData]   = useState<UserData>({ firstName:"", lastName:"", email:"", phone:"" });
  const [bookings,   setBookings]   = useState<Booking[]>([]);
  const [editModal,  setEditModal]  = useState(false);
  const [editData,   setEditData]   = useState<UserData>({ firstName:"", lastName:"", email:"", phone:"" });
  const [saving,     setSaving]     = useState(false);
  const [errors,     setErrors]     = useState<any>({});
  const [activeTab,  setActiveTab]  = useState<"info"|"bookings">("info");

  useFocusEffect(useCallback(() => { loadAll(); }, []));

  const loadAll = async () => {
    try {
      const raw = await AsyncStorage.getItem("user");
      if (raw) {
        const data = JSON.parse(raw);
        // يشتغل مع API اللي بيرجع name أو firstName/lastName
        const firstName = data.firstName ?? data.name?.split(" ")[0] ?? "";
        const lastName  = data.lastName  ?? data.name?.split(" ").slice(1).join(" ") ?? "";
        setUserData({ firstName, lastName, email: data.email ?? "", phone: data.phone ?? "" });
      }
      const bRaw = await AsyncStorage.getItem("my_bookings");
      if (bRaw) setBookings(JSON.parse(bRaw));
      else setBookings([]);
    } catch {}
  };

  const openEdit = () => { setEditData({ ...userData }); setErrors({}); setEditModal(true); };

  const validate = () => {
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
      const raw  = await AsyncStorage.getItem("user");
      const base = raw ? JSON.parse(raw) : {};
      const updated = { ...base, ...editData, name: `${editData.firstName} ${editData.lastName}`.trim() };
      await AsyncStorage.setItem("user",     JSON.stringify(updated));
      await AsyncStorage.setItem("userName", updated.name);
      setUserData(editData);
      setEditModal(false);
    } finally { setSaving(false); }
  };

  const cancelBooking = (id: string) => {
    Alert.alert("Cancel Booking", "Are you sure?", [
      { text: "No", style: "cancel" },
      { text: "Yes, Cancel", style: "destructive", onPress: async () => {
        const updated = bookings.filter((b) => b.id !== id);
        setBookings(updated);
        await AsyncStorage.setItem("my_bookings", JSON.stringify(updated));
      }},
    ]);
  };

  const handleLogout = () => {
    Alert.alert("Logout", "Are you sure you want to logout?", [
      { text: "Cancel", style: "cancel" },
      { text: "Logout", style: "destructive", onPress: async () => {
        await AsyncStorage.multiRemove(["isLoggedIn","token","userToken","user","userName","userRole"]);
        router.replace("../(auth)");
      }},
    ]);
  };

  const fullName = [userData.firstName, userData.lastName].filter(Boolean).join(" ") || "User";
  const initials = fullName.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* ── Hero Header ── */}
      <View style={styles.hero}>
        <View style={styles.heroTop}>
          <TouchableOpacity onPress={() => router.back()} style={styles.heroBtn}>
            <Ionicons name="arrow-back" size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.heroTitle}>My Profile</Text>
          <TouchableOpacity onPress={openEdit} style={styles.heroBtn}>
            <Ionicons name="pencil-outline" size={20} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Avatar */}
        <View style={styles.avatarCircle}>
          <Text style={styles.avatarTxt}>{initials || "U"}</Text>
        </View>
        <Text style={styles.heroName}>{fullName}</Text>
        <Text style={styles.heroEmail}>{userData.email || "—"}</Text>

        {/* Quick stats */}
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

      {/* ── Tabs ── */}
      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tab, activeTab === "info" && styles.tabActive]}
          onPress={() => setActiveTab("info")}
        >
          <Ionicons name="person-outline" size={16} color={activeTab === "info" ? COLORS.primary : "#AAA"} />
          <Text style={[styles.tabTxt, activeTab === "info" && styles.tabTxtActive]}>Info</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === "bookings" && styles.tabActive]}
          onPress={() => setActiveTab("bookings")}
        >
          <Ionicons name="calendar-outline" size={16} color={activeTab === "bookings" ? COLORS.primary : "#AAA"} />
          <Text style={[styles.tabTxt, activeTab === "bookings" && styles.tabTxtActive]}>
            Bookings {bookings.length > 0 ? `(${bookings.length})` : ""}
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40, paddingHorizontal: 18 }}>

        {/* ── Info Tab ── */}
        {activeTab === "info" && (
          <View style={{ marginTop: 16 }}>
            <View style={styles.card}>
              <InfoRow icon="person-outline"  label="First Name"  value={userData.firstName || "—"} />
              <Divider />
              <InfoRow icon="person-outline"  label="Last Name"   value={userData.lastName  || "—"} />
              <Divider />
              <InfoRow icon="mail-outline"    label="Email"       value={userData.email     || "—"} />
              <Divider />
              <InfoRow icon="call-outline"    label="Phone"       value={userData.phone     || "—"} />
            </View>

            <TouchableOpacity style={styles.editFullBtn} onPress={openEdit}>
              <Ionicons name="pencil" size={16} color="#fff" />
              <Text style={styles.editFullBtnTxt}>Edit Profile</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
              <Ionicons name="log-out-outline" size={18} color="#FF4444" />
              <Text style={styles.logoutTxt}>Logout</Text>
            </TouchableOpacity>

            <Text style={styles.version}>MedBook v1.0.0</Text>
          </View>
        )}

        {/* ── Bookings Tab ── */}
        {activeTab === "bookings" && (
          <View style={{ marginTop: 16 }}>
            {bookings.length === 0 ? (
              <View style={styles.emptyWrap}>
                <Ionicons name="calendar-outline" size={60} color="#DDD" />
                <Text style={styles.emptyTxt}>No bookings yet</Text>
                <TouchableOpacity style={styles.browseBtn} onPress={() => router.push("/(patient)/doctors")}>
                  <Text style={styles.browseBtnTxt}>Browse Doctors</Text>
                </TouchableOpacity>
              </View>
            ) : (
              bookings.map((b) => (
                <View key={b.id} style={styles.bookingCard}>
                  <View style={styles.bookingAccent} />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.bookingDoc}>{b.doctorName}</Text>
                    <Text style={styles.bookingSpec}>{b.specialty}</Text>
                    <View style={styles.metaRow}>
                      <MetaChip icon="calendar-outline" label={b.date} />
                      <MetaChip icon="time-outline"     label={b.time} />
                      <MetaChip
                        icon={b.paymentMethod === "visa" ? "card-outline" : "cash-outline"}
                        label={b.paymentMethod === "visa" ? "Visa" : "Cash"}
                      />
                    </View>
                  </View>
                  <TouchableOpacity style={styles.cancelBtn} onPress={() => cancelBooking(b.id)}>
                    <Text style={styles.cancelTxt}>Cancel</Text>
                  </TouchableOpacity>
                </View>
              ))
            )}
          </View>
        )}
      </ScrollView>

      {/* ── Edit Modal ── */}
      <Modal visible={editModal} transparent animationType="slide">
        <View style={styles.overlay}>
          <View style={styles.editSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Edit Profile</Text>

            <View style={{ flexDirection: "row", gap: 10 }}>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLbl}>First Name</Text>
                <TextInput
                  style={[styles.input, errors.firstName && styles.inputErr]}
                  value={editData.firstName} onChangeText={(v) => setEditData({ ...editData, firstName: v })}
                  placeholder="First name" placeholderTextColor="#BBB"
                />
                {errors.firstName && <Text style={styles.errTxt}>{errors.firstName}</Text>}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLbl}>Last Name</Text>
                <TextInput
                  style={styles.input} value={editData.lastName}
                  onChangeText={(v) => setEditData({ ...editData, lastName: v })}
                  placeholder="Last name" placeholderTextColor="#BBB"
                />
              </View>
            </View>

            <Text style={styles.fieldLbl}>Email</Text>
            <TextInput
              style={[styles.input, errors.email && styles.inputErr]}
              value={editData.email} onChangeText={(v) => setEditData({ ...editData, email: v })}
              placeholder="Email" placeholderTextColor="#BBB"
              keyboardType="email-address" autoCapitalize="none"
            />
            {errors.email && <Text style={styles.errTxt}>{errors.email}</Text>}

            <Text style={styles.fieldLbl}>Phone</Text>
            <TextInput
              style={styles.input} value={editData.phone}
              onChangeText={(v) => setEditData({ ...editData, phone: v })}
              placeholder="Phone number" placeholderTextColor="#BBB" keyboardType="phone-pad"
            />

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

function InfoRow({ icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <View style={styles.infoIcon}>
        <Ionicons name={icon} size={15} color={COLORS.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.infoLbl}>{label}</Text>
        <Text style={styles.infoVal}>{value}</Text>
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

function Divider() {
  return <View style={{ height: 1, backgroundColor: "#F5F5F5", marginLeft: 48 }} />;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F4F6FA" },

  hero: {
    backgroundColor: COLORS.primary, alignItems: "center",
    paddingBottom: 24, borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
  },
  heroTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", width: "100%", paddingHorizontal: 18, paddingTop: 52, paddingBottom: 16 },
  heroBtn:   { width: 38, height: 38, justifyContent: "center", alignItems: "center" },
  heroTitle: { fontSize: 18, fontWeight: "700", color: "#fff" },

  avatarCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: "#fff", justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: "rgba(255,255,255,0.35)", marginBottom: 10 },
  avatarTxt:    { fontSize: 28, fontWeight: "800", color: COLORS.primary },
  heroName:     { fontSize: 20, fontWeight: "800", color: "#fff" },
  heroEmail:    { fontSize: 12, color: "rgba(255,255,255,0.7)", marginTop: 2, marginBottom: 18 },

  statsRow: { flexDirection: "row", backgroundColor: "rgba(255,255,255,0.15)", borderRadius: 16, paddingVertical: 12, paddingHorizontal: 24, gap: 0, width: "88%" },
  statItem: { flex: 1, alignItems: "center" },
  statDiv:  { width: 1, backgroundColor: "rgba(255,255,255,0.25)" },
  statNum:  { fontSize: 18, fontWeight: "800", color: "#fff" },
  statLbl:  { fontSize: 10, color: "rgba(255,255,255,0.7)", marginTop: 2, textAlign: "center" },

  tabs: { flexDirection: "row", marginHorizontal: 18, marginTop: 16, backgroundColor: "#fff", borderRadius: 14, padding: 4, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 4, elevation: 2 },
  tab:        { flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, paddingVertical: 10, borderRadius: 11 },
  tabActive:  { backgroundColor: COLORS.primary + "15" },
  tabTxt:     { fontSize: 13, color: "#AAA", fontWeight: "500" },
  tabTxtActive:{ color: COLORS.primary, fontWeight: "700" },

  card: { backgroundColor: "#fff", borderRadius: 18, overflow: "hidden", shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2 },
  infoRow:  { flexDirection: "row", alignItems: "center", gap: 12, padding: 14 },
  infoIcon: { width: 34, height: 34, borderRadius: 10, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center" },
  infoLbl:  { fontSize: 10, color: "#AAA", marginBottom: 2 },
  infoVal:  { fontSize: 14, color: "#1A1A1A", fontWeight: "600" },

  editFullBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 14, marginTop: 14 },
  editFullBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },

  logoutBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, borderRadius: 16, paddingVertical: 14, marginTop: 10, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#FFE5E5" },
  logoutTxt: { color: "#FF4444", fontSize: 15, fontWeight: "700" },
  version:   { textAlign: "center", color: "#CCC", fontSize: 11, marginTop: 18 },

  emptyWrap: { alignItems: "center", paddingVertical: 48, gap: 10 },
  emptyTxt:  { fontSize: 14, color: "#BBB" },
  browseBtn: { backgroundColor: COLORS.primary, paddingHorizontal: 22, paddingVertical: 10, borderRadius: 20, marginTop: 4 },
  browseBtnTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },

  bookingCard: { backgroundColor: "#fff", borderRadius: 16, padding: 14, marginBottom: 10, flexDirection: "row", alignItems: "center", gap: 0, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2, overflow: "hidden" },
  bookingAccent: { position: "absolute", left: 0, top: 0, bottom: 0, width: 4, backgroundColor: COLORS.primary },
  bookingDoc:  { fontSize: 14, fontWeight: "700", color: "#1A1A1A", marginLeft: 10 },
  bookingSpec: { fontSize: 11, color: COLORS.primary, fontWeight: "500", marginTop: 2, marginLeft: 10 },
  metaRow:     { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 8, marginLeft: 10 },
  metaChip:    { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: COLORS.primary + "12", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 20 },
  metaChipTxt: { fontSize: 10, color: "#555", fontWeight: "500" },
  cancelBtn:   { backgroundColor: "#FFF0F0", paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  cancelTxt:   { color: "#FF4444", fontSize: 11, fontWeight: "700" },

  overlay:   { flex: 1, backgroundColor: "rgba(0,0,0,0.45)", justifyContent: "flex-end" },
  editSheet: { backgroundColor: "#fff", borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 22, paddingTop: 14, paddingBottom: 36 },
  sheetHandle:{ width: 40, height: 4, borderRadius: 2, backgroundColor: "#DDD", alignSelf: "center", marginBottom: 18 },
  sheetTitle: { fontSize: 19, fontWeight: "800", color: "#1A1A1A", marginBottom: 14 },
  fieldLbl:   { fontSize: 12, fontWeight: "600", color: "#555", marginBottom: 6, marginTop: 10 },
  input:      { backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 14, color: "#1A1A1A", borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2 },
  inputErr:   { borderColor: "#e53935" },
  errTxt:     { fontSize: 11, color: "#e53935", marginTop: 2 },
  saveBtn:    { backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 14, alignItems: "center", justifyContent: "center" },
  saveBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
});