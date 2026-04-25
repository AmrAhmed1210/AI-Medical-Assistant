import React, { useState, useCallback, useEffect } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Image, Alert, ActivityIndicator, TextInput, Switch
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect, useRouter } from "expo-router";
import { COLORS } from "../../constants/colors";
import { logout } from "../../services/authService";
import { getDoctorProfile, updateDoctorProfile, uploadDoctorPhoto, type DoctorProfileDto } from "../../services/doctorService";
import Toast from 'react-native-toast-message';
import * as ImagePicker from "expo-image-picker";

export default function DoctorProfile() {
  const router = useRouter();
  const [profile, setProfile] = useState<DoctorProfileDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Form states
  const [name, setName] = useState("");
  const [bio, setBio] = useState("");
  const [fee, setFee] = useState("");
  const [exp, setExp] = useState("");
  const [avail, setAvail] = useState(false);

  const loadProfile = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getDoctorProfile();
      setProfile(data);
      setName(data.fullName || "");
      setBio(data.bio || "");
      setFee(String(data.consultFee || ""));
      setExp(String(data.yearsExperience || ""));
      setAvail(data.isAvailable || false);
    } catch {
      setProfile(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(useCallback(() => {
    loadProfile();
  }, [loadProfile]));

  const handleLogout = () => {
    Alert.alert(
      "Logout",
      "Are you sure you want to log out?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Logout", 
          style: "destructive",
          onPress: async () => {
            await logout();
            router.replace("/(auth)/login");
          }
        },
      ]
    );
  };

  const handleUploadPhoto = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Denied", "We need access to your photos to update your profile.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0].uri) {
      setUploading(true);
      try {
        await uploadDoctorPhoto(result.assets[0].uri);
        Toast.show({ type: "success", text1: "Photo Updated!" });
        await loadProfile();
      } catch (e: any) {
        Alert.alert("Upload Failed", e.message || "Failed to upload photo");
      } finally {
        setUploading(false);
      }
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateDoctorProfile({
        fullName: name,
        bio: bio,
        consultFee: parseFloat(fee) || 0,
        yearsExperience: parseInt(exp) || 0,
        isAvailable: avail,
      });
      Toast.show({
        type: 'success',
        text1: 'Success',
        text2: 'Profile updated successfully! 👋'
      });
      await loadProfile();
      setIsEditing(false);
    } catch (e: any) {
      Alert.alert("Error", e.message || "Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  if (loading && !profile) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  const fullName = profile?.fullName?.trim() || "Doctor";
  const initials = fullName.split(" ").filter(Boolean).map((p) => p[0]?.toUpperCase() ?? "").slice(0, 2).join("") || "DR";
  const profileComplete = Boolean(profile?.bio?.trim() && profile?.photoUrl);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>My Profile</Text>
          <TouchableOpacity onPress={() => isEditing ? handleSave() : setIsEditing(true)} disabled={saving}>
            {saving ? (
              <ActivityIndicator color={COLORS.primary} size="small" />
            ) : (
              <Text style={styles.editBtnText}>{isEditing ? "Save" : "Edit Profile"}</Text>
            )}
          </TouchableOpacity>
        </View>

        <View style={styles.profileCard}>
          <TouchableOpacity style={styles.avatarContainer} onPress={handleUploadPhoto} disabled={uploading}>
            {profile?.photoUrl ? (
              <Image source={{ uri: profile.photoUrl }} style={styles.avatar} />
            ) : (
              <View style={styles.avatarFallback}>
                <Text style={styles.avatarText}>{initials}</Text>
              </View>
            )}
            <View style={styles.cameraBadge}>
              {uploading ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Ionicons name="camera" size={14} color="#fff" />
              )}
            </View>
          </TouchableOpacity>
          <View style={styles.profileInfo}>
            {isEditing ? (
              <TextInput style={styles.nameInput} value={name} onChangeText={setName} placeholder="Full Name" />
            ) : (
              <Text style={styles.doctorName}>{fullName}</Text>
            )}
            <Text style={styles.specialtyText}>{profile?.specialty || "Specialty not set"}</Text>
            <View style={styles.verifiedBadge}>
              <Ionicons name={profileComplete ? "checkmark-circle" : "alert-circle"} size={14} color={profileComplete ? COLORS.primary : "#F59E0B"} />
              <Text style={styles.verifiedText}>{profileComplete ? "Profile complete" : "Profile being completed"}</Text>
            </View>
          </View>
        </View>

        {isEditing ? (
          <View style={styles.editSection}>
            <Text style={styles.fieldLabel}>Biography</Text>
            <TextInput style={[styles.input, styles.textarea]} value={bio} onChangeText={setBio} multiline placeholder="Tell patients about your background..." />

            <View style={styles.inputRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLabel}>Consultation Fee ($)</Text>
                <TextInput style={styles.input} value={fee} onChangeText={setFee} keyboardType="numeric" placeholder="e.g. 50" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLabel}>Experience (Years)</Text>
                <TextInput style={styles.input} value={exp} onChangeText={setExp} keyboardType="numeric" placeholder="e.g. 10" />
              </View>
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.fieldLabel}>Available for Appointments</Text>
              <Switch value={avail} onValueChange={setAvail} trackColor={{ false: "#767577", true: COLORS.primary }} />
            </View>
            
            <TouchableOpacity style={styles.cancelBtn} onPress={() => setIsEditing(false)}>
              <Text style={styles.cancelBtnText}>Cancel Changes</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            <View style={styles.summaryCard}>
              <SummaryRow icon="time-outline" label="Experience" value={profile?.yearsExperience ? `${profile.yearsExperience} years` : "Not set"} />
              <SummaryRow icon="cash-outline" label="Consultation Fee" value={profile?.consultFee ? `$${profile.consultFee}` : "Not set"} />
              <SummaryRow icon="document-text-outline" label="Bio" value={profile?.bio || "Add your bio"} />
              <SummaryRow icon="eye-outline" label="Visibility" value={profile?.isAvailable ? "Online" : "Offline"} />
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Account Settings</Text>
              <View style={styles.menuList}>
                <MenuField icon="person-outline" title="Personal Information" subtitle="Update your name and photo" onPress={() => setIsEditing(true)} />
                <MenuField icon="notifications-outline" title="Notifications" subtitle="Alerts for new appointments" />
              </View>
            </View>
          </>
        )}

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <View style={styles.logoutIconBox}><Ionicons name="log-out-outline" size={22} color="#EF4444" /></View>
          <Text style={styles.logoutText}>Logout</Text>
          <Ionicons name="chevron-forward" size={18} color="#FEE2E2" />
        </TouchableOpacity>
        
        <Text style={styles.versionText}>Version 1.1.0 (Bug Fixed)</Text>
      </ScrollView>
      <Toast />
    </View>
  );
}

function SummaryRow({ icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <View style={styles.summaryRow}>
      <View style={styles.summaryIcon}><Ionicons name={icon} size={16} color={COLORS.primary} /></View>
      <View style={{ flex: 1 }}><Text style={styles.summaryLabel}>{label}</Text><Text style={styles.summaryValue} numberOfLines={3}>{value}</Text></View>
    </View>
  );
}

function MenuField({ icon, title, subtitle, onPress }: any) {
  return (
    <TouchableOpacity style={styles.menuItem} onPress={onPress}>
      <View style={styles.menuIconBox}><Ionicons name={icon} size={22} color="#333" /></View>
      <View style={styles.menuInfo}><Text style={styles.menuTitle}>{title}</Text>{subtitle && <Text style={styles.menuSubtitle}>{subtitle}</Text>}</View>
      <Ionicons name="chevron-forward" size={18} color="#CCC" />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FBFBFB" },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  content: { paddingTop: 60, paddingHorizontal: 20, paddingBottom: 40 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 25 },
  title: { fontSize: 24, fontWeight: "bold", color: "#1A1A1A" },
  editBtnText: { color: COLORS.primary, fontWeight: "700", fontSize: 15 },
  profileCard: { flexDirection: "row", alignItems: "center", backgroundColor: "#FFF", padding: 20, borderRadius: 24, borderWidth: 1, borderColor: "#F0F0F0", marginBottom: 20 },
  avatarContainer: { position: "relative" },
  avatar: { width: 70, height: 70, borderRadius: 20, backgroundColor: "#F5F5F5", overflow: "hidden" },
  avatarFallback: { width: 70, height: 70, borderRadius: 20, backgroundColor: COLORS.primary, alignItems: "center", justifyContent: "center", overflow: "hidden" },
  avatarText: { color: "#fff", fontSize: 24, fontWeight: "800" },
  cameraBadge: { position: "absolute", bottom: -2, right: -2, width: 24, height: 24, borderRadius: 12, backgroundColor: COLORS.primary, alignItems: "center", justifyContent: "center", borderWidth: 2, borderColor: "#fff" },
  profileInfo: { marginLeft: 15, flex: 1 },
  doctorName: { fontSize: 18, fontWeight: "bold", color: "#333" },
  nameInput: { fontSize: 18, fontWeight: "bold", color: "#333", borderBottomWidth: 1, borderBottomColor: COLORS.primary, padding: 0 },
  specialtyText: { fontSize: 13, color: "#888", marginTop: 2 },
  verifiedBadge: { flexDirection: "row", alignItems: "center", gap: 4, marginTop: 6 },
  verifiedText: { fontSize: 11, fontWeight: "600", color: COLORS.primary },
  summaryCard: { backgroundColor: "#fff", borderRadius: 16, borderWidth: 1, borderColor: "#F0F0F0", padding: 12, marginBottom: 20 },
  summaryRow: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 8 },
  summaryIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: "#ECFDF5", alignItems: "center", justifyContent: "center" },
  summaryLabel: { fontSize: 11, color: "#6B7280" },
  summaryValue: { fontSize: 13, color: "#1F2937", fontWeight: "600" },
  editSection: { marginBottom: 20 },
  fieldLabel: { fontSize: 12, fontWeight: "700", color: "#666", marginBottom: 6, marginTop: 10 },
  input: { backgroundColor: "#FFF", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 14, color: "#1A1A1A", borderWidth: 1, borderColor: "#DDD" },
  textarea: { height: 100, textAlignVertical: "top" },
  inputRow: { flexDirection: "row", gap: 12 },
  switchRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 10 },
  cancelBtn: { marginTop: 20, alignItems: "center" },
  cancelBtnText: { color: "#EF4444", fontWeight: "600" },
  section: { marginBottom: 25 },
  sectionTitle: { fontSize: 14, fontWeight: "bold", color: "#999", marginLeft: 5, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 },
  menuList: { backgroundColor: "#FFF", borderRadius: 20, borderWidth: 1, borderColor: "#F0F0F0", overflow: "hidden" },
  menuItem: { flexDirection: "row", alignItems: "center", padding: 15, borderBottomWidth: 1, borderBottomColor: "#FBFBFB" },
  menuIconBox: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#F9F9F9", justifyContent: "center", alignItems: "center" },
  menuInfo: { flex: 1, marginLeft: 12 },
  menuTitle: { fontSize: 14, fontWeight: "600" },
  menuSubtitle: { fontSize: 11, color: "#AAA", marginTop: 2 },
  logoutButton: { flexDirection: "row", alignItems: "center", backgroundColor: "#FFF", padding: 15, borderRadius: 20, borderWidth: 1, borderColor: "#FEE2E2", marginTop: 10 },
  logoutIconBox: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#FFF5F5", justifyContent: "center", alignItems: "center" },
  logoutText: { flex: 1, marginLeft: 12, fontSize: 14, fontWeight: "bold", color: "#EF4444" },
  versionText: { textAlign: "center", color: "#CCC", fontSize: 10, marginTop: 30 }
});