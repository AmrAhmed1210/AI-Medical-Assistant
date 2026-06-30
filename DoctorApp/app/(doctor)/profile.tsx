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
import { useLanguage } from "../../context/LanguageContext";

export default function DoctorProfile() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();
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
  const [phone, setPhone] = useState("");
  const [location, setLocation] = useState("");

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
      setPhone((data as any).phoneNumber || "");
      setLocation((data as any).location || "");
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
      tr("sign_out"),
      tr("sign_out_desc"),
      [
        { text: tr("cancel"), style: "cancel" },
        { 
          text: tr("sign_out"), 
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
      Alert.alert(tr("permission_denied"), tr("allow_photos"));
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
        Toast.show({ type: "success", text1: tr("photo_updated") });
        await loadProfile();
      } catch (e: any) {
        Alert.alert(tr("upload_failed"), e.message || tr("error"));
      } finally {
        setUploading(false);
      }
    }
  };

  const handleSave = async () => {
    if (!phone.trim()) {
      Alert.alert(tr("phone_required"), tr("please_add_phone"));
      return;
    }
    setSaving(true);
    try {
      await updateDoctorProfile({
        fullName: name,
        bio: bio,
        consultFee: parseFloat(fee) || 0,
        yearsExperience: parseInt(exp) || 0,
        isAvailable: avail,
        ...(phone.trim() && { phoneNumber: phone.trim() }),
        ...(location.trim() && { location: location.trim() }),
      } as any);
      Toast.show({
        type: 'success',
        text1: tr("success"),
        text2: isRTL ? 'تم تحديث الملف الشخصي بنجاح! 👋' : 'Profile updated successfully! 👋'
      });
      await loadProfile();
      setIsEditing(false);
    } catch (e: any) {
      Alert.alert(tr("error"), e.message || tr("error"));
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

  const fullName = profile?.fullName?.trim() || tr("doctor");
  const initials = fullName.split(" ").filter(Boolean).map((p) => p[0]?.toUpperCase() ?? "").slice(0, 2).join("") || "DR";
  const profileComplete = Boolean(profile?.bio?.trim() && profile?.photoUrl);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={[styles.header, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <Text style={styles.title}>{tr("profile")}</Text>
          <TouchableOpacity onPress={() => isEditing ? handleSave() : setIsEditing(true)} disabled={saving}>
            {saving ? (
              <ActivityIndicator color={COLORS.primary} size="small" />
            ) : (
              <Text style={styles.editBtnText}>{isEditing ? tr("save") : (tr("edit") + " " + tr("profile"))}</Text>
            )}
          </TouchableOpacity>
        </View>

        <View style={[styles.profileCard, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TouchableOpacity style={styles.avatarContainer} onPress={handleUploadPhoto} disabled={uploading}>
            {profile?.photoUrl ? (
              <Image source={{ uri: profile.photoUrl }} style={styles.avatar} />
            ) : (
              <View style={styles.avatarFallback}>
                <Text style={styles.avatarText}>{initials}</Text>
              </View>
            )}
            <View style={[styles.cameraBadge, isRTL ? { left: -2, right: undefined } : { right: -2 }]}>
              {uploading ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Ionicons name="camera" size={14} color="#fff" />
              )}
            </View>
          </TouchableOpacity>
          <View style={[styles.profileInfo, isRTL ? { marginRight: 15, marginLeft: 0, alignItems: "flex-end" } : { marginLeft: 15 }]}>
            {isEditing ? (
              <TextInput style={[styles.nameInput, { textAlign: isRTL ? "right" : "left" }]} value={name} onChangeText={setName} placeholder={tr("full_name_placeholder")} />
            ) : (
              <Text style={styles.doctorName}>{fullName}</Text>
            )}
            <Text style={styles.specialtyText}>{profile?.specialty || tr("not_set")}</Text>
            <View style={[styles.verifiedBadge, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
              <Ionicons name={profileComplete ? "checkmark-circle" : "alert-circle"} size={14} color={profileComplete ? COLORS.primary : "#F59E0B"} />
              <Text style={styles.verifiedText}>{profileComplete ? tr("profile_complete") : tr("profile_being_completed")}</Text>
            </View>
          </View>
        </View>

        {isEditing ? (
          <View style={[styles.editSection, { alignItems: isRTL ? "flex-end" : "stretch" }]}>
            <Text style={styles.fieldLabel}>{tr("biography")}</Text>
            <TextInput style={[styles.input, styles.textarea, { textAlign: isRTL ? "right" : "left" }]} value={bio} onChangeText={setBio} multiline placeholder={tr("bio_placeholder")} />

            <View style={[styles.inputRow, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
              <View style={{ flex: 1 }}>
                <Text style={[styles.fieldLabel, { textAlign: isRTL ? "right" : "left" }]}>{tr("consultation_fee_egp")}</Text>
                <TextInput style={[styles.input, { textAlign: isRTL ? "right" : "left" }]} value={fee} onChangeText={setFee} keyboardType="numeric" placeholder={tr("fee_placeholder")} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.fieldLabel, { textAlign: isRTL ? "right" : "left" }]}>{tr("experience_years")}</Text>
                <TextInput style={[styles.input, { textAlign: isRTL ? "right" : "left" }]} value={exp} onChangeText={setExp} keyboardType="numeric" placeholder={tr("exp_placeholder")} />
              </View>
            </View>

            <Text style={[styles.fieldLabel, { color: '#EF4444' }, { textAlign: isRTL ? "right" : "left" }]}>
              {isRTL ? "📞 هاتف التواصل * (مطلوب)" : "📞 Contact Phone * (Required)"}
            </Text>
            <TextInput
              style={[styles.input, !phone.trim() && { borderColor: '#FCA5A5', borderWidth: 2 }, { textAlign: isRTL ? "right" : "left", width: "100%" }]}
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
              placeholder={tr("phone_placeholder")}
            />
            {!phone.trim() && (
              <Text style={{ color: '#EF4444', fontSize: 11, marginTop: 4, textAlign: isRTL ? "right" : "left" }}>{tr("phone_required_warning")}</Text>
            )}

            <Text style={[styles.fieldLabel, { textAlign: isRTL ? "right" : "left" }]}>{isRTL ? "📍 موقع العيادة" : "📍 Clinic Location"}</Text>
            <TextInput
              style={[styles.input, { textAlign: isRTL ? "right" : "left", width: "100%" }]}
              value={location}
              onChangeText={setLocation}
              placeholder={tr("location_placeholder")}
            />

            <View style={[styles.switchRow, { flexDirection: isRTL ? "row-reverse" : "row", width: "100%" }]}>
              <Text style={styles.fieldLabel}>{tr("avail_appointments")}</Text>
              <Switch value={avail} onValueChange={setAvail} trackColor={{ false: "#767577", true: COLORS.primary }} />
            </View>
            
            <TouchableOpacity style={styles.cancelBtn} onPress={() => setIsEditing(false)}>
              <Text style={styles.cancelBtnText}>{tr("cancel_changes")}</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            <View style={styles.summaryCard}>
              <SummaryRow icon="time-outline" label={isRTL ? "الخبرة" : "Experience"} value={profile?.yearsExperience ? (isRTL ? `${profile.yearsExperience} ${tr("years")}` : `${profile.yearsExperience} years`) : tr("not_set")} isRTL={isRTL} />
              <SummaryRow icon="cash-outline" label={isRTL ? "رسوم الكشف" : "Consultation Fee"} value={profile?.consultFee ? (isRTL ? `${profile.consultFee} ج.م` : `${profile.consultFee} EGP`) : tr("not_set")} isRTL={isRTL} />
              <SummaryRow icon="call-outline" label={isRTL ? "هاتف التواصل" : "Contact Phone"} value={(profile as any)?.phoneNumber || tr("not_set_warning")} isRTL={isRTL} />
              <SummaryRow icon="location-outline" label={isRTL ? "موقع العيادة" : "Clinic Location"} value={(profile as any)?.location || tr("not_set")} isRTL={isRTL} />
              <SummaryRow icon="document-text-outline" label={isRTL ? "النبذة التعريفية" : "Bio"} value={profile?.bio || tr("add_bio_hint")} isRTL={isRTL} />
              <SummaryRow icon="eye-outline" label={isRTL ? "حالة الظهور" : "Visibility"} value={profile?.isAvailable ? tr("online") : tr("offline")} isRTL={isRTL} />
            </View>

            <View style={styles.section}>
              <Text style={[styles.sectionTitle, { textAlign: isRTL ? "right" : "left" }]}>{tr("account_settings")}</Text>
              <View style={styles.menuList}>
                <MenuField icon="person-outline" title={tr("personal_info")} subtitle={tr("update_name_photo")} onPress={() => setIsEditing(true)} isRTL={isRTL} />
                <MenuField icon="notifications-outline" title={isRTL ? "التنبيهات" : "Notifications"} subtitle={tr("alerts_new_appt")} isRTL={isRTL} />
              </View>
            </View>
          </>
        )}

        <TouchableOpacity style={[styles.logoutButton, { flexDirection: isRTL ? "row-reverse" : "row" }]} onPress={handleLogout}>
          <View style={styles.logoutIconBox}><Ionicons name="log-out-outline" size={22} color="#EF4444" /></View>
          <Text style={[styles.logoutText, isRTL ? { marginRight: 12, marginLeft: 0, textAlign: "right" } : { marginLeft: 12 }]}>{tr("sign_out")}</Text>
          <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={18} color="#FEE2E2" />
        </TouchableOpacity>
        
        <Text style={styles.versionText}>{isRTL ? "النسخة 1.1.0 (تم إصلاح الأخطاء)" : "Version 1.1.0 (Bug Fixed)"}</Text>
      </ScrollView>
      <Toast />
    </View>
  );
}

function SummaryRow({ icon, label, value, isRTL }: { icon: any; label: string; value: string; isRTL: boolean }) {
  return (
    <View style={[styles.summaryRow, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
      <View style={styles.summaryIcon}><Ionicons name={icon} size={16} color={COLORS.primary} /></View>
      <View style={[{ flex: 1 }, isRTL ? { marginRight: 10, marginLeft: 0, alignItems: "flex-end" } : { marginLeft: 10 }]}><Text style={styles.summaryLabel}>{label}</Text><Text style={styles.summaryValue} numberOfLines={3}>{value}</Text></View>
    </View>
  );
}

function MenuField({ icon, title, subtitle, onPress, isRTL }: any) {
  return (
    <TouchableOpacity style={[styles.menuItem, { flexDirection: isRTL ? "row-reverse" : "row" }]} onPress={onPress}>
      <View style={styles.menuIconBox}><Ionicons name={icon} size={22} color="#333" /></View>
      <View style={[styles.menuInfo, isRTL ? { marginRight: 12, marginLeft: 0, alignItems: "flex-end" } : { marginLeft: 12 }]}><Text style={styles.menuTitle}>{title}</Text>{subtitle && <Text style={styles.menuSubtitle}>{subtitle}</Text>}</View>
      <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={18} color="#CCC" />
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