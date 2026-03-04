import React, { useState } from "react";
import { 
  View, Text, StyleSheet, ScrollView, TouchableOpacity, 
  Image, Alert, Modal 
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { COLORS } from "../../constants/colors";
import ProfileEdit from "../../components/Profile-edit";

export default function DoctorProfile() {
  const router = useRouter();
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);

  const handleLogout = () => {
    Alert.alert(
      "Logout",
      "Are you sure you want to log out?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Logout", 
          style: "destructive",
          onPress: () => router.replace("/(auth)") 
        },
      ]
    );
  };

  const MenuField = ({ icon, title, subtitle, color = "#333", onPress }: any) => (
    <TouchableOpacity style={styles.menuItem} onPress={onPress}>
      <View style={styles.menuIconBox}>
        <Ionicons name={icon} size={22} color={color} />
      </View>
      <View style={styles.menuInfo}>
        <Text style={[styles.menuTitle, { color }]}>{title}</Text>
        {subtitle && <Text style={styles.menuSubtitle}>{subtitle}</Text>}
      </View>
      <Ionicons name="chevron-forward" size={18} color="#CCC" />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>My Profile</Text>
        </View>

        <View style={styles.profileCard}>
          <Image 
            source={{ uri: "https://via.placeholder.com/150" }} 
            style={styles.avatar} 
          />
          <View style={styles.profileInfo}>
            <Text style={styles.doctorName}>Dr. Ahmed Ali</Text>
            <Text style={styles.specialtyText}>Senior Cardiologist</Text>
            <View style={styles.verifiedBadge}>
              <Ionicons name="checkmark-circle" size={14} color={COLORS.primary} />
              <Text style={styles.verifiedText}>Verified Professional</Text>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Settings</Text>
          <View style={styles.menuList}>
            <MenuField 
              icon="person-outline" 
              title="Personal Information" 
              subtitle="Update your name and photo" 
              onPress={() => setIsEditModalVisible(true)} 
            />
            <MenuField icon="medical-outline" title="Clinic Details" subtitle="Location, hours, and fees" />
            <MenuField icon="notifications-outline" title="Notifications" subtitle="Alerts for new appointments" />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support & Legal</Text>
          <View style={styles.menuList}>
            <MenuField icon="help-circle-outline" title="Help Center" />
            <MenuField icon="shield-checkmark-outline" title="Privacy Policy" />
          </View>
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <View style={styles.logoutIconBox}>
            <Ionicons name="log-out-outline" size={22} color="#EF4444" />
          </View>
          <Text style={styles.logoutText}>Logout</Text>
          <Ionicons name="chevron-forward" size={18} color="#FEE2E2" />
        </TouchableOpacity>
        
        <Text style={styles.versionText}>Version 1.0.4 (AI Enhanced)</Text>
      </ScrollView>

      <Modal
        visible={isEditModalVisible}
        animationType="slide"
        onRequestClose={() => setIsEditModalVisible(false)}
      >
        <ProfileEdit onClose={() => setIsEditModalVisible(false)} />
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FBFBFB" },
  content: { paddingTop: 60, paddingHorizontal: 20, paddingBottom: 40 },
  header: { marginBottom: 25 },
  title: { fontSize: 24, fontWeight: "bold", color: "#1A1A1A" },
  profileCard: { 
    flexDirection: "row", 
    alignItems: "center", 
    backgroundColor: "#FFF", 
    padding: 20, 
    borderRadius: 24, 
    borderWidth: 1, 
    borderColor: "#F0F0F0",
    marginBottom: 30
  },
  avatar: { width: 70, height: 70, borderRadius: 20, backgroundColor: "#F5F5F5" },
  profileInfo: { marginLeft: 15, flex: 1 },
  doctorName: { fontSize: 18, fontWeight: "bold", color: "#333" },
  specialtyText: { fontSize: 13, color: "#888", marginTop: 2 },
  verifiedBadge: { flexDirection: "row", alignItems: "center", gap: 4, marginTop: 6 },
  verifiedText: { fontSize: 11, fontWeight: "600", color: COLORS.primary },
  section: { marginBottom: 25 },
  sectionTitle: { fontSize: 14, fontWeight: "bold", color: "#999", marginLeft: 5, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 },
  menuList: { backgroundColor: "#FFF", borderRadius: 20, borderWidth: 1, borderColor: "#F0F0F0", overflow: "hidden" },
  menuItem: { flexDirection: "row", alignItems: "center", padding: 15, borderBottomWidth: 1, borderBottomColor: "#FBFBFB" },
  menuIconBox: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#F9F9F9", justifyContent: "center", alignItems: "center" },
  menuInfo: { flex: 1, marginLeft: 12 },
  menuTitle: { fontSize: 14, fontWeight: "600" },
  menuSubtitle: { fontSize: 11, color: "#AAA", marginTop: 2 },
  logoutButton: { 
    flexDirection: "row", 
    alignItems: "center", 
    backgroundColor: "#FFF", 
    padding: 15, 
    borderRadius: 20, 
    borderWidth: 1, 
    borderColor: "#FEE2E2",
    marginTop: 10
  },
  logoutIconBox: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#FFF5F5", justifyContent: "center", alignItems: "center" },
  logoutText: { flex: 1, marginLeft: 12, fontSize: 14, fontWeight: "bold", color: "#EF4444" },
  versionText: { textAlign: "center", color: "#CCC", fontSize: 10, marginTop: 30 }
});