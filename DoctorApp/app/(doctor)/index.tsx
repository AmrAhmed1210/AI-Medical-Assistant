import React, { useCallback, useState } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar, ActivityIndicator } from "react-native";
import { Users, CalendarDays, Clock, TrendingUp } from "lucide-react-native";
import { useFocusEffect } from "expo-router";
import { COLORS } from "../../constants/colors";
import { getDoctorDashboard, getDoctorProfile, type DoctorDashboardDto, type DoctorProfileDto } from "../../services/doctorService";

const emptyDashboard: DoctorDashboardDto = {
  todayAppointments: 0,
  pendingAppointments: 0,
  totalPatients: 0,
  weekAppointments: 0,
  todayAppointmentsList: [],
  weeklySessionsChart: [],
  recentReports: [],
};

export default function DoctorDashboard() {
  const [loading, setLoading] = useState(true);
  const [dashboard, setDashboard] = useState<DoctorDashboardDto>(emptyDashboard);
  const [profile, setProfile] = useState<DoctorProfileDto | null>(null);

  useFocusEffect(
    useCallback(() => {
      let mounted = true;
      const run = async () => {
        setLoading(true);
        try {
          const [dash, prof] = await Promise.all([getDoctorDashboard(), getDoctorProfile()]);
          if (!mounted) return;
          setDashboard(dash ?? emptyDashboard);
          setProfile(prof ?? null);
        } catch {
          if (!mounted) return;
          setDashboard(emptyDashboard);
          setProfile(null);
        } finally {
          if (mounted) setLoading(false);
        }
      };
      run();
      return () => {
        mounted = false;
      };
    }, [])
  );

  const stats = [
    { label: "Patients Today", value: String(dashboard.totalPatients), icon: Users, change: "" },
    { label: "Appointments", value: String(dashboard.todayAppointments), icon: CalendarDays, change: "" },
    { label: "Pending", value: String(dashboard.pendingAppointments), icon: Clock, change: "" },
    { label: "Week", value: String(dashboard.weekAppointments), icon: TrendingUp, change: "" },
  ];
  const fullName = profile?.fullName || "Doctor";
  const specialty = profile?.specialty || "Specialty not set";
  const initials = fullName.split(" ").filter(Boolean).map((n) => n[0]?.toUpperCase() ?? "").slice(0, 2).join("") || "DR";
  const incompleteProfile = !(profile?.bio && profile.bio.trim().length > 0 && profile?.photoUrl);

  if (loading) {
    return (
      <View style={styles.loaderWrap}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        
        {/* 1. Header Section */}
        <View style={styles.header}>
          <View>
            <Text style={styles.welcomeText}>Good Morning,</Text>
            <Text style={styles.doctorName}>{fullName}</Text>
            <Text style={styles.specialtyText}>{specialty}</Text>
          </View>
          <TouchableOpacity style={styles.profileBadge}>
            <Text style={styles.avatarText}>{initials}</Text>
          </TouchableOpacity>
        </View>

        {incompleteProfile && (
          <View style={styles.noticeCard}>
            <Text style={styles.noticeText}>Complete your profile: add bio and photo.</Text>
          </View>
        )}

        {/* 2. Stats Grid (2 columns) */}
        <View style={styles.statsGrid}>
          {stats.map((stat, index) => (
            <View key={index} style={styles.statCard}>
              <View style={styles.statHeader}>
                <stat.icon size={18} color={COLORS.primary} />
                <Text style={styles.statChangeText}>{stat.change}</Text>
              </View>
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* 3. Today's Schedule Section */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Today&apos;s Schedule</Text>
          <TouchableOpacity>
            <Text style={styles.viewAllText}>View All</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.scheduleContainer}>
          {dashboard.todayAppointmentsList.length === 0 ? (
            <View style={styles.appointmentCard}>
              <Text style={styles.typeText}>No appointments today</Text>
            </View>
          ) : dashboard.todayAppointmentsList.map((item, i) => (
            <View 
              key={i} 
              style={styles.appointmentCard}
            >
              <View style={styles.timeBox}>
                <Text style={styles.timeText}>{item.scheduledAt ? new Date(item.scheduledAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--"}</Text>
              </View>
              
              <View style={styles.verticalDivider} />
              
              <View style={styles.appointmentInfo}>
                <Text style={styles.patientNameText}>{item.patientName ?? "Unknown Patient"}</Text>
                <Text style={styles.typeText}>{item.status ?? "Pending"}</Text>
              </View>

              <View style={styles.statusBadge}>
                <Text style={styles.statusText}>
                  {item.status ?? "Pending"}
                </Text>
              </View>
            </View>
          ))}
        </View>

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FBFBFB",
  },
  loaderWrap: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#FBFBFB" },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 25,
  },
  noticeCard: { backgroundColor: "#FEF3C7", borderColor: "#FDE68A", borderWidth: 1, borderRadius: 12, padding: 10, marginBottom: 16 },
  noticeText: { color: "#92400E", fontSize: 12, fontWeight: "600" },
  welcomeText: {
    fontSize: 14,
    color: "#888",
  },
  doctorName: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#1A1A1A",
  },
  specialtyText: {
    fontSize: 14,
    color: COLORS.primary,
    fontWeight: "600",
  },
  profileBadge: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "#EEE",
    justifyContent: "center",
    alignItems: "center",
  },
  avatarText: {
    fontWeight: "bold",
    color: "#666",
  },
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginBottom: 25,
  },
  statCard: {
    width: "48%",
    backgroundColor: "#FFF",
    padding: 15,
    borderRadius: 18,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#F0F0F0",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.03,
    shadowRadius: 10,
  },
  statHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  statChangeText: {
    fontSize: 10,
    fontWeight: "700",
    color: COLORS.primary,
  },
  statValue: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#1A1A1A",
  },
  statLabel: {
    fontSize: 11,
    color: "#999",
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1A1A1A",
  },
  viewAllText: {
    fontSize: 12,
    color: COLORS.primary,
    fontWeight: "600",
  },
  scheduleContainer: {
    gap: 12,
  },
  appointmentCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFF",
    padding: 15,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#F0F0F0",
  },
  timeBox: {
    width: 70,
  },
  timeText: {
    fontSize: 12,
    fontWeight: "bold",
    color: "#444",
  },
  verticalDivider: {
    width: 1,
    height: 30,
    backgroundColor: "#EEE",
    marginHorizontal: 15,
  },
  appointmentInfo: {
    flex: 1,
  },
  patientNameText: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#1A1A1A",
  },
  typeText: {
    fontSize: 11,
    color: "#888",
    marginTop: 2,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    backgroundColor: "#F5F5F5",
  },
  statusText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#777",
  },
});