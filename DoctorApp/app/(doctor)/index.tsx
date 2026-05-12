import React, { useCallback, useState } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar, ActivityIndicator, Image } from "react-native";
import { Users, CalendarDays, Clock, TrendingUp } from "lucide-react-native";
import { useFocusEffect, useRouter } from "expo-router";
import { COLORS } from "../../constants/colors";
import { getDoctorDashboard, getDoctorProfile, type DoctorDashboardDto, type DoctorProfileDto } from "../../services/doctorService";
import { getVisitById, openVisit } from "../../services/visitService";
import Toast from "react-native-toast-message";

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
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [startingVisitId, setStartingVisitId] = useState<number | null>(null);
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
      return () => { mounted = false; };
    }, [])
  );

  const stats = [
    { label: "Patients Today", value: String(dashboard.totalPatients), icon: Users, iconBg: "#EBF5FB", change: "" },
    { label: "Appointments", value: String(dashboard.todayAppointments), icon: CalendarDays, iconBg: "#FEF3E2", change: "" },
    { label: "Pending", value: String(dashboard.pendingAppointments), icon: Clock, iconBg: "#F3E8FF", change: "" },
    { label: "Week", value: String(dashboard.weekAppointments), icon: TrendingUp, iconBg: "#E0F2F1", change: "" },
  ];
  const fullName = profile?.fullName || "Doctor";
  const specialty = profile?.specialty || "Specialty not set";
  const initials = fullName.split(" ").filter(Boolean).map((n) => n[0]?.toUpperCase() ?? "").slice(0, 2).join("") || "DR";
  const incompleteProfile = !(profile?.bio && profile.bio.trim().length > 0 && profile?.photoUrl);

  const handleStartVisit = async (appointment: DoctorDashboardDto["todayAppointmentsList"][number]) => {
    if (!appointment?.patientId || startingVisitId === appointment.id) return;
    try {
      setStartingVisitId(appointment.id);
      const visit = await openVisit({
        patientId: appointment.patientId,
        appointmentId: appointment.id,
        chiefComplaint: appointment.notes?.trim() || "General consultation",
      });
      router.push({ pathname: "/(doctor)/workspace", params: { visitId: String(visit.id), patientId: String(appointment.patientId) } });
    } catch (e: any) {
      const maybeId = Number(String(e?.message || "").match(/\d+/)?.[0] ?? 0);
      if (maybeId > 0) {
        try {
          const existing = await getVisitById(maybeId);
          router.push({ pathname: "/(doctor)/workspace", params: { visitId: String(existing.id), patientId: String(existing.patientId) } });
          return;
        } catch {}
      }
      Toast.show({ type: "error", text1: e?.message || "Failed to start visit" });
    } finally {
      setStartingVisitId(null);
    }
  };

  if (loading) {
    return (
      <View style={styles.loaderWrap}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Gradient Header */}
      <View style={styles.headerGradient}>
        <View style={styles.headerContent}>
          <View style={styles.headerLeft}>
            <Text style={styles.greeting}>Good Morning</Text>
            <Text style={styles.doctorName}>{fullName}</Text>
            <View style={styles.specialtyChip}>
              <Text style={styles.specialtyText}>{specialty}</Text>
            </View>
          </View>
          <TouchableOpacity style={styles.avatarRing}>
            {profile?.photoUrl ? (
              <Image source={{ uri: profile.photoUrl }} style={styles.avatarImg} />
            ) : (
              <View style={styles.avatarPlaceholder}>
                <Text style={styles.avatarInitials}>{initials}</Text>
              </View>
            )}
            <View style={styles.onlineDot} />
          </TouchableOpacity>
        </View>
      </View>

      {incompleteProfile && (
        <View style={styles.noticeCard}>
          <Text style={styles.noticeIcon}>⚠️</Text>
          <Text style={styles.noticeText}>Complete your profile with a bio and photo</Text>
        </View>
      )}

      {/* Stats Grid */}
      <View style={styles.statsGrid}>
        {stats.map((stat, index) => (
          <View key={index} style={styles.statCard}>
            <View style={[styles.statIconWrap, { backgroundColor: stat.iconBg }]}>
              <stat.icon size={20} color={COLORS.primary} />
            </View>
            <Text style={styles.statValue}>{stat.value}</Text>
            <Text style={styles.statLabel}>{stat.label}</Text>
          </View>
        ))}
      </View>

      {/* Schedule Section */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Today's Schedule</Text>
        <TouchableOpacity style={styles.viewAllBtn}>
          <Text style={styles.viewAllText}>View All →</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.scheduleList}>
        {dashboard.todayAppointmentsList.length === 0 ? (
          <View style={styles.emptyCard}>
            <CalendarDays size={32} color="#D0D5DD" />
            <Text style={styles.emptyTitle}>No appointments today</Text>
            <Text style={styles.emptySubtitle}>Enjoy your day!</Text>
          </View>
        ) : dashboard.todayAppointmentsList.map((item, i) => (
          <View key={i} style={styles.appointmentCard}>
            <View style={styles.timeSection}>
              <Text style={styles.timeText}>
                {item.time ?? (item.scheduledAt ? new Date(item.scheduledAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--")}
              </Text>
              <Text style={styles.timeLabel}>{item.status ?? "Pending"}</Text>
            </View>
            <View style={styles.apptDivider} />
            <View style={styles.apptInfo}>
              <Text style={styles.patientNameText}>{item.patientName ?? "Unknown Patient"}</Text>
              {item.notes && <Text style={styles.notesText} numberOfLines={1}>{item.notes}</Text>}
            </View>
            <TouchableOpacity
              style={styles.startBtn}
              onPress={() => handleStartVisit(item)}
              disabled={startingVisitId === item.id}
            >
              {startingVisitId === item.id ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.startBtnText}>Start</Text>
              )}
            </TouchableOpacity>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
  },
  loaderWrap: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  
  headerGradient: {
    backgroundColor: "#00695C",
    paddingTop: 50,
    paddingBottom: 30,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 28,
    borderBottomRightRadius: 28,
    shadowColor: "#00695C",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 12,
  },
  headerContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerLeft: {
    flex: 1,
  },
  greeting: {
    fontSize: 14,
    color: "#80CBC4",
    fontWeight: "500",
    letterSpacing: 0.3,
  },
  doctorName: {
    fontSize: 24,
    fontWeight: "800",
    color: "#FFFFFF",
    marginTop: 2,
  },
  specialtyChip: {
    backgroundColor: "rgba(255,255,255,0.15)",
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 20,
    alignSelf: "flex-start",
    marginTop: 8,
  },
  specialtyText: {
    fontSize: 12,
    color: "#E0F2F1",
    fontWeight: "600",
  },
  avatarRing: {
    position: "relative",
    width: 56,
    height: 56,
    borderRadius: 28,
    borderWidth: 3,
    borderColor: "rgba(255,255,255,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
  avatarPlaceholder: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "rgba(255,255,255,0.2)",
    justifyContent: "center",
    alignItems: "center",
  },
  avatarInitials: {
    fontSize: 18,
    fontWeight: "800",
    color: "#FFFFFF",
  },
  avatarImg: {
    width: 50,
    height: 50,
    borderRadius: 25,
  },
  onlineDot: {
    position: "absolute",
    bottom: 0,
    right: 0,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: "#34D399",
    borderWidth: 2.5,
    borderColor: "#00695C",
  },

  noticeCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FEF3C7",
    marginHorizontal: 20,
    marginTop: -15,
    padding: 14,
    borderRadius: 16,
    gap: 10,
    shadowColor: "#FEF3C7",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
  },
  noticeIcon: { fontSize: 16 },
  noticeText: { color: "#92400E", fontSize: 13, fontWeight: "600", flex: 1 },

  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    marginTop: 20,
  },
  statCard: {
    width: "48%",
    backgroundColor: "#FFFFFF",
    padding: 18,
    borderRadius: 20,
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 3,
  },
  statIconWrap: {
    width: 42,
    height: 42,
    borderRadius: 14,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 12,
  },
  statValue: {
    fontSize: 26,
    fontWeight: "800",
    color: "#1E293B",
  },
  statLabel: {
    fontSize: 12,
    color: "#94A3B8",
    fontWeight: "500",
    marginTop: 2,
  },

  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    marginTop: 10,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#1E293B",
  },
  viewAllBtn: {
    paddingVertical: 4,
  },
  viewAllText: {
    fontSize: 13,
    color: "#00695C",
    fontWeight: "700",
  },

  scheduleList: {
    paddingHorizontal: 20,
    gap: 12,
    paddingBottom: 30,
  },

  emptyCard: {
    alignItems: "center",
    paddingVertical: 40,
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
    elevation: 2,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#94A3B8",
    marginTop: 12,
  },
  emptySubtitle: {
    fontSize: 13,
    color: "#CBD5E1",
    marginTop: 4,
  },

  appointmentCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    padding: 16,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 3,
  },
  timeSection: {
    alignItems: "center",
    width: 65,
  },
  timeText: {
    fontSize: 13,
    fontWeight: "800",
    color: "#1E293B",
  },
  timeLabel: {
    fontSize: 10,
    color: "#94A3B8",
    fontWeight: "600",
    marginTop: 3,
    textTransform: "capitalize",
  },
  apptDivider: {
    width: 1,
    height: 36,
    backgroundColor: "#E2E8F0",
    marginHorizontal: 14,
  },
  apptInfo: {
    flex: 1,
  },
  patientNameText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#1E293B",
  },
  notesText: {
    fontSize: 11,
    color: "#94A3B8",
    marginTop: 2,
  },
  startBtn: {
    backgroundColor: "#00695C",
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 14,
    minWidth: 75,
    alignItems: "center",
    shadowColor: "#00695C",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  startBtnText: {
    color: "#FFFFFF",
    fontWeight: "800",
    fontSize: 13,
  },
});
