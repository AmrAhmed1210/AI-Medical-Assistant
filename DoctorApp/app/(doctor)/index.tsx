import React from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar } from "react-native";
import { Users, CalendarDays, Clock, TrendingUp } from "lucide-react-native";
import { COLORS } from "../../constants/colors";

// البيانات التجريبية (يمكن نقلها لملف منفصل لاحقاً)
const todaySchedule = [
  { time: "09:00 AM", patient: "Mr. Williamson", type: "Follow-up", status: "done" },
  { time: "10:00 AM", patient: "Sarah Mitchell", type: "New Consultation", status: "done" },
  { time: "11:30 AM", patient: "Emily Watson", type: "Review Results", status: "current" },
  { time: "02:00 PM", patient: "Robert Chen", type: "Pre-Op Assessment", status: "upcoming" },
];

const stats = [
  { label: "Patients Today", value: "8", icon: Users, change: "+2" },
  { label: "Appointments", value: "12", icon: CalendarDays, change: "+3" },
  { label: "Avg. Consult", value: "24m", icon: Clock, change: "-3m" },
  { label: "AI Reports", value: "5", icon: TrendingUp, change: "+1" },
];

export default function DoctorDashboard() {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        
        {/* 1. Header Section */}
        <View style={styles.header}>
          <View>
            <Text style={styles.welcomeText}>Good Morning,</Text>
            <Text style={styles.doctorName}>Dr. Eion Morgan</Text>
            <Text style={styles.specialtyText}>Neurologist</Text>
          </View>
          <TouchableOpacity style={styles.profileBadge}>
            <Text style={styles.avatarText}>EM</Text>
          </TouchableOpacity>
        </View>

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
          <Text style={styles.sectionTitle}>Today's Schedule</Text>
          <TouchableOpacity>
            <Text style={styles.viewAllText}>View All</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.scheduleContainer}>
          {todaySchedule.map((item, i) => (
            <View 
              key={i} 
              style={[
                styles.appointmentCard,
                item.status === "current" && styles.currentCard
              ]}
            >
              <View style={styles.timeBox}>
                <Text style={[
                  styles.timeText,
                  item.status === "current" && { color: COLORS.primary }
                ]}>{item.time}</Text>
              </View>
              
              <View style={styles.verticalDivider} />
              
              <View style={styles.appointmentInfo}>
                <Text style={styles.patientNameText}>{item.patient}</Text>
                <Text style={styles.typeText}>{item.type}</Text>
              </View>

              <View style={[
                styles.statusBadge,
                item.status === "done" && styles.doneBadge,
                item.status === "current" && styles.currentBadge
              ]}>
                <Text style={[
                  styles.statusText,
                  item.status === "done" && styles.doneStatusText,
                  item.status === "current" && styles.currentStatusText
                ]}>
                  {item.status === "current" ? "In Progress" : item.status}
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
  currentCard: {
    borderColor: COLORS.primary,
    borderWidth: 1.5,
    backgroundColor: "#F0FAFA",
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
  doneBadge: {
    backgroundColor: "#E8F5E9",
  },
  currentBadge: {
    backgroundColor: COLORS.primary,
  },
  statusText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#777",
  },
  doneStatusText: {
    color: "#4CAF50",
  },
  currentStatusText: {
    color: "#FFF",
  },
});