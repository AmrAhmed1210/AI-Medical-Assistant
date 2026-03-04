import React, { useState } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from "react-native";
import { Clock, User, Calendar as CalendarIcon } from "lucide-react-native";
import { COLORS } from "../../constants/colors";

const weekDays = [
  { date: 15, day: "Mon" },
  { date: 16, day: "Tue" },
  { date: 17, day: "Wed" },
  { date: 18, day: "Thu" },
  { date: 19, day: "Fri" },
  { date: 20, day: "Sat" },
  { date: 21, day: "Sun" },
];

const scheduleData: Record<number, { time: string; patient: string; type: string }[]> = {
  15: [
    { time: "09:00 AM", patient: "Mr. Williamson", type: "Follow-up" },
    { time: "10:00 AM", patient: "Sarah Mitchell", type: "Consultation" },
    { time: "11:30 AM", patient: "Emily Watson", type: "Review" },
    { time: "02:00 PM", patient: "Robert Chen", type: "Pre-Op" },
  ],
  16: [
    { time: "09:30 AM", patient: "Ahmed Hassan", type: "Emergency" },
    { time: "11:00 AM", patient: "Lisa Wong", type: "New Patient" },
    { time: "03:00 PM", patient: "Mark Davis", type: "Follow-up" },
  ],
  17: [
    { time: "10:00 AM", patient: "Anna Lee", type: "Consultation" },
    { time: "01:00 PM", patient: "John Smith", type: "Review" },
  ],
  // ... يمكنك إضافة باقي الأيام هنا
};

export default function DoctorSchedule() {
  const [selectedDay, setSelectedDay] = useState(15);
  const slots = scheduleData[selectedDay] || [];

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>My Schedule</Text>
        <Text style={styles.subtitle}>October 2024</Text>
      </View>

      {/* Day Picker (Horizontal Scroll) */}
      <View style={{ marginBottom: 25 }}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.dayPickerContainer}>
          {weekDays.map((d) => {
            const count = (scheduleData[d.date] || []).length;
            const isSelected = selectedDay === d.date;
            
            return (
              <TouchableOpacity
                key={d.date}
                onPress={() => setSelectedDay(d.date)}
                style={[styles.dayButton, isSelected && styles.selectedDayButton]}
              >
                <Text style={[styles.dateText, isSelected && styles.selectedText]}>{d.date}</Text>
                <Text style={[styles.dayNameText, isSelected && styles.selectedSubtext]}>{d.day}</Text>
                {count > 0 && (
                  <View style={[styles.dot, isSelected ? { backgroundColor: '#fff' } : { backgroundColor: COLORS.primary }]} />
                )}
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      </View>

      {/* Timeline Section */}
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 100 }}>
        <View style={styles.sectionTitleRow}>
          <Text style={styles.sectionTitle}>
            {weekDays.find((d) => d.date === selectedDay)?.day}, Oct {selectedDay}
          </Text>
          <Text style={styles.countText}>{slots.length} appointments</Text>
        </View>

        {slots.length > 0 ? (
          <View style={styles.timelineWrapper}>
            {/* Vertical Line */}
            <View style={styles.timelineLine} />

            {slots.map((slot, i) => (
              <View key={i} style={styles.timelineItem}>
                {/* Dot on line */}
                <View style={styles.dotContainer}>
                  <View style={styles.timelineDot} />
                </View>

                {/* Card */}
                <View style={styles.card}>
                  <View style={styles.cardHeader}>
                    <View style={styles.timeContainer}>
                      <Clock size={12} color={COLORS.primary} />
                      <Text style={styles.timeText}>{slot.time}</Text>
                    </View>
                    <View style={styles.typeTag}>
                      <Text style={styles.typeText}>{slot.type}</Text>
                    </View>
                  </View>
                  
                  <View style={styles.patientRow}>
                    <User size={12} color="#888" />
                    <Text style={styles.patientName}>{slot.patient}</Text>
                  </View>
                </View>
              </View>
            ))}
          </View>
        ) : (
          /* Empty State */
          <View style={styles.emptyState}>
            <View style={styles.emptyIconCircle}>
              <CalendarIcon size={32} color="#CCC" />
            </View>
            <Text style={styles.emptyTitle}>No appointments</Text>
            <Text style={styles.emptySubtitle}>Enjoy your day off, Doctor!</Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FBFBFB", paddingTop: 60, paddingHorizontal: 20 },
  header: { marginBottom: 20 },
  title: { fontSize: 22, fontWeight: "bold", color: "#1A1A1A" },
  subtitle: { fontSize: 13, color: "#888", marginTop: 2 },
  
  dayPickerContainer: { paddingRight: 40, gap: 10 },
  dayButton: { 
    minWidth: 55, 
    height: 75, 
    backgroundColor: "#F0F0F0", 
    borderRadius: 16, 
    justifyContent: "center", 
    alignItems: "center" 
  },
  selectedDayButton: { backgroundColor: COLORS.primary, elevation: 4, shadowColor: COLORS.primary, shadowOpacity: 0.3, shadowRadius: 5 },
  dateText: { fontSize: 18, fontWeight: "bold", color: "#444" },
  dayNameText: { fontSize: 10, color: "#888", fontWeight: "600" },
  selectedText: { color: "#fff" },
  selectedSubtext: { color: "#E0F2F1" },
  dot: { width: 4, height: 4, borderRadius: 2, marginTop: 6 },

  sectionTitleRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 20 },
  sectionTitle: { fontSize: 16, fontWeight: "bold", color: "#333" },
  countText: { fontSize: 11, color: "#999" },

  timelineWrapper: { paddingLeft: 10 },
  timelineLine: { 
    position: "absolute", 
    left: 19, 
    top: 10, 
    bottom: 10, 
    width: 1, 
    backgroundColor: "#EEE" 
  },
  timelineItem: { flexDirection: "row", marginBottom: 15 },
  dotContainer: { width: 40, alignItems: "center", justifyContent: "center" },
  timelineDot: { 
    width: 12, 
    height: 12, 
    borderRadius: 6, 
    backgroundColor: "#FFF", 
    borderWidth: 2, 
    borderColor: COLORS.primary,
    zIndex: 10
  },
  card: { 
    flex: 1, 
    backgroundColor: "#FFF", 
    borderRadius: 15, 
    padding: 12, 
    borderWidth: 1, 
    borderColor: "#F0F0F0",
    shadowColor: "#000",
    shadowOpacity: 0.02,
    shadowRadius: 5,
    elevation: 1
  },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", marginBottom: 8 },
  timeContainer: { flexDirection: "row", alignItems: "center", gap: 5 },
  timeText: { fontSize: 12, fontWeight: "bold", color: "#333" },
  typeTag: { backgroundColor: "#F5F5F5", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  typeText: { fontSize: 9, fontWeight: "700", color: "#666" },
  patientRow: { flexDirection: "row", alignItems: "center", gap: 5 },
  patientName: { fontSize: 12, color: "#777" },

  emptyState: { alignItems: "center", marginTop: 60 },
  emptyIconCircle: { width: 70, height: 70, borderRadius: 35, backgroundColor: "#F5F5F5", justifyContent: "center", alignItems: "center", marginBottom: 15 },
  emptyTitle: { fontSize: 16, fontWeight: "bold", color: "#555" },
  emptySubtitle: { fontSize: 13, color: "#999", marginTop: 5 }
});