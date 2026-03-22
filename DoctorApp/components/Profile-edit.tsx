import React, { useState } from "react";
import { 
  View, Text, StyleSheet, Image, ScrollView, TouchableOpacity, 
  StatusBar, TextInput, Switch, Alert 
} from "react-native";
import { useRouter } from "expo-router";
import { ChevronLeft, Camera, Plus, Trash2, Save } from "lucide-react-native";
import { COLORS } from "../constants/colors";
interface ProfileEditProps {
  onClose: () => void;
}
export default function ProfileEdit({ onClose }: ProfileEditProps) {
  const router = useRouter();
  
  const [isEditing, setIsEditing] = useState(false);
  const [doctorData, setDoctorData] = useState({
    name: "Dr. Chloe Kelly",
    qualifications: "M.B.B.S, M.D - Cardiology",
    bio: "Dr. Chloe Kelly is a top-rated cardiologist with over 10 years of experience. She specializes in heart failure, hypertension, and preventive cardiology.",
    fees: "50",
    tags: ["Heart Specialist", "Hypertension", "Preventive"],
  });
   
  const handleSave = () => {
    setIsEditing(false);
    Alert.alert("Success", "Profile updated successfully!");
  };
 

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <View style={styles.header}>
        <TouchableOpacity onPress={() => onClose()} style={styles.backButton}>
          <ChevronLeft size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{isEditing ? "Edit Profile" : "My Public Profile"}</Text>
        <TouchableOpacity 
          onPress={() => isEditing ? handleSave() : setIsEditing(true)}
            
        >
          {isEditing ? <Save size={20} color={COLORS.primary} /> : <Text style={styles.editText}>Edit</Text>}
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 100 }}>
        
        <View style={styles.doctorInfoContainer}>
          <View style={styles.imageWrapper}>
            <Image source={{ uri: "https://via.placeholder.com/150" }} style={styles.doctorImage} />
            {isEditing && (
              <TouchableOpacity style={styles.cameraIcon}>
                <Camera size={16} color="#FFF" />
              </TouchableOpacity>
            )}
          </View>
          
          {isEditing ? (
            <View style={styles.inputGroup}>
              <TextInput 
                style={styles.nameInput} 
                value={doctorData.name} 
                onChangeText={(t) => setDoctorData({...doctorData, name: t})}
                placeholder="Doctor Name"
              />
              <TextInput 
                style={styles.qualInput} 
                value={doctorData.qualifications} 
                onChangeText={(t) => setDoctorData({...doctorData, qualifications: t})}
                placeholder="Qualifications"
              />
            </View>
          ) : (
            <>
              <Text style={styles.doctorName}>{doctorData.name}</Text>
              <Text style={styles.qualifications}>{doctorData.qualifications}</Text>
            </>
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Biography</Text>
          {isEditing ? (
            <TextInput 
              style={styles.bioInput} 
              value={doctorData.bio} 
              multiline
              onChangeText={(t) => setDoctorData({...doctorData, bio: t})}
            />
          ) : (
            <Text style={styles.bioText}>{doctorData.bio}</Text>
          )}
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Specialties (Tags)</Text>
            {isEditing && <Plus size={20} color={COLORS.primary} />}
          </View>
          <View style={styles.tagsContainer}>
            {doctorData.tags.map((tag, index) => (
              <View key={index} style={styles.tag}>
                <Text style={styles.tagText}>{tag}</Text>
                {isEditing && <Trash2 size={12} color="#EF4444" style={{marginLeft: 8}} />}
              </View>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Consultation Fees ($)</Text>
          {isEditing ? (
            <TextInput 
              style={styles.feesInput} 
              value={doctorData.fees} 
              keyboardType="numeric"
              onChangeText={(t) => setDoctorData({...doctorData, fees: t})}
            />
          ) : (
            <View style={styles.feesDisplay}>
              <Text style={styles.feesText}>${doctorData.fees} per session</Text>
            </View>
          )}
        </View>

        <View style={styles.section}>
          <View style={styles.availabilityRow}>
            <View>
              <Text style={styles.sectionTitle}>Accepting Appointments</Text>
              <Text style={styles.subLabel}>Show your profile to patients</Text>
            </View>
            <Switch 
              value={true} 
              trackColor={{ false: "#ddd", true: COLORS.primary }} 
            />
          </View>
        </View>

      </ScrollView>

      {isEditing && (
        <View style={styles.footer}>
          <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
            <Text style={styles.saveButtonText}>Save Changes</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },
  header: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingTop: 60, paddingHorizontal: 20, paddingBottom: 15 },
  backButton: { width: 40, height: 40, justifyContent: "center", alignItems: "center", backgroundColor: "#f5f5f5", borderRadius: 20 },
  headerTitle: { fontSize: 18, fontWeight: "bold", color: "#333" },
  editText: { color: COLORS.primary, fontWeight: "bold" },
  doctorInfoContainer: { alignItems: "center", padding: 20 },
  imageWrapper: { width: 100, height: 100, borderRadius: 50, borderWidth: 3, borderColor: "#E0F2F1", position: 'relative', marginBottom: 15 },
  doctorImage: { width: "100%", height: "100%", borderRadius: 50 },
  cameraIcon: { position: "absolute", bottom: 0, right: 0, backgroundColor: COLORS.primary, padding: 6, borderRadius: 15, borderWidth: 2, borderColor: "#FFF" },
  doctorName: { fontSize: 22, fontWeight: "bold", color: "#333" },
  qualifications: { fontSize: 13, color: "#777", marginTop: 4 },
  inputGroup: { width: "100%", alignItems: "center" },
  nameInput: { fontSize: 20, fontWeight: "bold", color: "#333", borderBottomWidth: 1, borderColor: "#EEE", width: "80%", textAlign: "center", marginBottom: 10 },
  qualInput: { fontSize: 13, color: "#666", borderBottomWidth: 1, borderColor: "#EEE", width: "70%", textAlign: "center" },
  section: { paddingHorizontal: 20, marginTop: 25 },
  sectionHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  sectionTitle: { fontSize: 16, fontWeight: "bold", color: "#333" },
  bioText: { fontSize: 13, color: "#666", lineHeight: 20 },
  bioInput: { fontSize: 13, color: "#666", backgroundColor: "#F9F9F9", borderRadius: 12, padding: 15, textAlignVertical: "top", minHeight: 100, borderWidth: 1, borderColor: "#EEE" },
  tagsContainer: { flexDirection: "row", marginTop: 10, flexWrap: "wrap" },
  tag: { backgroundColor: "#f0f0f0", paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, marginRight: 8, marginBottom: 8, flexDirection: "row", alignItems: "center" },
  tagText: { fontSize: 11, color: "#666", fontWeight: "600" },
  feesInput: { fontSize: 16, fontWeight: "bold", color: COLORS.primary, borderBottomWidth: 1, borderColor: "#EEE", width: 100, paddingVertical: 5 },
  feesDisplay: { backgroundColor: "#F0F9F8", padding: 12, borderRadius: 10, alignSelf: "flex-start" },
  feesText: { color: COLORS.primary, fontWeight: "bold", fontSize: 14 },
  availabilityRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", backgroundColor: "#FBFBFB", padding: 15, borderRadius: 15, borderWidth: 1, borderColor: "#F0F0F0" },
  subLabel: { fontSize: 11, color: "#AAA", marginTop: 2 },
  footer: { position: "absolute", bottom: 0, width: "100%", padding: 20, backgroundColor: "#fff", borderTopWidth: 1, borderTopColor: "#eee" },
  saveButton: { backgroundColor: COLORS.primary, paddingVertical: 16, borderRadius: 15, alignItems: "center" },
  saveButtonText: { color: "#fff", fontSize: 16, fontWeight: "bold" },
});