import React, { useState, useEffect, useCallback } from "react";
import { View, Text, FlatList, StyleSheet, ActivityIndicator, TouchableOpacity, RefreshControl } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { COLORS } from "../../constants/colors";
import { getDoctorById } from "../../services/doctorService";
import DoctorCard from "../../components/DoctorCard";

export default function FollowedDoctorsScreen() {
  const router = useRouter();
  const [doctors, setDoctors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchFollowed = useCallback(async () => {
    try {
      const stored = await AsyncStorage.getItem('followedDoctors')
      const followedIds = stored ? JSON.parse(stored) : []
      
      if (followedIds.length === 0) {
        setDoctors([]);
        return;
      }

      // Fetch details for each followed ID
      const details = await Promise.all(
        followedIds.map(async (id: number) => {
          try {
            return await getDoctorById(id);
          } catch {
            return null;
          }
        })
      );
      
      setDoctors(details.filter(d => d !== null));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchFollowed();
  }, [fetchFollowed]);

  if (loading && !refreshing) return <View style={styles.center}><ActivityIndicator size="large" color={COLORS.primary} /></View>;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color="#1A1A1A" />
        </TouchableOpacity>
        <Text style={styles.title}>Followed Doctors</Text>
      </View>

      <FlatList
        data={doctors}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => (
          <DoctorCard 
            doctor={{
              ...item,
              id: String(item.id),
              experience: `${item.yearsExperience || item.experience || 0} yrs`,
              isProfileComplete: true
            }} 
          />
        )}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); fetchFollowed(); }} colors={[COLORS.primary]} />
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="heart-outline" size={64} color="#DDD" />
            <Text style={styles.emptyTxt}>You have not followed any doctors yet.</Text>
            <TouchableOpacity style={styles.findBtn} onPress={() => router.push("/(patient)/doctors")}>
              <Text style={styles.findBtnTxt}>Find Doctors</Text>
            </TouchableOpacity>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F4F6FA" },
  center:    { flex: 1, justifyContent: "center", alignItems: "center" },
  header:    { flexDirection: "row", alignItems: "center", paddingHorizontal: 16, paddingTop: 60, paddingBottom: 20, backgroundColor: "#fff" },
  backBtn:   { marginRight: 12 },
  title:     { fontSize: 20, fontWeight: "800", color: "#1A1A1A" },
  list:      { paddingVertical: 12 },
  empty:     { alignItems: "center", marginTop: 100, paddingHorizontal: 40 },
  emptyTxt:  { fontSize: 15, color: "#AAA", textAlign: "center", marginTop: 16, lineHeight: 22 },
  findBtn:   { marginTop: 24, backgroundColor: COLORS.primary, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 24 },
  findBtnTxt:{ color: "#fff", fontWeight: "700", fontSize: 14 }
});
