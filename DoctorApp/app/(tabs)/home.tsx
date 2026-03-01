import { View, Text, StyleSheet, ScrollView, Image, TouchableOpacity } from "react-native";
import { COLORS } from "../../constants/colors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEffect, useState } from "react";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

export default function HomeScreen() {
  const router = useRouter();
  const [userName, setUserName] = useState("");

  useEffect(() => {
    const loadUser = async () => {
      const name = await AsyncStorage.getItem("userName");
      if (name) setUserName(name);
    };
    loadUser();
  }, []);

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      
      {/* Welcome with Profile Image */}
      <View style={styles.header}>
        <View style={styles.welcomeContainer}>
          <Text style={styles.welcome}>
            Welcome Back 👋
          </Text>
          <Text style={styles.userName}>
            {userName ? userName : "Guest"}
          </Text>
        </View>
        
        <TouchableOpacity onPress={() => router.push("/(tabs)/profile")}>
          <View style={styles.profileImage}>
            <Ionicons name="person" size={30} color={COLORS.primary} />
          </View>
        </TouchableOpacity>
      </View>

      {/* Green Card */}
      <View style={styles.greenCard}>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardTitle}>
            Looking for desired doctor?
          </Text>
          <TouchableOpacity style={styles.searchBtn}>
            <Text style={styles.searchText}>Search now</Text>
          </TouchableOpacity>
        </View>

        <Image
          source={require("../../assets/doctor3.jpg")}
          style={styles.greenDoctorImage}
        />
      </View>

      {/* Popular Doctors */}
      <View style={styles.popularHeader}>
        <Text style={styles.sectionTitle}>Popular Doctors</Text>
        <TouchableOpacity>
          <Text style={styles.seeAllText}>See All</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.doctorCard}>
        <Image
          source={require("../../assets/doctor1.jpg")}
          style={styles.doctorImage}
        />
        <View style={{ flex: 1 }}>
          <Text style={styles.doctorName}>Dr. Chloe Kelly</Text>
          <Text style={styles.speciality}>M.Ch (Neuro)</Text>
          <View style={styles.ratingContainer}>
            <Text style={styles.rating}>⭐ 4.5</Text>
            <Text style={styles.reviews}>(2530)</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.bookBtn}>
          <Text style={styles.bookText}>Book</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.doctorCard}>
        <Image
          source={require("../../assets/doctor2.jpg")}
          style={styles.doctorImage}
        />
        <View style={{ flex: 1 }}>
          <Text style={styles.doctorName}>Dr. Lauren Hemp</Text>
          <Text style={styles.speciality}>Spinal Surgery</Text>
          <View style={styles.ratingContainer}>
            <Text style={styles.rating}>⭐ 4.5</Text>
            <Text style={styles.reviews}>(2530)</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.bookBtn}>
          <Text style={styles.bookText}>Book</Text>
        </TouchableOpacity>
      </View>

      {/* Add bottom padding for scroll */}
      <View style={{ height: 30 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.white,
    padding: 20,
  },

  header: {
    marginTop: 50,
    marginBottom: 25,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  welcomeContainer: {
    flex: 1,
  },

  welcome: {
    fontSize: 16,
    color: "#777",
    marginBottom: 4,
  },

  userName: {
    fontSize: 24,
    fontWeight: "bold",
    color: COLORS.primary,
  },

  profileImage: {
    width: 55,
    height: 55,
    borderRadius: 27.5,
    backgroundColor: '#f0f8ff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: COLORS.primary,
  },

  greenCard: {
    backgroundColor: COLORS.primary,
    borderRadius: 25,
    padding: 20,
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 30,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },

  cardTitle: {
    color: "white",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 15,
  },

  searchBtn: {
    backgroundColor: "white",
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
    alignSelf: "flex-start",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },

  searchText: {
    color: COLORS.primary,
    fontWeight: "600",
    fontSize: 14,
  },

  greenDoctorImage: {
    width: 100,
    height: 100,
    resizeMode: "contain",
    marginLeft: 10,
  },

  popularHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },

  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: '#333',
  },

  seeAllText: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '600',
  },

  doctorCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f7f7f7",
    padding: 15,
    borderRadius: 18,
    marginBottom: 15,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 2,
  },

  doctorImage: {
    width: 70,
    height: 70,
    borderRadius: 35,
    marginRight: 15,
    borderWidth: 2,
    borderColor: COLORS.primary + '20',
  },

  doctorName: {
    fontWeight: "bold",
    fontSize: 16,
    color: '#333',
    marginBottom: 2,
  },

  speciality: {
    color: "#777",
    fontSize: 13,
    marginBottom: 4,
  },

  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  rating: {
    fontSize: 13,
    color: '#333',
    fontWeight: '500',
  },

  reviews: {
    fontSize: 12,
    color: '#999',
    marginLeft: 4,
  },

  bookBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 12,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
  },

  bookText: {
    color: "white",
    fontWeight: "600",
    fontSize: 14,
  },
});