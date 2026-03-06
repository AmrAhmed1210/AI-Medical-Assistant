import React from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import { Star, MapPin, Clock } from "lucide-react-native";

const COLORS = {
  primary: "#1E9E84",
  background: "#F4F6FA",
  white: "#FFFFFF",
  black: "#000000",
  gray: "#8A8A8A",
  lightGray: "#E8E8E8",
};

interface Doctor {
  id: string | number;
  name: string;
  specialty: string;
  rating?: number;
  reviewCount?: number;
  location?: string;
  experience?: string;
  consultationFee?: number;
  imageUrl?: string;
  isAvailable?: boolean;
}

interface DoctorCardProps {
  doctor: Doctor;
}

export default function DoctorCard({ doctor }: DoctorCardProps) {
  const router = useRouter();

  return (
    <TouchableOpacity
      style={styles.card}
      onPress={() =>
        router.push({
          pathname: "/(patient)/doctor-details",
          params: { doctorId: String(doctor.id) },
        })
      }
      activeOpacity={0.85}
    >
      <View style={styles.cardContent}>
        {/* Avatar */}
        <View style={styles.avatarContainer}>
          {doctor.imageUrl ? (
            <Image source={{ uri: doctor.imageUrl }} style={styles.avatar} />
          ) : (
            <View style={styles.avatarFallback}>
              <Text style={styles.avatarText}>
                {doctor.name?.charAt(0)?.toUpperCase() || "D"}
              </Text>
            </View>
          )}
          {doctor.isAvailable && <View style={styles.availableDot} />}
        </View>

        {/* Info */}
        <View style={styles.info}>
          <Text style={styles.name} numberOfLines={1}>
            {doctor.name}
          </Text>
          <Text style={styles.specialty} numberOfLines={1}>
            {doctor.specialty}
          </Text>

          <View style={styles.row}>
            {doctor.rating != null && (
              <View style={styles.ratingRow}>
                <Star size={12} color="#FFB300" fill="#FFB300" />
                <Text style={styles.ratingText}>
                  {Number(doctor.rating).toFixed(1)}
                </Text>
                {doctor.reviewCount != null && (
                  <Text style={styles.reviewCount}>({doctor.reviewCount})</Text>
                )}
              </View>
            )}
            {doctor.location && (
              <View style={styles.locationRow}>
                <MapPin size={12} color={COLORS.gray} />
                <Text style={styles.locationText} numberOfLines={1}>
                  {doctor.location}
                </Text>
              </View>
            )}
          </View>
        </View>

        {/* Fee */}
        <View style={styles.feeContainer}>
          {doctor.consultationFee != null && (
            <>
              <Text style={styles.feeAmount}>${doctor.consultationFee}</Text>
              <Text style={styles.feeLabel}>Visit</Text>
            </>
          )}
          <View style={styles.bookBtn}>
            <Text style={styles.bookBtnText}>Book</Text>
          </View>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 16,
    marginHorizontal: 16,
    marginVertical: 6,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 8,
    elevation: 3,
  },
  cardContent: {
    flexDirection: "row",
    alignItems: "center",
    padding: 14,
    gap: 12,
  },
  avatarContainer: {
    position: "relative",
  },
  avatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: COLORS.lightGray,
  },
  avatarFallback: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: COLORS.primary + "20",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: {
    fontSize: 22,
    fontWeight: "700",
    color: COLORS.primary,
  },
  availableDot: {
    position: "absolute",
    bottom: 2,
    right: 2,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#4CAF50",
    borderWidth: 2,
    borderColor: COLORS.white,
  },
  info: {
    flex: 1,
    gap: 3,
  },
  name: {
    fontSize: 15,
    fontWeight: "700",
    color: COLORS.black,
  },
  specialty: {
    fontSize: 13,
    color: COLORS.primary,
    fontWeight: "500",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 2,
  },
  ratingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
  },
  ratingText: {
    fontSize: 12,
    fontWeight: "600",
    color: COLORS.black,
  },
  reviewCount: {
    fontSize: 11,
    color: COLORS.gray,
  },
  locationRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
  },
  locationText: {
    fontSize: 12,
    color: COLORS.gray,
    maxWidth: 100,
  },
  feeContainer: {
    alignItems: "center",
    gap: 4,
  },
  feeAmount: {
    fontSize: 16,
    fontWeight: "700",
    color: COLORS.primary,
  },
  feeLabel: {
    fontSize: 11,
    color: COLORS.gray,
  },
  bookBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    marginTop: 4,
  },
  bookBtnText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: "600",
  },
});