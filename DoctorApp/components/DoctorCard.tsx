import React, { useState, useEffect, useRef } from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet, Modal, TextInput, Animated } from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { MapPin, Clock, MessageCircle, Heart } from "lucide-react-native";
import RatingStars from "./RatingStars";

const COLORS = {
  primary: "#1E9E84",
  white: "#FFFFFF",
  black: "#1A1A1A",
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
  isProfileComplete?: boolean;
  hasSchedule?: boolean;
}

export default function DoctorCard({ doctor, highlight }: { doctor: Doctor; highlight?: boolean }) {
  const router = useRouter();
  const [modalVisible, setModalVisible] = useState(false);
  const [message, setMessage] = useState("");
  const [showBadge, setShowBadge] = useState(false);
  const [isFollowed, setIsFollowed] = useState(false);
  const [followBusy, setFollowBusy] = useState(false);

  const glowAnim = useRef(new Animated.Value(0)).current;
  const badgeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (highlight) {
      setShowBadge(true);
      Animated.sequence([
        Animated.timing(glowAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.timing(glowAnim, { toValue: 0, duration: 300, delay: 1400, useNativeDriver: true }),
      ]).start();

      Animated.sequence([
        Animated.timing(badgeAnim, { toValue: 1, duration: 200, useNativeDriver: true }),
        Animated.timing(badgeAnim, { toValue: 0, duration: 200, delay: 2600, useNativeDriver: true }),
      ]).start(() => setShowBadge(false));
    }
  }, [highlight]);

  useEffect(() => {
    let mounted = true;
    const loadFollowState = async () => {
      try {
        const stored = await AsyncStorage.getItem("followedDoctors");
        const ids: number[] = stored ? JSON.parse(stored) : [];
        if (!mounted) return;
        setIsFollowed(ids.includes(Number(doctor.id)));
      } catch {
        if (mounted) setIsFollowed(false);
      }
    };
    loadFollowState();
    return () => { mounted = false; };
  }, [doctor.id]);

  const goToDetails = () =>
    router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doctor.id) } });

  const toggleFollow = async () => {
    if (followBusy) return;
    setFollowBusy(true);
    try {
      const stored = await AsyncStorage.getItem("followedDoctors");
      const ids: number[] = stored ? JSON.parse(stored) : [];
      const doctorId = Number(doctor.id);
      const next = ids.includes(doctorId)
        ? ids.filter((id) => id !== doctorId)
        : [...ids, doctorId];
      await AsyncStorage.setItem("followedDoctors", JSON.stringify(next));
      setIsFollowed(next.includes(doctorId));
    } finally {
      setFollowBusy(false);
    }
  };

  return (
    <>
      <Animated.View style={[styles.cardWrapper, highlight && {
        shadowOpacity: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [0.08, 0.25] }),
        shadowRadius: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [10, 20] }),
        elevation: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [4, 12] }),
      }]}>
        <TouchableOpacity style={[styles.card, highlight && styles.cardHighlightBorder]} onPress={goToDetails} activeOpacity={0.88}>
          <View style={styles.topRow}>
            <View style={styles.avatarWrap}>
              {doctor.imageUrl ? (
                <Image source={{ uri: doctor.imageUrl }} style={styles.avatar} />
              ) : (
                <View style={styles.avatarFallback}>
                  <Text style={styles.avatarTxt}>{doctor.name?.charAt(0)?.toUpperCase() || "D"}</Text>
                </View>
              )}
              <View style={[styles.dot, { backgroundColor: doctor.isAvailable ? "#22C55E" : "#CBD5E1" }]} />
            </View>

            <View style={styles.infoCol}>
              <Text style={styles.name} numberOfLines={1}>{doctor.name}</Text>
              <View style={styles.specBadge}>
                <Text style={styles.specTxt}>{doctor.specialty}</Text>
              </View>
              <View style={styles.statusRow}>
                <View style={[styles.statusDot, { backgroundColor: doctor.isAvailable ? "#22C55E" : "#CBD5E1" }]} />
                <Text style={[styles.statusTxt, { color: doctor.isAvailable ? "#16A34A" : "#94A3B8" }]}>
                  {doctor.isAvailable ? "Online" : "Offline"}
                </Text>
              </View>
              
              <View style={styles.metaRow}>
                {doctor.rating != null && (
                  <RatingStars rating={doctor.rating} reviewCount={doctor.reviewCount} size={13} />
                )}
                {doctor.experience && (
                  <View style={styles.metaItem}>
                    <Clock size={11} color={COLORS.gray} />
                    <Text style={styles.metaGray}>{doctor.experience}</Text>
                  </View>
                )}
              </View>

              {doctor.location && (
                <View style={styles.metaItem}>
                  <MapPin size={11} color={COLORS.gray} />
                  <Text style={styles.metaGray} numberOfLines={1}>{doctor.location}</Text>
                </View>
              )}
            </View>

            <View style={styles.feeCol}>
              <TouchableOpacity
                style={[styles.followBtn, isFollowed && styles.followBtnActive]}
                onPress={(e) => { e.stopPropagation(); toggleFollow().catch(() => undefined); }}
              >
                <Heart size={13} color={isFollowed ? "#fff" : "#E11D48"} fill={isFollowed ? "#fff" : "transparent"} />
              </TouchableOpacity>
              {doctor.consultationFee != null && (
                <>
                  <Text style={styles.feeAmt}>${doctor.consultationFee}</Text>
                  <Text style={styles.feeLbl}>/ visit</Text>
                </>
              )}
            </View>
          </View>

          <View style={styles.divider} />

          <View style={styles.bottomRow}>
            <TouchableOpacity
              style={styles.consultBtn}
              onPress={(e) => { e.stopPropagation(); setModalVisible(true); }}
              activeOpacity={0.8}
            >
              <MessageCircle size={13} color={COLORS.primary} />
              <Text style={styles.consultTxt}>Consultation</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.bookBtn} onPress={goToDetails} activeOpacity={0.8}>
              <Text style={styles.bookTxt}>Details</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>

        {showBadge && (
          <Animated.View style={[styles.updatedBadge, {
            opacity: badgeAnim,
            transform: [{ translateY: badgeAnim.interpolate({ inputRange: [0, 1], outputRange: [-10, 0] }) }],
          }]}>
            <Text style={styles.updatedBadgeText}>Updated</Text>
          </Animated.View>
        )}
      </Animated.View>

      <Modal visible={modalVisible} transparent animationType="slide">
        <View style={styles.overlay}>
          <View style={styles.sheet}>
            <View style={styles.sheetHeaderRow}>
              <View style={styles.sheetAvatar}>
                <Text style={styles.sheetAvatarTxt}>{doctor.name?.charAt(0)?.toUpperCase() || "D"}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.sheetTitle}>Medical Consultation</Text>
                <Text style={styles.sheetSub}>{doctor.name} · {doctor.specialty}</Text>
              </View>
              <TouchableOpacity onPress={() => setModalVisible(false)} style={styles.closeBtn}>
                <Text style={styles.closeTxt}>✕</Text>
              </TouchableOpacity>
            </View>
            <TextInput
              style={styles.input}
              placeholder="Describe your symptoms..."
              placeholderTextColor="#BBB"
              value={message}
              onChangeText={setMessage}
              multiline
            />
            <TouchableOpacity style={styles.sendBtn} onPress={() => { setModalVisible(false); router.push({ pathname: "/(patient)/messages", params: { doctorName: doctor.name } }); }}>
              <MessageCircle size={16} color="#FFF" />
              <Text style={styles.sendTxt}>Send Message</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  cardWrapper: { marginHorizontal: 16, marginVertical: 7, borderRadius: 18, shadowColor: "#1E9E84", shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.08, shadowRadius: 10, elevation: 4 },
  card: { backgroundColor: COLORS.white, borderRadius: 18, overflow: "hidden" },
  cardHighlightBorder: { borderWidth: 2, borderColor: "#1E9E84" },
  topRow: { flexDirection: "row", padding: 14, gap: 12, alignItems: "flex-start" },
  avatarWrap: { position: "relative" },
  avatar: { width: 62, height: 62, borderRadius: 31, backgroundColor: COLORS.lightGray },
  avatarFallback: { width: 62, height: 62, borderRadius: 31, backgroundColor: COLORS.primary, alignItems: "center", justifyContent: "center" },
  avatarTxt: { fontSize: 24, fontWeight: "800", color: "#FFF" },
  dot: { position: "absolute", bottom: 1, right: 1, width: 14, height: 14, borderRadius: 7, borderWidth: 2, borderColor: "#FFF" },
  infoCol: { flex: 1, gap: 5 },
  name: { fontSize: 15, fontWeight: "700", color: COLORS.black },
  specBadge: { alignSelf: "flex-start", backgroundColor: "#E8F7F4", paddingHorizontal: 10, paddingVertical: 3, borderRadius: 20 },
  specTxt: { fontSize: 11, color: COLORS.primary, fontWeight: "600" },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 5 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusTxt: { fontSize: 11, fontWeight: "600" },
  metaRow: { flexDirection: "row", gap: 10, flexWrap: "wrap", alignItems: 'center' },
  metaItem: { flexDirection: "row", alignItems: "center", gap: 3 },
  metaGray: { fontSize: 11, color: COLORS.gray },
  feeCol: { alignItems: "flex-end", paddingTop: 2 },
  followBtn: { width: 26, height: 26, borderRadius: 13, backgroundColor: "#FFF1F2", alignItems: "center", justifyContent: "center", marginBottom: 8 },
  followBtnActive: { backgroundColor: "#E11D48" },
  feeAmt: { fontSize: 17, fontWeight: "800", color: COLORS.primary },
  feeLbl: { fontSize: 10, color: COLORS.gray },
  divider: { height: 1, backgroundColor: "#F0F0F0", marginHorizontal: 14 },
  bottomRow: { flexDirection: "row", gap: 10, padding: 12 },
  consultBtn: { flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, backgroundColor: "#EEF9F6", borderWidth: 1.5, borderColor: COLORS.primary, paddingVertical: 10, borderRadius: 22 },
  consultTxt: { color: COLORS.primary, fontSize: 12, fontWeight: "600" },
  bookBtn: { flex: 1, backgroundColor: COLORS.primary, paddingVertical: 10, borderRadius: 22, alignItems: "center" },
  bookTxt: { color: "#FFF", fontSize: 13, fontWeight: "700" },
  updatedBadge: { position: "absolute", top: 10, right: 10, backgroundColor: "#1E9E84", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  updatedBadgeText: { color: "#FFFFFF", fontSize: 11, fontWeight: "700" },
  overlay: { flex: 1, backgroundColor: "#00000070", justifyContent: "flex-end" },
  sheet: { backgroundColor: "#FFF", borderTopLeftRadius: 28, borderTopRightRadius: 28, padding: 22, paddingBottom: 38 },
  sheetHeaderRow: { flexDirection: "row", alignItems: "center", gap: 12, marginBottom: 18 },
  sheetAvatar: { width: 46, height: 46, borderRadius: 23, backgroundColor: COLORS.primary, alignItems: "center", justifyContent: "center" },
  sheetAvatarTxt: { fontSize: 18, fontWeight: "800", color: "#FFF" },
  sheetTitle: { fontSize: 15, fontWeight: "700", color: "#1A1A1A" },
  sheetSub: { fontSize: 12, color: COLORS.gray, marginTop: 1 },
  closeBtn: { width: 32, height: 32, borderRadius: 16, backgroundColor: "#F5F5F5", alignItems: "center", justifyContent: "center" },
  closeTxt: { fontSize: 12, color: COLORS.gray },
  input: { height: 110, borderWidth: 1.5, borderColor: "#E8E8E8", borderRadius: 14, padding: 12, marginBottom: 14, fontSize: 14, color: "#1A1A1A", textAlignVertical: "top", backgroundColor: "#FAFAFA" },
  sendBtn: { backgroundColor: COLORS.primary, paddingVertical: 14, borderRadius: 14, alignItems: "center", flexDirection: "row", justifyContent: "center", gap: 8 },
  sendTxt: { color: "#FFF", fontSize: 15, fontWeight: "700" },
});
