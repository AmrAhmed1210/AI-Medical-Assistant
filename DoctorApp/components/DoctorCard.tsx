import React, { useState, useEffect, useRef } from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet, Modal, TextInput, Animated } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { MapPin, Clock, MessageCircle, Heart as LucideHeart } from "lucide-react-native";
const Heart = LucideHeart as any;
const MapPinIcon = MapPin as any;
const ClockIcon = Clock as any;
const MessageCircleIcon = MessageCircle as any;
import RatingStars from "./RatingStars";
import { LinearGradient } from "expo-linear-gradient";
import { checkIfFollowed, setFollowed } from "../services/followService";

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
  photoUrl?: string | null;
  isAvailable?: boolean;
  isProfileComplete?: boolean;
  hasSchedule?: boolean;
}

export default function DoctorCard({ doctor, highlight, compact }: { doctor: Doctor; highlight?: boolean; compact?: boolean }) {
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
        if (!mounted) return;
        setIsFollowed(await checkIfFollowed(Number(doctor.id)));
      } catch {
        if (mounted) setIsFollowed(false);
      }
    };
    loadFollowState();
    return () => { mounted = false; };
  }, [doctor.id]);

  const goToDetails = () =>
    router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doctor.id) } });

  const openConsultationModal = () => {
    setModalVisible(true);
  };

  const toggleFollow = async () => {
    if (followBusy) return;
    setFollowBusy(true);
    try {
      const doctorId = Number(doctor.id);
      const followed = await checkIfFollowed(doctorId);
      const next = !followed;
      await setFollowed(doctorId, next);
      setIsFollowed(next);
    } finally {
      setFollowBusy(false);
    }
  };

  const scaleAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(fadeAnim, {
      toValue: 1,
      tension: 20,
      friction: 7,
      useNativeDriver: true,
    }).start();
  }, []);

  const onPressIn = () => {
    Animated.spring(scaleAnim, { toValue: 0.97, useNativeDriver: true }).start();
  };
  const onPressOut = () => {
    Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true }).start();
  };

  return (
    <>
      <Animated.View style={[styles.cardWrapper, {
        opacity: fadeAnim,
        transform: [
          { scale: scaleAnim },
          { translateY: fadeAnim.interpolate({ inputRange: [0, 1], outputRange: [20, 0] }) }
        ]
      }, highlight && {
        shadowOpacity: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [0.08, 0.25] }),
        shadowRadius: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [10, 20] }),
        elevation: glowAnim.interpolate({ inputRange: [0, 1], outputRange: [4, 12] }),
      }]}>
        <TouchableOpacity
          style={[styles.card, highlight && styles.cardHighlightBorder]}
          onPress={goToDetails}
          activeOpacity={1}
          onPressIn={onPressIn}
          onPressOut={onPressOut}
        >
          <LinearGradient colors={["#fff", "#FDFDFD"]} style={styles.cardGradient}>
            <View style={styles.topRow}>
              <View style={styles.avatarWrap}>
                <View style={styles.avatarGlow} />
                {doctor.photoUrl || (doctor.imageUrl && !doctor.imageUrl.includes('default')) ? (
                  <Image source={{ uri: doctor.photoUrl || doctor.imageUrl }} style={styles.avatar} />
                ) : (
                  <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.avatar} />
                )}
                <View style={[styles.statusDot, { backgroundColor: doctor.isAvailable ? "#10B981" : "#94A3B8" }]} />
              </View>

              <View style={styles.infoCol}>
                <View style={styles.nameRow}>
                  <Text style={styles.name} numberOfLines={1}>{doctor.name}</Text>
                  <View style={styles.verifiedBadge}><Ionicons name="checkmark-circle" size={10} color="#059669" /></View>
                </View>
                <Text style={styles.specialty}>{doctor.specialty}</Text>

                <View style={styles.metaRow}>
                  <View style={styles.metaItem}>
                    <Ionicons name="star" size={10} color="#FBBF24" />
                    <Text style={styles.metaTxt}>{doctor.rating || "5.0"}</Text>
                  </View>
                  <View style={styles.dotSeparator} />
                  <View style={styles.metaItem}>
                    <ClockIcon size={10} color="#64748B" />
                    <Text style={styles.metaTxt}>{doctor.experience || "5+ yrs"}</Text>
                  </View>
                </View>
              </View>

              <View style={styles.feeCol}>
                <Text style={styles.feeVal}>${doctor.consultationFee}</Text>
                <TouchableOpacity
                  style={[styles.miniHeart, isFollowed && styles.miniHeartActive]}
                  onPress={(e) => { e.stopPropagation(); toggleFollow().catch(() => undefined); }}
                >
                  <Heart size={12} stroke={isFollowed ? "#fff" : "#EF4444"} fill={isFollowed ? "#fff" : "transparent"} />
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.actionRow}>
              <TouchableOpacity
                style={styles.actionBtn}
                onPress={goToDetails}
              >
                <Text style={styles.actionBtnText}>View Profile</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.actionBtn, styles.actionBtnPrimary]}
                onPress={(e) => { e.stopPropagation(); openConsultationModal(); }}
              >
                <Text style={styles.actionBtnTextPrimary}>Consult</Text>
              </TouchableOpacity>
            </View>
          </LinearGradient>
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
              {doctor.photoUrl || doctor.imageUrl ? (
                <Image source={{ uri: doctor.photoUrl || doctor.imageUrl }} style={styles.sheetAvatar} />
              ) : (
                <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.sheetAvatar} />
              )}
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
            <TouchableOpacity
              style={styles.sendBtn}
              onPress={() => {
                setModalVisible(false);
                router.push({
                  pathname: "/(patient)/messages",
                  params: {
                    doctorId: String(doctor.id),
                    doctorName: doctor.name,
                    initialMessage: message.trim(),
                  },
                });
                setMessage("");
              }}
            >
              <MessageCircleIcon size={16} stroke="#FFF" />
              <Text style={styles.sendTxt}>Send Message</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  cardWrapper: { marginHorizontal: 16, marginVertical: 6, borderRadius: 18, shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 3 },
  card: { borderRadius: 18, overflow: "hidden", backgroundColor: "#fff", borderWidth: 1, borderColor: "#F1F5F9" },
  cardGradient: { padding: 12 },
  cardHighlightBorder: { borderColor: "#FBBF24", borderWidth: 1.2 },
  topRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  avatarWrap: { position: 'relative' },
  avatarGlow: { position: 'absolute', width: 50, height: 50, borderRadius: 25, backgroundColor: '#ECFDF5', top: -2, left: -2 },
  avatar: { width: 46, height: 46, borderRadius: 23, borderWidth: 1.5, borderColor: '#fff' },
  statusDot: { position: 'absolute', bottom: 1, right: 1, width: 10, height: 10, borderRadius: 5, borderWidth: 2, borderColor: '#fff' },
  infoCol: { flex: 1, gap: 2 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  name: { fontSize: 14, fontWeight: '700', color: '#1E293B' },
  verifiedBadge: { padding: 0 },
  specialty: { fontSize: 11, color: '#64748B', fontWeight: '500' },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 2 },
  metaItem: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  metaTxt: { fontSize: 10, color: '#64748B', fontWeight: '700' },
  dotSeparator: { width: 3, height: 3, borderRadius: 1.5, backgroundColor: '#CBD5E1' },
  feeCol: { alignItems: 'flex-end', gap: 8 },
  feeVal: { fontSize: 13, fontWeight: '800', color: '#059669' },
  actionRow: { flexDirection: 'row', gap: 8, marginTop: 12 },
  actionBtn: { flex: 1, backgroundColor: '#F1F5F9', paddingVertical: 10, borderRadius: 12, alignItems: 'center' },
  actionBtnText: { fontSize: 12, fontWeight: '700', color: '#475569' },
  actionBtnPrimary: { backgroundColor: COLORS.primary },
  actionBtnTextPrimary: { color: '#fff' },
  miniHeart: { width: 26, height: 26, borderRadius: 13, backgroundColor: '#FFF1F2', justifyContent: 'center', alignItems: 'center' },
  miniHeartActive: { backgroundColor: '#EF4444' },
  updatedBadge: { position: "absolute", top: 10, right: 10, backgroundColor: "#FBBF24", paddingHorizontal: 8, paddingVertical: 2, borderRadius: 8 },
  updatedBadgeText: { color: "#000", fontSize: 10, fontWeight: "800" },
  overlay: { flex: 1, backgroundColor: "rgba(15, 23, 42, 0.6)", justifyContent: "flex-end" },
  sheet: { backgroundColor: "#FFF", borderTopLeftRadius: 32, borderTopRightRadius: 32, padding: 25, paddingBottom: 40 },
  sheetHeaderRow: { flexDirection: "row", alignItems: "center", gap: 15, marginBottom: 20 },
  sheetAvatar: { width: 50, height: 50, borderRadius: 25, backgroundColor: '#F1F5F9' },
  sheetTitle: { fontSize: 17, fontWeight: "700", color: "#1E293B" },
  sheetSub: { fontSize: 13, color: '#64748B', marginTop: 2 },
  closeBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: "#F1F5F9", alignItems: "center", justifyContent: "center" },
  closeTxt: { fontSize: 14, color: '#64748B' },
  input: { height: 120, backgroundColor: "#F8FAFC", borderRadius: 18, padding: 15, marginBottom: 20, fontSize: 14, color: "#1E293B", textAlignVertical: "top", borderWidth: 1, borderColor: "#F1F5F9" },
  sendBtn: { backgroundColor: "#059669", height: 56, borderRadius: 18, alignItems: "center", flexDirection: "row", justifyContent: "center", gap: 10, elevation: 4 },
  sendTxt: { color: "#FFF", fontSize: 15, fontWeight: "700" },
});
