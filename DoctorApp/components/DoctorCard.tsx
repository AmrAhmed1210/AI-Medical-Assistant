import React, { useState, useEffect, useRef } from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet, Modal, TextInput, Animated, Dimensions } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { MapPin, Clock, MessageCircle, Heart as LucideHeart, Star, ChevronRight, ShieldCheck } from "lucide-react-native";
const Heart = LucideHeart as any;
const MapPinIcon = MapPin as any;
const ClockIcon = Clock as any;
const MessageCircleIcon = MessageCircle as any;
import { LinearGradient } from "expo-linear-gradient";
import { checkIfFollowed, setFollowed } from "../services/followService";
import { useTheme } from "../context/ThemeContext";

const { width: SCREEN_WIDTH } = Dimensions.get("window");

const COLORS = {
  primary: "#1E9E84",
  primaryDark: "#0D5E4E",
  accent: "#10B981",
  white: "#FFFFFF",
  black: "#1A1A1A",
  gray: "#8A8A8A",
  lightGray: "#F1F5F9",
  gold: "#FFB300",
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
  const { theme, isDark, colors } = useTheme();
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
  const ratingValue = Number(doctor.rating || 0);
  const reviewCount = Number(doctor.reviewCount || 0);
  const ratingLabel = reviewCount > 0 && ratingValue > 0 ? ratingValue.toFixed(1) : "New";
  const reviewLabel = reviewCount > 0 ? `(${reviewCount})` : "";

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
          style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }, highlight && styles.cardHighlightBorder]}
          onPress={goToDetails}
          activeOpacity={1}
          onPressIn={onPressIn}
          onPressOut={onPressOut}
        >
          <LinearGradient colors={isDark ? ["#1E293B", "#0F172A"] : ["#fff", "#F8FAFC"]} style={styles.cardGradient}>
            {/* Verified Strip */}
            <View style={styles.verifiedStrip}>
               <ShieldCheck size={10} color="#1E9E84" fill="#E6F4F1" />
               <Text style={styles.verifiedText}>PRO VERIFIED</Text>
            </View>

            <View style={styles.topRow}>
              <View style={styles.avatarWrap}>
                <View style={styles.avatarGlow} />
                <View style={styles.avatarBorder}>
                  {doctor.photoUrl || (doctor.imageUrl && !doctor.imageUrl.includes('default')) ? (
                    <Image source={{ uri: doctor.photoUrl || doctor.imageUrl }} style={styles.avatar} />
                  ) : (
                    <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.avatar} />
                  )}
                </View>
                {doctor.isAvailable && <View style={styles.statusDot} />}
              </View>

              <View style={styles.infoCol}>
                <View style={styles.nameRow}>
                  <Text style={[styles.name, { color: colors.text }]} numberOfLines={1}>{doctor.name}</Text>
                </View>
                <Text style={styles.specialty}>{doctor.specialty}</Text>
                <View style={styles.addressMiniRow}>
                  <MapPinIcon size={12} color="#059669" />
                  <Text style={styles.addressMiniTxt} numberOfLines={1}>
                    {doctor.location || "Main Medical Center, Downtown"}
                  </Text>
                </View>

                <View style={styles.metaRow}>
                  <View style={styles.metaItem}>
                    <Star size={10} color="#FFB300" fill="#FFB300" />
                    <Text style={styles.metaTxt}>{ratingLabel} {reviewLabel}</Text>
                  </View>
                  <View style={styles.dotSeparator} />
                  <View style={styles.metaItem}>
                    <ClockIcon size={10} color="#64748B" />
                    <Text style={styles.metaTxt}>{doctor.experience || "5+ yrs"}</Text>
                  </View>
                  {doctor.location && (
                    <>
                      <View style={styles.dotSeparator} />
                      <View style={styles.metaItem}>
                        <MapPinIcon size={10} color="#64748B" />
                        <Text style={styles.metaTxt} numberOfLines={1}>{doctor.location.split(',')[0]}</Text>
                      </View>
                    </>
                  )}
                </View>
              </View>

              <View style={styles.rightCol}>
                 <TouchableOpacity
                  style={[styles.miniHeart, isFollowed && styles.miniHeartActive]}
                  onPress={(e) => { e.stopPropagation(); toggleFollow().catch(() => undefined); }}
                >
                  <Heart size={14} stroke={isFollowed ? "#fff" : "#EF4444"} fill={isFollowed ? "#fff" : "transparent"} />
                </TouchableOpacity>
                <View style={styles.feeBadge}>
                   <Text style={styles.feeVal}>{doctor.consultationFee ?? 0} EGP</Text>
                </View>
              </View>
            </View>

            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[styles.actionBtnSecondary, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: colors.border }]}
                onPress={goToDetails}
              >
                <Text style={[styles.actionBtnTextSecondary, { color: colors.textMuted }]}>Profile</Text>
                <ChevronRight size={14} color={colors.textMuted} />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.actionBtnPrimary}
                onPress={(e) => { e.stopPropagation(); openConsultationModal(); }}
              >
                <LinearGradient 
                  colors={[COLORS.primary, COLORS.primaryDark]} 
                  start={{x:0, y:0}} 
                  end={{x:1, y:1}} 
                  style={styles.primaryGradient}
                >
                  <MessageCircleIcon size={14} color="#fff" />
                  <Text style={styles.actionBtnTextPrimary}>Consult Now</Text>
                </LinearGradient>
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

      {/* Modal remains same logic, just minor styling updates if needed */}
      <Modal visible={modalVisible} transparent animationType="slide">
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}>
            <View style={styles.sheetHeaderRow}>
              {doctor.photoUrl || doctor.imageUrl ? (
                <Image source={{ uri: doctor.photoUrl || doctor.imageUrl }} style={styles.sheetAvatar} />
              ) : (
                <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.sheetAvatar} />
              )}
              <View style={{ flex: 1 }}>
                <Text style={[styles.sheetTitle, { color: colors.text }]}>Medical Consultation</Text>
                <Text style={[styles.sheetSub, { color: colors.textMuted }]}>{doctor.name} · {doctor.specialty}</Text>
              </View>
              <TouchableOpacity onPress={() => setModalVisible(false)} style={[styles.closeBtn, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: colors.border }]}>
                <Ionicons name="close" size={20} color={colors.textMuted} />
              </TouchableOpacity>
            </View>
            <TextInput
              style={[styles.input, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", color: colors.text, borderColor: colors.border }]}
              placeholder="Describe your symptoms..."
              placeholderTextColor={isDark ? "#64748B" : "#BBB"}
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
              <LinearGradient colors={[COLORS.primary, COLORS.primaryDark]} style={styles.sendBtnGradient}>
                 <MessageCircleIcon size={16} stroke="#FFF" />
                 <Text style={styles.sendTxt}>Send Message</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  cardWrapper: { marginHorizontal: 16, marginVertical: 8, borderRadius: 24, shadowColor: "#000", shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.08, shadowRadius: 20, elevation: 5 },
  card: { borderRadius: 24, overflow: "hidden", backgroundColor: "#fff", borderWidth: 1, borderColor: "#F1F5F9" },
  cardGradient: { padding: 16 },
  cardHighlightBorder: { borderColor: "#FBBF24", borderWidth: 1.5 },
  
  verifiedStrip: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 12, backgroundColor: '#E6F4F1', alignSelf: 'flex-start', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  verifiedText: { fontSize: 8, fontWeight: '900', color: '#1E9E84', letterSpacing: 0.5 },

  topRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  avatarWrap: { position: 'relative' },
  avatarGlow: { position: 'absolute', width: 64, height: 64, borderRadius: 32, backgroundColor: '#E6F4F1', top: -4, left: -4, opacity: 0.5 },
  avatarBorder: { padding: 2, borderRadius: 28, backgroundColor: '#fff', borderWidth: 1, borderColor: '#F1F5F9' },
  avatar: { width: 52, height: 52, borderRadius: 26 },
  statusDot: { position: 'absolute', bottom: 2, right: 2, width: 12, height: 12, borderRadius: 6, backgroundColor: '#10B981', borderWidth: 2, borderColor: '#fff' },
  
  infoCol: { flex: 1, gap: 2 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  name: { fontSize: 16, fontWeight: '800', color: '#0F172A', letterSpacing: -0.3 },
  specialty: { fontSize: 12, color: '#64748B', fontWeight: '600', marginBottom: 2 },
  addressMiniRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 4 },
  addressMiniTxt: { fontSize: 11, color: '#059669', fontWeight: '700' },
  
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  metaItem: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  metaTxt: { fontSize: 11, color: '#64748B', fontWeight: '700' },
  dotSeparator: { width: 3, height: 3, borderRadius: 1.5, backgroundColor: '#CBD5E1' },
  
  rightCol: { alignItems: 'flex-end', justifyContent: 'space-between', height: 60 },
  miniHeart: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#FFF1F2', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#FFE4E6' },
  miniHeartActive: { backgroundColor: '#EF4444', borderColor: '#EF4444' },
  feeBadge: { backgroundColor: '#F0FDF4', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
  feeVal: { fontSize: 14, fontWeight: '900', color: '#059669' },

  actionRow: { flexDirection: 'row', gap: 10, marginTop: 18 },
  actionBtnSecondary: { flex: 0.4, backgroundColor: '#F8FAFC', paddingVertical: 12, borderRadius: 14, alignItems: 'center', flexDirection: 'row', justifyContent: 'center', gap: 4, borderWidth: 1, borderColor: '#F1F5F9' },
  actionBtnTextSecondary: { fontSize: 13, fontWeight: '800', color: '#64748B' },
  
  actionBtnPrimary: { flex: 1, borderRadius: 14, overflow: 'hidden', elevation: 4, shadowColor: '#1E9E84', shadowOpacity: 0.2, shadowRadius: 8, shadowOffset: { width:0, height:4 } },
  primaryGradient: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 12 },
  actionBtnTextPrimary: { color: '#fff', fontSize: 13, fontWeight: '800' },

  updatedBadge: { position: "absolute", top: 12, right: 12, backgroundColor: "#FBBF24", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 10, elevation: 4 },
  updatedBadgeText: { color: "#000", fontSize: 10, fontWeight: "900" },
  
  overlay: { flex: 1, backgroundColor: "rgba(15, 23, 42, 0.6)", justifyContent: "flex-end" },
  sheet: { backgroundColor: "#FFF", borderTopLeftRadius: 36, borderTopRightRadius: 36, padding: 25, paddingBottom: 40, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 20 },
  sheetHeaderRow: { flexDirection: "row", alignItems: "center", gap: 15, marginBottom: 25 },
  sheetAvatar: { width: 56, height: 56, borderRadius: 20, backgroundColor: '#F1F5F9' },
  sheetTitle: { fontSize: 19, fontWeight: "900", color: "#0F172A", letterSpacing: -0.5 },
  sheetSub: { fontSize: 13, color: '#64748B', marginTop: 2, fontWeight: '600' },
  closeBtn: { width: 36, height: 36, borderRadius: 12, backgroundColor: "#F8FAFC", alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: '#F1F5F9' },
  
  input: { height: 140, backgroundColor: "#F8FAFC", borderRadius: 20, padding: 20, marginBottom: 25, fontSize: 15, color: "#1E293B", textAlignVertical: "top", borderWidth: 1, borderColor: "#F1F5F9", lineHeight: 22 },
  sendBtn: { borderRadius: 20, overflow: 'hidden', elevation: 8, shadowColor: '#1E9E84', shadowOpacity: 0.3, shadowRadius: 12, shadowOffset: { width:0, height:6 } },
  sendBtnGradient: { height: 60, alignItems: "center", flexDirection: "row", justifyContent: "center", gap: 10 },
  sendTxt: { color: "#FFF", fontSize: 16, fontWeight: "800" },
});
