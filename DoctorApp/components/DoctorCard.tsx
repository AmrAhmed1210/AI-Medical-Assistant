import React, { useState } from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet, Modal, TextInput } from "react-native";
import { useRouter } from "expo-router";
import { Star, MapPin, Clock, MessageCircle } from "lucide-react-native";

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
}

export default function DoctorCard({ doctor }: { doctor: Doctor }) {
  const router = useRouter();
  const [modalVisible, setModalVisible] = useState(false);
  const [message, setMessage] = useState("");

  const goToDetails = () =>
    router.push({ pathname: "/(patient)/doctor-details", params: { doctorId: String(doctor.id) } });

  const sendMessage = () => {
    if (!message.trim()) return;
    setModalVisible(false);
    setMessage("");
    router.push({ pathname: "/(patient)/messages", params: { doctorName: doctor.name } });
  };

  return (
    <>
      <TouchableOpacity style={styles.card} onPress={goToDetails} activeOpacity={0.88}>

        {/* Top Row */}
        <View style={styles.topRow}>

          {/* Avatar — أخضر زي البروفايل */}
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

          {/* Info */}
          <View style={styles.infoCol}>
            <Text style={styles.name} numberOfLines={1}>{doctor.name}</Text>
            <View style={styles.specBadge}>
              <Text style={styles.specTxt}>{doctor.specialty}</Text>
            </View>
            <View style={styles.metaRow}>
              {doctor.rating != null && (
                <View style={styles.metaItem}>
                  <Star size={11} color="#FFB300" fill="#FFB300" />
                  <Text style={styles.metaTxt}>{Number(doctor.rating).toFixed(1)}</Text>
                  {doctor.reviewCount != null && (
                    <Text style={styles.metaGray}>({doctor.reviewCount})</Text>
                  )}
                </View>
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

          {/* Fee */}
          <View style={styles.feeCol}>
            {doctor.consultationFee != null && (
              <>
                <Text style={styles.feeAmt}>${doctor.consultationFee}</Text>
                <Text style={styles.feeLbl}>/ visit</Text>
              </>
            )}
          </View>
        </View>

        <View style={styles.divider} />

        {/* Bottom Buttons */}
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
            <Text style={styles.bookTxt}>Book Now</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>

      {/* Consultation Modal */}
      <Modal visible={modalVisible} transparent animationType="slide">
        <View style={styles.overlay}>
          <View style={styles.sheet}>

            {/* Modal Header */}
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
              placeholder="Describe your symptoms or question..."
              placeholderTextColor="#BBB"
              value={message}
              onChangeText={setMessage}
              multiline
            />

            <TouchableOpacity style={styles.sendBtn} onPress={sendMessage}>
              <MessageCircle size={16} color={COLORS.white} />
              <Text style={styles.sendTxt}>Send Message</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.cancelBtn} onPress={() => setModalVisible(false)}>
              <Text style={styles.cancelTxt}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 18,
    marginHorizontal: 16,
    marginVertical: 7,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 4,
    overflow: "hidden",
  },

  topRow:      { flexDirection: "row", padding: 14, gap: 12, alignItems: "flex-start" },

  // Avatar أخضر زي البروفايل
  avatarWrap:  { position: "relative" },
  avatar:      { width: 62, height: 62, borderRadius: 31, backgroundColor: COLORS.lightGray },
  avatarFallback: {
    width: 62, height: 62, borderRadius: 31,
    backgroundColor: COLORS.primary,
    alignItems: "center", justifyContent: "center",
    borderWidth: 3, borderColor: COLORS.primary + "50",
  },
  avatarTxt:   { fontSize: 24, fontWeight: "800", color: COLORS.white },
  dot: {
    position: "absolute", bottom: 1, right: 1,
    width: 14, height: 14, borderRadius: 7,
    borderWidth: 2, borderColor: COLORS.white,
  },

  infoCol:   { flex: 1, gap: 5 },
  name:      { fontSize: 15, fontWeight: "700", color: COLORS.black },
  specBadge: {
    alignSelf: "flex-start", backgroundColor: "#E8F7F4",
    paddingHorizontal: 10, paddingVertical: 3, borderRadius: 20,
  },
  specTxt:   { fontSize: 11, color: COLORS.primary, fontWeight: "600" },
  metaRow:   { flexDirection: "row", gap: 10, flexWrap: "wrap" },
  metaItem:  { flexDirection: "row", alignItems: "center", gap: 3 },
  metaTxt:   { fontSize: 12, fontWeight: "600", color: COLORS.black },
  metaGray:  { fontSize: 11, color: COLORS.gray },

  feeCol:    { alignItems: "flex-end", justifyContent: "flex-start", paddingTop: 2 },
  feeAmt:    { fontSize: 17, fontWeight: "800", color: COLORS.primary },
  feeLbl:    { fontSize: 10, color: COLORS.gray },

  divider:   { height: 1, backgroundColor: "#F0F0F0", marginHorizontal: 14 },

  bottomRow: { flexDirection: "row", gap: 10, padding: 12 },
  consultBtn: {
    flex: 1,
    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6,
    backgroundColor: "#EEF9F6",
    borderWidth: 1.5, borderColor: COLORS.primary,
    paddingVertical: 10, borderRadius: 22,
  },
  consultTxt: { color: COLORS.primary, fontSize: 12, fontWeight: "600" },
  bookBtn: {
    flex: 1,
    backgroundColor: COLORS.primary,
    paddingVertical: 10,
    borderRadius: 22, alignItems: "center",
  },
  bookTxt: { color: COLORS.white, fontSize: 13, fontWeight: "700" },

  // Modal
  overlay: { flex: 1, backgroundColor: "#00000070", justifyContent: "flex-end" },
  sheet: {
    backgroundColor: COLORS.white,
    borderTopLeftRadius: 28, borderTopRightRadius: 28,
    padding: 22, paddingBottom: 38,
  },
  sheetHeaderRow: {
    flexDirection: "row", alignItems: "center", gap: 12, marginBottom: 18,
  },
  sheetAvatar: {
    width: 46, height: 46, borderRadius: 23,
    backgroundColor: COLORS.primary,
    alignItems: "center", justifyContent: "center",
  },
  sheetAvatarTxt: { fontSize: 18, fontWeight: "800", color: COLORS.white },
  sheetTitle:  { fontSize: 15, fontWeight: "700", color: COLORS.black },
  sheetSub:    { fontSize: 12, color: COLORS.gray, marginTop: 1 },
  closeBtn:    { width: 32, height: 32, borderRadius: 16, backgroundColor: "#F5F5F5", alignItems: "center", justifyContent: "center" },
  closeTxt:    { fontSize: 12, color: COLORS.gray, fontWeight: "600" },

  input: {
    height: 110, borderWidth: 1.5, borderColor: "#E8E8E8",
    borderRadius: 14, padding: 12, marginBottom: 14,
    fontSize: 14, color: COLORS.black, textAlignVertical: "top",
    backgroundColor: "#FAFAFA",
  },
  sendBtn: {
    backgroundColor: COLORS.primary, paddingVertical: 14, borderRadius: 14,
    alignItems: "center", marginBottom: 10,
    flexDirection: "row", justifyContent: "center", gap: 8,
  },
  sendTxt:   { color: COLORS.white, fontSize: 15, fontWeight: "700" },
  cancelBtn: { paddingVertical: 10, alignItems: "center" },
  cancelTxt: { color: COLORS.gray, fontSize: 14 },
});