import { useEffect, useRef } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity, Animated, Dimensions,
} from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useLanguage } from "../../context/LanguageContext";

const { width, height } = Dimensions.get("window");

const T = {
  primary: "#1E9E84",
  deep: "#115E59",
  white: "#FFFFFF",
  bg: "#F0FDFA",
  text: "#0F172A",
  muted: "#94A3B8",
};

function Bubble({ size, left, delay, dur }: { size: number; left: number; delay: number; dur: number }) {
  const ty = useRef(new Animated.Value(0)).current;
  const op = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(Animated.sequence([
      Animated.parallel([
        Animated.timing(ty, { toValue: -(height * 0.5), duration: dur, delay, useNativeDriver: true }),
        Animated.sequence([
          Animated.timing(op, { toValue: 0.55, duration: dur * 0.15, delay, useNativeDriver: true }),
          Animated.timing(op, { toValue: 0, duration: dur * 0.85, useNativeDriver: true }),
        ]),
      ]),
      Animated.timing(ty, { toValue: 0, duration: 0, useNativeDriver: true }),
    ])).start();
  }, []);
  return (
    <Animated.View pointerEvents="none" style={{
      position: "absolute", bottom: -20, left: `${left}%` as any,
      width: size, height: size, borderRadius: size / 2,
      backgroundColor: "rgba(255,255,255,0.12)", transform: [{ translateY: ty }], opacity: op,
    }} />
  );
}

export default function WelcomeScreen() {
  const router = useRouter();
  const { tr, isRTL } = useLanguage();

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const btnFade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.timing(fadeAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
        Animated.timing(slideAnim, { toValue: 0, duration: 700, useNativeDriver: true }),
      ]),
      Animated.timing(btnFade, { toValue: 1, duration: 500, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <View style={s.container}>
      {/* Full Gradient with Bubbles */}
      <LinearGradient colors={[T.primary, T.deep]} style={s.gradient}>
        <Bubble size={70} left={5} delay={0} dur={4500} />
        <Bubble size={35} left={20} delay={800} dur={3500} />
        <Bubble size={50} left={40} delay={400} dur={5200} />
        <Bubble size={28} left={60} delay={1200} dur={3000} />
        <Bubble size={55} left={78} delay={600} dur={4800} />
        <Bubble size={32} left={92} delay={1000} dur={3800} />

        <Animated.View style={[s.heroContent, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
          <View style={s.logoBox}>
            <Ionicons name="heart" size={42} color={T.primary} />
          </View>

          <Text style={s.brand}>{tr("app_name")}</Text>
          <Text style={s.tagline}>{tr("tagline")}</Text>

          <View style={s.featureRow}>
            <View style={s.featureBadge}>
              <Ionicons name="sparkles" size={14} color="#fff" />
              <Text style={s.featureText}>{tr("ai_diagnostics")}</Text>
            </View>
            <View style={s.featureBadge}>
              <Ionicons name="shield-checkmark" size={14} color="#fff" />
              <Text style={s.featureText}>{tr("secure_files")}</Text>
            </View>
          </View>
        </Animated.View>
      </LinearGradient>

      {/* Bottom Actions */}
      <Animated.View style={[s.bottomSection, { opacity: btnFade }]}>
        <TouchableOpacity style={s.primaryBtn} activeOpacity={0.85} onPress={() => router.push("/(auth)/login")}>
          <LinearGradient colors={[T.primary, T.deep]} style={s.btnGrad}>
            <Text style={s.primaryBtnTxt}>{tr("sign_in")}</Text>
            <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={18} color="#fff" style={{ [isRTL ? "marginRight" : "marginLeft"]: 6 }} />
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity style={s.outlineBtn} activeOpacity={0.85} onPress={() => router.push("/(auth)/register")}>
          <Text style={s.outlineBtnTxt}>{tr("create_account")}</Text>
        </TouchableOpacity>

        <Text style={s.disclaimer}>{tr("tos_disclaimer")}</Text>
      </Animated.View>
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: T.bg },
  gradient: {
    height: "62%",
    borderBottomLeftRadius: 48,
    borderBottomRightRadius: 48,
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
  },
  heroContent: { alignItems: "center", paddingHorizontal: 30 },
  logoBox: {
    width: 88, height: 88, borderRadius: 26,
    backgroundColor: "#fff", justifyContent: "center", alignItems: "center",
    marginBottom: 20,
    elevation: 10, shadowColor: "#000", shadowOpacity: 0.15, shadowRadius: 12, shadowOffset: { width: 0, height: 8 },
  },
  brand: { fontSize: 36, fontWeight: "900", color: "#fff", letterSpacing: -1 },
  tagline: { fontSize: 14, color: "rgba(255,255,255,0.85)", fontWeight: "600", marginTop: 6, textAlign: "center" },
  featureRow: { flexDirection: "row", gap: 12, marginTop: 22 },
  featureBadge: {
    flexDirection: "row", alignItems: "center", gap: 5,
    backgroundColor: "rgba(255,255,255,0.18)", paddingHorizontal: 14, paddingVertical: 7, borderRadius: 20,
  },
  featureText: { color: "#fff", fontSize: 12, fontWeight: "700" },

  bottomSection: { flex: 1, padding: 30, justifyContent: "center" },
  primaryBtn: {
    height: 56, borderRadius: 18, overflow: "hidden", marginBottom: 14,
    elevation: 6, shadowColor: T.primary, shadowOpacity: 0.25, shadowRadius: 10, shadowOffset: { width: 0, height: 5 },
  },
  btnGrad: { width: "100%", height: "100%", flexDirection: "row", justifyContent: "center", alignItems: "center" },
  primaryBtnTxt: { color: "#fff", fontSize: 16, fontWeight: "800" },
  outlineBtn: {
    height: 56, borderRadius: 18, borderWidth: 2, borderColor: T.primary,
    justifyContent: "center", alignItems: "center",
  },
  outlineBtnTxt: { color: T.primary, fontSize: 16, fontWeight: "700" },
  disclaimer: { textAlign: "center", color: T.muted, fontSize: 11, marginTop: 22, fontWeight: "500" },
});