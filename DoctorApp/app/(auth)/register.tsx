import React, { useState, useRef, useEffect } from "react";
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, TouchableOpacity, Keyboard, ActivityIndicator,
  Animated, Easing, StatusBar,
} from "react-native";
import { useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import DateTimePicker from "@react-native-community/datetimepicker";
import { registerApi, saveSession } from "../../services/authService";
import { useLanguage } from "../../context/LanguageContext";

// ─── Design tokens ────────────────────────────────────────────────────────────
const C = {
  primary: "#059669",
  dark: "#047857",
  bg: "#ECFDF5",          // soft mint background — not blinding white
  card: "#FFFFFF",
  text: "#0F172A",
  sub: "#475569",
  muted: "#94A3B8",
  border: "#D1FAE5",          // mint-tinted border
  inputBg: "#F0FDF4",          // very light mint for inputs
  focusBg: "#FFFFFF",
  red: "#EF4444",
};

const EGYPT_CITIES_EN = [
  "Cairo", "Giza", "Alexandria", "Qalyubia", "Sharqia", "Dakahlia",
  "Gharbia", "Monufia", "Kafr El-Sheikh", "Beheira", "Ismailia", "Port Said",
  "Suez", "Damietta", "North Sinai", "South Sinai", "Fayoum", "Beni Suef",
  "Minya", "Assiut", "Sohag", "Qena", "Luxor", "Aswan", "Red Sea",
  "Matruh", "New Valley",
];
const EGYPT_CITIES_AR = [
  "القاهرة", "الجيزة", "الإسكندرية", "القليوبية", "الشرقية", "الدقهلية",
  "الغربية", "المنوفية", "كفر الشيخ", "البحيرة", "الإسماعيلية", "بور سعيد",
  "السويس", "دمياط", "شمال سيناء", "جنوب سيناء", "الفيوم", "بني سويف",
  "المنيا", "أسيوط", "سوهاج", "قنا", "الأقصر", "أسوان", "البحر الأحمر",
  "مطروح", "الوادي الجديد",
];

const STEP_INFO = [
  { icon: "person", titleKey: "join_medbook", subKey: "step_1_subtitle" },
  { icon: "mail", titleKey: "join_medbook", subKey: "step_2_subtitle" },
  { icon: "lock-closed", titleKey: "join_medbook", subKey: "step_3_subtitle" },
];

export default function RegisterScreen() {
  const router = useRouter();
  const scrollRef = useRef<ScrollView>(null);
  const { tr, isRTL } = useLanguage();

  // ── entry animation ────────────────────────────────────────────────────────
  const fadeIn = useRef(new Animated.Value(0)).current;
  const slideIn = useRef(new Animated.Value(36)).current;
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeIn, { toValue: 1, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(slideIn, { toValue: 0, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  }, []);

  // ── step cross-fade ────────────────────────────────────────────────────────
  const stepFade = useRef(new Animated.Value(1)).current;
  const stepSlide = useRef(new Animated.Value(0)).current;
  const changeStep = (next: number) => {
    Animated.parallel([
      Animated.timing(stepFade, { toValue: 0, duration: 130, useNativeDriver: true }),
      Animated.timing(stepSlide, { toValue: -16, duration: 130, useNativeDriver: true }),
    ]).start(() => {
      setStep(next);
      stepSlide.setValue(16);
      Animated.parallel([
        Animated.timing(stepFade, { toValue: 1, duration: 250, useNativeDriver: true }),
        Animated.timing(stepSlide, { toValue: 0, duration: 250, useNativeDriver: true }),
      ]).start();
    });
  };

  // ── form state ─────────────────────────────────────────────────────────────
  const [step, setStep] = useState(1);
  const [focused, setFocused] = useState<string | null>(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [dobDate, setDobDate] = useState<Date | null>(null);
  const [showPicker, setShowPicker] = useState(false);
  const [gender, setGender] = useState("Male");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [country, setCountry] = useState("Egypt");
  const [city, setCity] = useState("");
  const [district, setDistrict] = useState("");
  const [bloodType, setBloodType] = useState("O+");
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [citySearch, setCitySearch] = useState("");

  const CITIES = isRTL ? EGYPT_CITIES_AR : EGYPT_CITIES_EN;
  const filteredIndexes = EGYPT_CITIES_EN
    .map((en, i) => ({ en, ar: EGYPT_CITIES_AR[i], i }))
    .filter(({ en, ar }) =>
      citySearch === "" ||
      en.toLowerCase().includes(citySearch.toLowerCase()) ||
      ar.includes(citySearch)
    );

  // ── logic — untouched ──────────────────────────────────────────────────────
  const handleNextStep1 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!firstName.trim() || !lastName.trim() || !dobDate) {
      Toast.show({ type: "error", text1: tr("error"), text2: tr("please_fill_fields" as any), position: "top" });
      return;
    }
    changeStep(2);
  };

  const handleNextStep2 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!email.trim() || !phone.trim() || !city.trim() || !district.trim()) {
      Toast.show({ type: "error", text1: tr("error"), text2: tr("please_fill_fields" as any), position: "top" });
      return;
    }
    changeStep(3);
  };

  const handleRegister = async () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    if (password.length < 6) {
      Toast.show({ type: "error", text1: tr("min_6_chars"), position: "top" });
      return;
    }
    if (!weight.trim() || !height.trim()) {
      Toast.show({ type: "error", text1: tr("error"), text2: tr("please_fill_fields" as any), position: "top" });
      return;
    }
    setLoading(true);
    try {
      const address = `${country} - ${city.trim()} - ${district.trim()}`;
      const tzOffset = dobDate!.getTimezoneOffset() * 60000;
      const localISO = new Date(dobDate!.getTime() - tzOffset).toISOString().split("T")[0];
      const auth = await registerApi({
        fullName: `${firstName.trim()} ${lastName.trim()}`,
        email: email.toLowerCase().trim(),
        password,
        role: "Patient",
        phoneNumber: phone.trim(),
        address,
        dateOfBirth: localISO,
        gender: gender === "Male" ? "Male" : "Female",
        bloodType,
        weight: Number(weight) || 0,
        height: Number(height) || 0,
        smokingStatus: "Non-Smoker",
      });
      await saveSession(auth);
      Toast.show({ type: "success", text1: `${tr("success")} 👋`, position: "top" });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => router.replace("/(auth)/onboarding"), 1600);
    } catch (err: any) {
      Toast.show({ type: "error", text1: err.message || tr("error"), position: "top" });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setLoading(false);
    }
  };

  // ── helpers ────────────────────────────────────────────────────────────────
  const fw = (id: string) => [
    s.inputWrap,
    focused === id && s.inputWrapFocused,
    { flexDirection: (isRTL ? "row-reverse" : "row") as any },
  ];
  const ic = (id: string) => focused === id ? C.primary : C.muted;
  const tx = (id: string) => [s.input, { textAlign: (isRTL ? "right" : "left") as any }];

  // ── progress bar ───────────────────────────────────────────────────────────
  const Progress = () => (
    <View style={[s.progressRow, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
      {[1, 2, 3].map((n) => (
        <React.Fragment key={n}>
          <View style={[s.dot, step >= n && s.dotActive]}>
            {step > n
              ? <Ionicons name="checkmark" size={14} color="#fff" />
              : <Text style={[s.dotTxt, step >= n && s.dotTxtActive]}>{n}</Text>
            }
          </View>
          {n < 3 && <View style={[s.line, step > n && s.lineActive]} />}
        </React.Fragment>
      ))}
    </View>
  );

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* ── Gradient header ── */}
      <LinearGradient colors={[C.primary, C.dark]} style={s.header}>
        {/* top row */}
        <View style={[s.headerTop, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
          <TouchableOpacity
            style={s.backBtn}
            onPress={() => step > 1 ? changeStep(step - 1) : router.back()}
          >
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color="#fff" />
          </TouchableOpacity>
          <Text style={s.headerStep}>{step === 1 ? tr("step_1") : step === 2 ? tr("step_2") : tr("step_3")}</Text>
        </View>

        {/* title */}
        <Text style={[s.headerTitle, { textAlign: isRTL ? "right" : "left" }]}>{tr("join_medbook")}</Text>
        <Text style={[s.headerSub, { textAlign: isRTL ? "right" : "left" }]}>
          {step === 1 ? tr("step_1_subtitle") : step === 2 ? tr("step_2_subtitle") : tr("step_3_subtitle")}
        </Text>

        <Progress />
      </LinearGradient>

      {/* ── Scrollable form ── */}
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView
          ref={scrollRef}
          style={s.scroll}
          contentContainerStyle={s.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <Animated.View style={[s.card, { opacity: stepFade, transform: [{ translateY: stepSlide }] }]}>

            {/* ══ STEP 1 ══════════════════════════════════════════════════ */}
            {step === 1 && (
              <View>
                <View style={[s.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                  <View style={s.half}>
                    <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("first_name")}</Text>
                    <View style={fw("first")}>
                      <Ionicons name="person-outline" size={18} color={ic("first")} style={isRTL ? s.iconL : s.iconR} />
                      <TextInput
                        placeholder={tr("first_name")} placeholderTextColor={C.muted}
                        style={tx("first")} value={firstName} onChangeText={setFirstName}
                        onFocus={() => setFocused("first")} onBlur={() => setFocused(null)}
                      />
                    </View>
                  </View>
                  <View style={s.half}>
                    <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("last_name")}</Text>
                    <View style={fw("last")}>
                      <TextInput
                        placeholder={tr("last_name")} placeholderTextColor={C.muted}
                        style={tx("last")} value={lastName} onChangeText={setLastName}
                        onFocus={() => setFocused("last")} onBlur={() => setFocused(null)}
                      />
                    </View>
                  </View>
                </View>

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("date_of_birth")}</Text>
                <TouchableOpacity onPress={() => setShowPicker(true)} style={[s.inputWrap, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                  <Ionicons name="calendar-outline" size={18} color={dobDate ? C.primary : C.muted} style={isRTL ? s.iconL : s.iconR} />
                  <Text style={[s.input, { color: dobDate ? C.text : C.muted, textAlign: isRTL ? "right" : "left", paddingTop: 2 }]}>
                    {dobDate
                      ? dobDate.toLocaleDateString(isRTL ? "ar-EG" : "en-US", { year: "numeric", month: "long", day: "numeric" })
                      : tr("dob_hint")}
                  </Text>
                </TouchableOpacity>

                {showPicker && (
                  <DateTimePicker
                    value={dobDate || new Date(2000, 0, 1)}
                    mode="date" display="spinner" maximumDate={new Date()}
                    onChange={(_, d) => {
                      if (Platform.OS === "android") setShowPicker(false);
                      if (d) setDobDate(d);
                    }}
                  />
                )}
                {Platform.OS === "ios" && showPicker && (
                  <TouchableOpacity style={s.iosDone} onPress={() => setShowPicker(false)}>
                    <Text style={{ color: "#fff", fontWeight: "700" }}>{tr("confirm")}</Text>
                  </TouchableOpacity>
                )}

                <Text style={[s.label, { marginTop: 8, textAlign: isRTL ? "right" : "left" }]}>{tr("gender")}</Text>
                <View style={[s.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                  {["Male", "Female"].map((g) => (
                    <TouchableOpacity
                      key={g}
                      style={[s.genderBtn, gender === g && s.genderBtnActive]}
                      onPress={() => setGender(g)} activeOpacity={0.8}
                    >
                      <Ionicons
                        name={g === "Male" ? "male" : "female"}
                        size={16}
                        color={gender === g ? C.primary : C.muted}
                        style={{ marginRight: 6 }}
                      />
                      <Text style={[s.genderTxt, gender === g && s.genderTxtActive]}>
                        {g === "Male" ? tr("male") : tr("female")}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>

                <TouchableOpacity style={s.cta} onPress={handleNextStep1} activeOpacity={0.85}>
                  <LinearGradient colors={[C.primary, C.dark]} style={s.ctaGrad}>
                    <Text style={s.ctaTxt}>{tr("continue_btn")}</Text>
                    <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={18} color="#fff" style={{ [isRTL ? "marginRight" : "marginLeft"]: 8 }} />
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            )}

            {/* ══ STEP 2 ══════════════════════════════════════════════════ */}
            {step === 2 && (
              <View>
                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("email")}</Text>
                <View style={fw("email")}>
                  <Ionicons name="mail-outline" size={18} color={ic("email")} style={isRTL ? s.iconL : s.iconR} />
                  <TextInput
                    placeholder={tr("email_placeholder" as any) || "example@email.com"} placeholderTextColor={C.muted}
                    keyboardType="email-address" autoCapitalize="none"
                    style={tx("email")} value={email} onChangeText={setEmail}
                    onFocus={() => setFocused("email")} onBlur={() => setFocused(null)}
                  />
                </View>

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("phone")}</Text>
                <View style={fw("phone")}>
                  <Ionicons name="call-outline" size={18} color={ic("phone")} style={isRTL ? s.iconL : s.iconR} />
                  <TextInput
                    placeholder={tr("phone_placeholder" as any) || "+20 1xx xxxx xxxx"} placeholderTextColor={C.muted}
                    keyboardType="phone-pad" maxLength={15}
                    style={tx("phone")} value={phone} onChangeText={setPhone}
                    onFocus={() => setFocused("phone")} onBlur={() => setFocused(null)}
                  />
                </View>

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("country")}</Text>
                <View style={[s.chipRow, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                  {["Egypt", "Other"].map((item) => (
                    <TouchableOpacity
                      key={item}
                      style={[s.chip, country === item && s.chipActive]}
                      onPress={() => { setCountry(item); setCity(""); setDistrict(""); }}
                      activeOpacity={0.8}
                    >
                      <Text style={[s.chipTxt, country === item && s.chipTxtActive]}>
                        {item === "Egypt" ? tr("egypt") : tr("other")}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("city")}</Text>
                {country === "Egypt" ? (
                  <View>
                    {/* search box */}
                    <View style={[fw("citySearch"), { marginBottom: 10 }]}>
                      <Ionicons name="search-outline" size={18} color={ic("citySearch")} style={isRTL ? s.iconL : s.iconR} />
                      <TextInput
                        placeholder={tr("search_governorate" as any)}
                        placeholderTextColor={C.muted}
                        style={tx("citySearch")}
                        value={citySearch}
                        onChangeText={setCitySearch}
                        onFocus={() => setFocused("citySearch")}
                        onBlur={() => setFocused(null)}
                      />
                      {citySearch.length > 0 && (
                        <TouchableOpacity onPress={() => setCitySearch("")} style={{ padding: 4 }}>
                          <Ionicons name="close-circle" size={18} color={C.muted} />
                        </TouchableOpacity>
                      )}
                    </View>
                    {/* chips grid */}
                    <View style={[s.cityGrid, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                      {filteredIndexes.map(({ en, ar, i }) => (
                        <TouchableOpacity
                          key={i}
                          style={[s.cityChip, city === en && s.cityChipActive]}
                          onPress={() => setCity(en)}
                          activeOpacity={0.8}
                        >
                          <Text style={[s.cityTxt, city === en && s.cityTxtActive]}>
                            {isRTL ? ar : en}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                    {city !== "" && (
                      <View style={s.selectedCity}>
                        <Ionicons name="checkmark-circle" size={16} color={C.primary} />
                        <Text style={s.selectedCityTxt}>{isRTL ? EGYPT_CITIES_AR[EGYPT_CITIES_EN.indexOf(city)] : city}</Text>
                      </View>
                    )}
                  </View>
                ) : (
                  <View style={fw("city")}>
                    <Ionicons name="location-outline" size={18} color={ic("city")} style={isRTL ? s.iconL : s.iconR} />
                    <TextInput
                      placeholder={tr("city")} placeholderTextColor={C.muted}
                      style={tx("city")} value={city} onChangeText={setCity}
                      onFocus={() => setFocused("city")} onBlur={() => setFocused(null)}
                    />
                  </View>
                )}

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("district_area")}</Text>
                <View style={fw("district")}>
                  <Ionicons name="navigate-outline" size={18} color={ic("district")} style={isRTL ? s.iconL : s.iconR} />
                  <TextInput
                    placeholder={tr("district_area")} placeholderTextColor={C.muted}
                    style={tx("district")} value={district} onChangeText={setDistrict}
                    onFocus={() => setFocused("district")} onBlur={() => setFocused(null)}
                  />
                </View>
                <Text style={s.hint}>{tr("district_hint")}</Text>

                <TouchableOpacity style={s.cta} onPress={handleNextStep2} activeOpacity={0.85}>
                  <LinearGradient colors={[C.primary, C.dark]} style={s.ctaGrad}>
                    <Text style={s.ctaTxt}>{tr("continue_btn")}</Text>
                    <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={18} color="#fff" style={{ [isRTL ? "marginRight" : "marginLeft"]: 8 }} />
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            )}

            {/* ══ STEP 3 ══════════════════════════════════════════════════ */}
            {step === 3 && (
              <View>
                <View style={[s.row, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
                  <View style={s.half}>
                    <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("weight_kg")}</Text>
                    <View style={fw("weight")}>
                      <Ionicons name="scale-outline" size={18} color={ic("weight")} style={isRTL ? s.iconL : s.iconR} />
                      <TextInput
                        placeholder="70" placeholderTextColor={C.muted}
                        keyboardType="numeric" maxLength={3}
                        style={tx("weight")} value={weight} onChangeText={setWeight}
                        onFocus={() => setFocused("weight")} onBlur={() => setFocused(null)}
                      />
                    </View>
                  </View>
                  <View style={s.half}>
                    <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("height_cm")}</Text>
                    <View style={fw("height")}>
                      <Ionicons name="body-outline" size={18} color={ic("height")} style={isRTL ? s.iconL : s.iconR} />
                      <TextInput
                        placeholder="170" placeholderTextColor={C.muted}
                        keyboardType="numeric" maxLength={3}
                        style={tx("height")} value={height} onChangeText={setHeight}
                        onFocus={() => setFocused("height")} onBlur={() => setFocused(null)}
                      />
                    </View>
                  </View>
                </View>

                <Text style={[s.label, { textAlign: isRTL ? "right" : "left" }]}>{tr("blood_type")}</Text>
                <View style={s.bloodGrid}>
                  {["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"].map((bt) => (
                    <TouchableOpacity
                      key={bt}
                      style={[s.bloodBtn, bloodType === bt && s.bloodBtnActive]}
                      onPress={() => setBloodType(bt)} activeOpacity={0.8}
                    >
                      <Text style={[s.bloodTxt, bloodType === bt && s.bloodTxtActive]}>{bt}</Text>
                    </TouchableOpacity>
                  ))}
                </View>

                <Text style={[s.label, { marginTop: 16, textAlign: isRTL ? "right" : "left" }]}>{tr("create_password")}</Text>
                <View style={fw("pass")}>
                  <Ionicons name="lock-closed-outline" size={18} color={ic("pass")} style={isRTL ? s.iconL : s.iconR} />
                  <TextInput
                    placeholder={tr("min_6_chars")} placeholderTextColor={C.muted}
                    secureTextEntry={!showPassword}
                    style={tx("pass")} value={password} onChangeText={setPassword}
                    onFocus={() => setFocused("pass")} onBlur={() => setFocused(null)}
                  />
                  <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={s.eye}>
                    <Ionicons name={showPassword ? "eye-outline" : "eye-off-outline"} size={18} color={C.muted} />
                  </TouchableOpacity>
                </View>

                <TouchableOpacity
                  style={[s.cta, loading && { opacity: 0.7 }]}
                  onPress={handleRegister} disabled={loading} activeOpacity={0.85}
                >
                  <LinearGradient colors={[C.primary, C.dark]} style={s.ctaGrad}>
                    {loading
                      ? <ActivityIndicator color="#fff" />
                      : <Text style={s.ctaTxt}>{tr("create_account")}</Text>
                    }
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            )}

            {/* footer */}
            <View style={[s.footer, { flexDirection: isRTL ? "row-reverse" : "row" }]}>
              <Text style={s.footerTxt}>{tr("already_have_account")} </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/login")} activeOpacity={0.7}>
                <Text style={s.link}>{tr("log_in")}</Text>
              </TouchableOpacity>
            </View>

          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
      <Toast />
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  // ── gradient header ──────────────────────────────────────────────────────
  header: {
    paddingTop: Platform.OS === "ios" ? 60 : 44,
    paddingHorizontal: 24,
    paddingBottom: 28,
    borderBottomLeftRadius: 32,
    borderBottomRightRadius: 32,
  },
  headerTop: { alignItems: "center", justifyContent: "space-between", marginBottom: 20 },
  backBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.2)",
    justifyContent: "center", alignItems: "center",
  },
  headerStep: { fontSize: 13, color: "rgba(255,255,255,0.8)", fontWeight: "700" },
  headerTitle: { fontSize: 30, fontWeight: "900", color: "#fff", letterSpacing: -0.5, marginBottom: 4 },
  headerSub: { fontSize: 14, color: "rgba(255,255,255,0.75)", marginBottom: 24 },

  // progress
  progressRow: { alignItems: "center" },
  dot: {
    width: 32, height: 32, borderRadius: 16,
    backgroundColor: "rgba(255,255,255,0.25)",
    justifyContent: "center", alignItems: "center",
    borderWidth: 2, borderColor: "rgba(255,255,255,0.4)",
  },
  dotActive: { backgroundColor: "#fff", borderColor: "#fff" },
  dotTxt: { fontSize: 13, fontWeight: "700", color: "rgba(255,255,255,0.7)" },
  dotTxtActive: { color: C.primary },
  line: { flex: 1, height: 2, backgroundColor: "rgba(255,255,255,0.25)", marginHorizontal: 6 },
  lineActive: { backgroundColor: "#fff" },

  // scroll / card
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 40 },
  card: {
    backgroundColor: C.card,
    borderRadius: 24, padding: 24,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07, shadowRadius: 20,
    elevation: 4,
    borderWidth: 1, borderColor: "rgba(5,150,105,0.08)",
  },

  // inputs
  label: {
    fontSize: 12, fontWeight: "700", color: C.text,
    textTransform: "uppercase", letterSpacing: 0.6,
    marginBottom: 8, marginLeft: 2,
  },
  inputWrap: {
    alignItems: "center",
    backgroundColor: C.inputBg,
    borderWidth: 1.5, borderColor: C.border,
    borderRadius: 14, height: 54,
    paddingHorizontal: 14, marginBottom: 16,
  },
  inputWrapFocused: { borderColor: C.primary, backgroundColor: C.focusBg },
  iconR: { marginRight: 10 },
  iconL: { marginLeft: 10 },
  input: { flex: 1, fontSize: 15, color: C.text, height: "100%" },
  eye: { padding: 6 },

  // layout helpers
  row: { gap: 10, marginBottom: 0 },
  half: { flex: 1 },

  // gender
  genderBtn: {
    flex: 1, height: 50, borderRadius: 14,
    borderWidth: 1.5, borderColor: C.border,
    flexDirection: "row", justifyContent: "center", alignItems: "center",
    backgroundColor: C.inputBg, marginBottom: 16,
  },
  genderBtnActive: { backgroundColor: "#ECFDF5", borderColor: C.primary },
  genderTxt: { fontSize: 14, fontWeight: "600", color: C.muted },
  genderTxtActive: { color: C.primary },

  // chips (country / severity)
  chipRow: { gap: 8, marginBottom: 16 },
  chip: {
    flex: 1, height: 44, borderRadius: 12,
    borderWidth: 1.5, borderColor: C.border,
    justifyContent: "center", alignItems: "center",
    backgroundColor: C.inputBg,
  },
  chipActive: { backgroundColor: "#ECFDF5", borderColor: C.primary },
  chipTxt: { fontSize: 13, fontWeight: "600", color: C.muted },
  chipTxtActive: { color: C.primary, fontWeight: "700" },

  // city grid
  cityGrid: { flexWrap: "wrap", gap: 8, marginBottom: 16 },
  cityChip: {
    paddingHorizontal: 12, height: 36, borderRadius: 10,
    borderWidth: 1.5, borderColor: C.border,
    justifyContent: "center", alignItems: "center",
    backgroundColor: C.inputBg,
  },
  cityChipActive: { backgroundColor: "#ECFDF5", borderColor: C.primary },
  cityTxt: { fontSize: 12, fontWeight: "600", color: C.muted },
  cityTxtActive: { color: C.primary, fontWeight: "700" },

  selectedCity: {
    flexDirection: "row", alignItems: "center", gap: 6,
    marginTop: 4, marginBottom: 8, marginLeft: 2,
  },
  selectedCityTxt: { fontSize: 13, fontWeight: "700", color: C.primary },

  hint: { fontSize: 11, color: C.muted, marginTop: -10, marginBottom: 16, marginLeft: 2 },

  // blood type
  bloodGrid: {
    flexDirection: "row", flexWrap: "wrap",
    gap: 8, justifyContent: "space-between", marginBottom: 4,
  },
  bloodBtn: {
    width: "23%", height: 46, borderRadius: 12,
    borderWidth: 1.5, borderColor: C.border,
    justifyContent: "center", alignItems: "center",
    backgroundColor: C.inputBg,
  },
  bloodBtnActive: { backgroundColor: C.primary, borderColor: C.primary },
  bloodTxt: { fontSize: 14, fontWeight: "700", color: C.muted },
  bloodTxtActive: { color: "#fff" },

  // CTA button
  cta: {
    borderRadius: 16, overflow: "hidden",
    marginTop: 12, marginBottom: 20,
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.28, shadowRadius: 12, elevation: 5,
  },
  ctaGrad: {
    height: 56, flexDirection: "row",
    justifyContent: "center", alignItems: "center",
  },
  ctaTxt: { color: "#fff", fontSize: 16, fontWeight: "800" },

  // footer
  footer: { justifyContent: "center", alignItems: "center" },
  footerTxt: { color: C.sub, fontSize: 14 },
  link: { color: C.primary, fontWeight: "700", fontSize: 14 },

  // iOS date picker done btn
  iosDone: {
    backgroundColor: C.primary, padding: 12,
    borderRadius: 12, alignItems: "center",
    marginTop: 8, marginBottom: 16,
  },
});