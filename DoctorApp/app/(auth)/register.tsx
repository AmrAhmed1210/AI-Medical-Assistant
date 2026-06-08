import React, { useState, useRef, useEffect } from "react";
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, TouchableOpacity, Keyboard, ActivityIndicator,
  Dimensions, Animated, Easing, StatusBar
} from "react-native";
import { useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import DateTimePicker from "@react-native-community/datetimepicker";
import { registerApi, saveSession } from "../../services/authService";
import { useLanguage } from "../../context/LanguageContext";

const { width } = Dimensions.get("window");

const COLORS = {
  primary: "#059669",
  primaryLight: "#10B981",
  secondary: "#0284C7",
  bg: "#F8FAFC",
  surface: "#FFFFFF",
  text: "#0F172A",
  textMuted: "#64748B",
  border: "#E2E8F0"
};

const EGYPT_CITIES_EN = [
  "Cairo", "Giza", "Alexandria", "Qalyubia", "Sharqia", "Dakahlia", 
  "Gharbia", "Monufia", "Fayoum", "Minya", "Assiut", "Sohag"
];

const EGYPT_CITIES_AR = [
  "القاهرة", "الجيزة", "الإسكندرية", "القليوبية", "الشرقية", "الدقهلية",
  "الغربية", "المنوفية", "الفيوم", "المنيا", "أسيوط", "سوهاج"
];

export default function RegisterScreen() {
  const router        = useRouter();
  const scrollViewRef = useRef<ScrollView>(null);
  const { tr, isRTL } = useLanguage();
  
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(40)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      })
    ]).start();
  }, []);

  const [step, setStep] = useState(1);
  const [focusedInput, setFocusedInput] = useState<string | null>(null);

  // Step 1: Personal
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [dobDate, setDobDate] = useState<Date | null>(null);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [gender, setGender] = useState("Male");

  // Step 2: Contact & Location
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [country, setCountry] = useState("Egypt");
  const [city, setCity] = useState("");
  const [district, setDistrict] = useState("");

  // Step 3: Medical & Security
  const [bloodType, setBloodType] = useState("O+");
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const EGYPT_CITIES = isRTL ? EGYPT_CITIES_AR : EGYPT_CITIES_EN;

  const handleNextStep1 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!firstName.trim() || !lastName.trim() || !dobDate) {
      Toast.show({ type: "error", text1: tr("error"), text2: tr("please_write_comment") || "Please fill all fields", position: "top" });
      return;
    }
    setStep(2);
  };

  const handleNextStep2 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!email.trim() || !phone.trim() || !city.trim() || !district.trim()) {
      Toast.show({ type: "error", text1: tr("error"), position: "top" });
      return;
    }
    setStep(3);
  };

  const handleRegister = async () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    if (password.length < 6) {
      Toast.show({ type: "error", text1: tr("min_6_chars"), position: "top" });
      return;
    }
    if (!weight.trim() || !height.trim()) {
      Toast.show({ type: "error", text1: tr("error"), position: "top" });
      return;
    }

    setLoading(true);
    try {
      const address = `${country} - ${city.trim()} - ${district.trim()}`;
      
      const tzOffset = dobDate!.getTimezoneOffset() * 60000;
      const localISOTime = new Date(dobDate!.getTime() - tzOffset).toISOString().split('T')[0];

      const auth = await registerApi({
        fullName: `${firstName.trim()} ${lastName.trim()}`,
        email: email.toLowerCase().trim(),
        password: password,
        role: "Patient",
        phoneNumber: phone.trim(),
        address,
        dateOfBirth: localISOTime,
        gender: gender === "Male" ? "Male" : "Female",
        bloodType: bloodType,
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

  const renderProgress = () => {
    return (
      <View style={[styles.progressContainer, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
        {[1, 2, 3].map((s) => (
          <View key={s} style={{ flexDirection: isRTL ? 'row-reverse' : 'row', alignItems: 'center', flex: s < 3 ? 1 : 0 }}>
            <View style={[styles.progressDot, step >= s && styles.progressDotActive]}>
              {step > s ? (
                <Ionicons name="checkmark" size={16} color="#fff" />
              ) : (
                <Text style={[styles.progressDotText, step >= s && styles.progressDotTextActive]}>{s}</Text>
              )}
            </View>
            {s < 3 && <View style={[styles.progressLine, step > s && styles.progressLineActive]} />}
          </View>
        ))}
      </View>
    );
  };

  return (
    <View style={styles.mainContainer}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />
      
      <View style={[styles.bgCircle, styles.circleTop]} />
      <View style={[styles.bgCircle, styles.circleBottom]} />

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined} enabled={Platform.OS === "ios"}>
        <ScrollView ref={scrollViewRef} contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
          
          <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
            
            <View style={[styles.headerRow, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
              <TouchableOpacity style={styles.backBtn} onPress={() => step > 1 ? setStep(step - 1) : router.back()}>
                <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={24} color={COLORS.text} />
              </TouchableOpacity>
              <Text style={styles.stepIndicator}>
                {step === 1 ? tr("step_1") : step === 2 ? tr("step_2") : tr("step_3")}
              </Text>
            </View>

            <Text style={[styles.title, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("join_medbook")}</Text>
            <Text style={[styles.subtitle, { textAlign: isRTL ? 'right' : 'left' }]}>
              {step === 1 ? tr("step_1_subtitle") : step === 2 ? tr("step_2_subtitle") : tr("step_3_subtitle")}
            </Text>

            {renderProgress()}

            <View style={styles.glassCard}>
              
              {/* STEP 1: PERSONAL INFO */}
              {step === 1 && (
                <View>
                  <View style={[styles.row, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    <View style={styles.flex1}>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("first_name")}</Text>
                      <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "first" && styles.inputWrapperFocused]}>
                        <Ionicons name="person-outline" size={20} color={focusedInput === "first" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                        <TextInput
                          style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} placeholderTextColor="#94A3B8"
                          value={firstName} onChangeText={setFirstName}
                          onFocus={() => setFocusedInput("first")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                    <View style={styles.flex1}>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("last_name")}</Text>
                      <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "last" && styles.inputWrapperFocused]}>
                        <TextInput
                          style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} placeholderTextColor="#94A3B8"
                          value={lastName} onChangeText={setLastName}
                          onFocus={() => setFocusedInput("last")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                  </View>

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("date_of_birth")}</Text>
                  <TouchableOpacity onPress={() => setShowDatePicker(true)} style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    <Ionicons name="calendar-outline" size={20} color={dobDate ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                    <Text style={[styles.input, { textAlign: isRTL ? 'right' : 'left', color: dobDate ? COLORS.text : "#94A3B8", paddingTop: 16 }]}>
                      {dobDate ? dobDate.toLocaleDateString(isRTL ? 'ar-EG' : 'en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : tr("dob_hint")}
                    </Text>
                  </TouchableOpacity>

                  {showDatePicker && (
                    <DateTimePicker
                      value={dobDate || new Date(2000, 0, 1)}
                      mode="date"
                      display="spinner"
                      maximumDate={new Date()}
                      onChange={(event, date) => {
                        if (Platform.OS === 'android') setShowDatePicker(false);
                        if (date) setDobDate(date);
                      }}
                    />
                  )}
                  {Platform.OS === 'ios' && showDatePicker && (
                    <TouchableOpacity style={styles.iosDateDoneBtn} onPress={() => setShowDatePicker(false)}>
                      <Text style={{ color: '#fff', fontWeight: 'bold' }}>{tr("confirm")}</Text>
                    </TouchableOpacity>
                  )}

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left', marginTop: 12 }]}>{tr("gender")}</Text>
                  <View style={[styles.row, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    {["Male", "Female"].map((g) => (
                      <TouchableOpacity
                        key={g}
                        style={[styles.genderBtn, gender === g && styles.genderBtnActive]}
                        onPress={() => setGender(g)}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.genderText, gender === g && styles.genderTextActive]}>{g === "Male" ? tr("male") : tr("female")}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>

                  <TouchableOpacity style={styles.btnWrap} onPress={handleNextStep1} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={[styles.btnGradient, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                      <Text style={styles.btnText}>{tr("continue_btn")}</Text>
                      <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={20} color="#fff" style={isRTL ? { marginRight: 8 } : { marginLeft: 8 }} />
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              {/* STEP 2: CONTACT & LOCATION */}
              {step === 2 && (
                <View>
                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("email")}</Text>
                  <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "email" && styles.inputWrapperFocused]}>
                    <Ionicons name="mail-outline" size={20} color={focusedInput === "email" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                    <TextInput
                      keyboardType="email-address" autoCapitalize="none"
                      style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={email} onChangeText={setEmail} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("email")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("phone")}</Text>
                  <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "phone" && styles.inputWrapperFocused]}>
                    <Ionicons name="call-outline" size={20} color={focusedInput === "phone" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                    <TextInput
                      keyboardType="phone-pad" maxLength={15}
                      style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={phone} onChangeText={setPhone} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("phone")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("country")}</Text>
                  <View style={[styles.cityGrid, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    {["Egypt", "Other"].map((item) => (
                      <TouchableOpacity
                        key={item}
                        style={[styles.cityOption, country === item && styles.cityOptionActive]}
                        onPress={() => { setCountry(item); setCity(""); setDistrict(""); }}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.cityOptionText, country === item && styles.cityOptionTextActive]}>
                          {item === "Egypt" ? tr("egypt") : tr("other")}
                        </Text>
                      </TouchableOpacity>
                    ))}
                  </View>

                  {country === "Egypt" ? (
                    <>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("city")}</Text>
                      <View style={[styles.cityGrid, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                        {EGYPT_CITIES.map((item, index) => (
                          <TouchableOpacity
                            key={index}
                            style={[styles.cityOption, city === EGYPT_CITIES_EN[index] && styles.cityOptionActive]}
                            onPress={() => setCity(EGYPT_CITIES_EN[index])}
                            activeOpacity={0.8}
                          >
                            <Text style={[styles.cityOptionText, city === EGYPT_CITIES_EN[index] && styles.cityOptionTextActive]}>
                              {item}
                            </Text>
                          </TouchableOpacity>
                        ))}
                      </View>
                    </>
                  ) : (
                    <>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("city")}</Text>
                      <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "city" && styles.inputWrapperFocused]}>
                        <Ionicons name="location-outline" size={20} color={focusedInput === "city" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                        <TextInput
                          style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={city} onChangeText={setCity} placeholderTextColor="#94A3B8"
                          onFocus={() => setFocusedInput("city")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </>
                  )}

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("district_area")}</Text>
                  <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "district" && styles.inputWrapperFocused]}>
                    <Ionicons name="location-outline" size={20} color={focusedInput === "district" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                    <TextInput
                      style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={district} onChangeText={setDistrict} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("district")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>
                  <Text style={[styles.inputHint, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("district_hint")}</Text>

                  <TouchableOpacity style={styles.btnWrap} onPress={handleNextStep2} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={[styles.btnGradient, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                      <Text style={styles.btnText}>{tr("continue_btn")}</Text>
                      <Ionicons name={isRTL ? "arrow-back" : "arrow-forward"} size={20} color="#fff" style={isRTL ? { marginRight: 8 } : { marginLeft: 8 }} />
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              {/* STEP 3: MEDICAL & SECURITY */}
              {step === 3 && (
                <View>
                  <View style={[styles.row, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    <View style={styles.flex1}>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("weight_kg")}</Text>
                      <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "weight" && styles.inputWrapperFocused]}>
                        <Ionicons name="scale-outline" size={20} color={focusedInput === "weight" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                        <TextInput
                          keyboardType="numeric" maxLength={3}
                          style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={weight} onChangeText={setWeight} placeholderTextColor="#94A3B8"
                          onFocus={() => setFocusedInput("weight")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                    <View style={styles.flex1}>
                      <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("height_cm")}</Text>
                      <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "height" && styles.inputWrapperFocused]}>
                        <Ionicons name="body-outline" size={20} color={focusedInput === "height" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                        <TextInput
                          keyboardType="numeric" maxLength={3}
                          style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={height} onChangeText={setHeight} placeholderTextColor="#94A3B8"
                          onFocus={() => setFocusedInput("height")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                  </View>

                  <Text style={[styles.inputLabel, { textAlign: isRTL ? 'right' : 'left' }]}>{tr("blood_type")}</Text>
                  <View style={[styles.bloodTypesGrid, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                    {["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"].map((bt) => (
                      <TouchableOpacity
                        key={bt}
                        style={[styles.bloodBtn, bloodType === bt && styles.bloodBtnActive]}
                        onPress={() => setBloodType(bt)}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.bloodText, bloodType === bt && styles.bloodTextActive]}>{bt}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>

                  <Text style={[styles.inputLabel, { marginTop: 16, textAlign: isRTL ? 'right' : 'left' }]}>{tr("create_password")}</Text>
                  <View style={[styles.inputWrapper, { flexDirection: isRTL ? 'row-reverse' : 'row' }, focusedInput === "password" && styles.inputWrapperFocused]}>
                    <Ionicons name="lock-closed-outline" size={20} color={focusedInput === "password" ? COLORS.primary : COLORS.textMuted} style={isRTL ? { marginLeft: 12 } : { marginRight: 12 }} />
                    <TextInput
                      secureTextEntry={!showPassword}
                      style={[styles.input, { textAlign: isRTL ? 'right' : 'left' }]} value={password} onChangeText={setPassword} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("password")} onBlur={() => setFocusedInput(null)}
                    />
                    <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={{ padding: 8 }}>
                      <Ionicons name={showPassword ? "eye-outline" : "eye-off-outline"} size={20} color={COLORS.textMuted} />
                    </TouchableOpacity>
                  </View>

                  <TouchableOpacity style={[styles.btnWrap, loading && { opacity: 0.7 }]} onPress={handleRegister} disabled={loading} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={[styles.btnGradient, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                      {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnText}>{tr("create_account")}</Text>}
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              <View style={[styles.footer, { flexDirection: isRTL ? 'row-reverse' : 'row' }]}>
                <Text style={styles.footerText}>{tr("already_have_account")}</Text>
                <TouchableOpacity onPress={() => router.push("/(auth)/login")} activeOpacity={0.7}>
                  <Text style={styles.linkText}>{tr("log_in")}</Text>
                </TouchableOpacity>
              </View>

            </View>
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
      <Toast />
    </View>
  );
}

const styles = StyleSheet.create({
  mainContainer: { flex: 1, backgroundColor: COLORS.bg },
  bgCircle: { position: 'absolute', borderRadius: 200, opacity: 0.3 },
  circleTop: { width: 300, height: 300, backgroundColor: 'rgba(16, 185, 129, 0.15)', top: -100, right: -100 },
  circleBottom: { width: 350, height: 350, backgroundColor: 'rgba(2, 132, 199, 0.1)', bottom: -150, left: -150 },
  scrollContainer: { flexGrow: 1, padding: 24, paddingTop: Platform.OS === 'ios' ? 60 : 40, paddingBottom: 60 },
  headerRow: { alignItems: "center", justifyContent: "space-between", marginBottom: 20 },
  backBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: "rgba(255,255,255,0.8)", justifyContent: "center", alignItems: "center" },
  stepIndicator: { fontSize: 14, fontWeight: "600", color: COLORS.primary },
  title: { fontSize: 32, fontWeight: "800", color: COLORS.text, letterSpacing: -0.5 },
  subtitle: { fontSize: 15, color: COLORS.textMuted, marginTop: 8, marginBottom: 24 },
  progressContainer: { alignItems: "center", marginBottom: 30, paddingHorizontal: 10 },
  progressDot: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.border, justifyContent: "center", alignItems: "center" },
  progressDotActive: { backgroundColor: COLORS.primary },
  progressDotText: { color: COLORS.textMuted, fontSize: 12, fontWeight: "700" },
  progressDotTextActive: { color: "#fff" },
  progressLine: { flex: 1, height: 3, backgroundColor: COLORS.border, marginHorizontal: 8, borderRadius: 2 },
  progressLineActive: { backgroundColor: COLORS.primary },
  glassCard: {
    backgroundColor: "rgba(255, 255, 255, 0.9)", borderRadius: 28, padding: 24,
    shadowColor: "#000", shadowOffset: { width: 0, height: 20 }, shadowOpacity: 0.05, shadowRadius: 30,
    elevation: 5, borderWidth: 1, borderColor: "rgba(255, 255, 255, 0.5)",
  },
  row: { gap: 12 },
  flex1: { flex: 1 },
  inputLabel: { fontSize: 12, fontWeight: "700", color: COLORS.text, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 },
  inputHint: { fontSize: 11, color: COLORS.textMuted, marginTop: -10, marginBottom: 16 },
  inputWrapper: {
    alignItems: "center", backgroundColor: "#F8FAFC", borderWidth: 1,
    borderColor: COLORS.border, borderRadius: 16, height: 56, paddingHorizontal: 16, marginBottom: 16
  },
  inputWrapperFocused: { borderColor: COLORS.primary, backgroundColor: "#fff" },
  input: { flex: 1, fontSize: 16, color: COLORS.text, height: "100%" },
  genderBtn: { flex: 1, height: 56, borderRadius: 16, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center", marginBottom: 20 },
  genderBtnActive: { backgroundColor: "rgba(16, 185, 129, 0.1)", borderColor: COLORS.primary },
  genderText: { fontSize: 15, fontWeight: "600", color: COLORS.textMuted },
  genderTextActive: { color: COLORS.primary },
  bloodTypesGrid: { flexWrap: "wrap", gap: 8, justifyContent: "space-between" },
  bloodBtn: { width: "23%", height: 45, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center" },
  bloodBtnActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  bloodText: { fontSize: 14, fontWeight: "600", color: COLORS.textMuted },
  bloodTextActive: { color: "#fff" },
  cityGrid: { flexWrap: "wrap", gap: 8, marginBottom: 14 },
  cityOption: { paddingHorizontal: 12, height: 38, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  cityOptionActive: { backgroundColor: "rgba(16, 185, 129, 0.12)", borderColor: COLORS.primary },
  cityOptionText: { fontSize: 12, fontWeight: "700", color: COLORS.textMuted },
  cityOptionTextActive: { color: COLORS.primary },
  btnWrap: { width: "100%", borderRadius: 16, overflow: "hidden", shadowColor: COLORS.primary, shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.25, shadowRadius: 12, elevation: 6, marginTop: 10, marginBottom: 24 },
  btnGradient: { height: 56, justifyContent: "center", alignItems: "center" },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "700" },
  footer: { justifyContent: "center", alignItems: "center" },
  footerText: { color: COLORS.textMuted, fontSize: 14 },
  linkText: { color: COLORS.primary, fontWeight: "700", fontSize: 14 },
  iosDateDoneBtn: { backgroundColor: COLORS.primary, padding: 12, borderRadius: 12, alignItems: 'center', marginTop: 10, marginBottom: 20 }
});
