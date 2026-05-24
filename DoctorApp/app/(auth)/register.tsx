import { useState, useRef, useEffect } from "react";
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
import { registerApi, saveSession } from "../../services/authService";

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

const EGYPT_CITIES = [
  "Cairo",
  "Giza",
  "Alexandria",
  "Qalyubia",
  "Sharqia",
  "Dakahlia",
  "Gharbia",
  "Monufia",
  "Fayoum",
  "Minya",
  "Assiut",
  "Sohag",
];

export default function RegisterScreen() {
  const router        = useRouter();
  const scrollViewRef = useRef<ScrollView>(null);
  
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
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [gender, setGender] = useState("Male");

  // Step 2: Contact & Location
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [city, setCity] = useState("");
  const [district, setDistrict] = useState("");

  // Step 3: Medical & Security
  const [bloodType, setBloodType] = useState("O+");
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  // Auto-format Date of Birth (YYYY-MM-DD)
  const handleDateChange = (text: string) => {
    const cleaned = text.replace(/[^0-9]/g, '');
    let formatted = cleaned;
    if (cleaned.length > 4 && cleaned.length <= 6) {
      formatted = `${cleaned.slice(0, 4)}-${cleaned.slice(4)}`;
    } else if (cleaned.length > 6) {
      formatted = `${cleaned.slice(0, 4)}-${cleaned.slice(4, 6)}-${cleaned.slice(6, 8)}`;
    }
    setDateOfBirth(formatted);
  };

  const handleNextStep1 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!firstName.trim() || !lastName.trim() || !dateOfBirth.trim()) {
      Toast.show({ type: "error", text1: "Please fill your name and birthday", position: "top", topOffset: 60 });
      return;
    }
    if (dateOfBirth.length !== 10) {
      Toast.show({ type: "error", text1: "Invalid Date", text2: "Use YYYY-MM-DD format (e.g., 1990-05-24)", position: "top", topOffset: 60 });
      return;
    }
    setStep(2);
  };

  const handleNextStep2 = () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!email.trim() || !phone.trim() || !city.trim() || !district.trim()) {
      Toast.show({ type: "error", text1: "Please fill all contact & location details", position: "top", topOffset: 60 });
      return;
    }
    setStep(3);
  };

  const handleRegister = async () => {
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    if (password.length < 6) {
      Toast.show({ type: "error", text1: "Password too short", text2: "At least 6 characters", position: "top", topOffset: 60 });
      return;
    }
    if (!weight.trim() || !height.trim()) {
      Toast.show({ type: "error", text1: "Please enter your weight and height", position: "top", topOffset: 60 });
      return;
    }

    setLoading(true);
    try {
      const address = `${city.trim()} - ${district.trim()}`;
      const auth = await registerApi({
        fullName: `${firstName.trim()} ${lastName.trim()}`,
        email: email.toLowerCase().trim(),
        password: password,
        role: "Patient",
        phoneNumber: phone.trim(),
        address,
        dateOfBirth: dateOfBirth,
        gender: gender,
        bloodType: bloodType,
        weight: Number(weight) || 0,
        height: Number(height) || 0,
        smokingStatus: "Non-Smoker", // Default
      });

      await saveSession(auth);
      Toast.show({ type: "success", text1: `Welcome ${firstName}! 👋`, text2: "Account created successfully", position: "top", topOffset: 60 });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => router.replace("/(auth)/onboarding"), 1600);

    } catch (err: any) {
      Toast.show({ type: "error", text1: err.message || "Registration failed", position: "top", topOffset: 60 });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setLoading(false);
    }
  };

  const renderProgress = () => {
    return (
      <View style={styles.progressContainer}>
        {[1, 2, 3].map((s) => (
          <View key={s} style={{ flexDirection: 'row', alignItems: 'center', flex: s < 3 ? 1 : 0 }}>
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
      
      {/* Dynamic Background */}
      <View style={[styles.bgCircle, styles.circleTop]} />
      <View style={[styles.bgCircle, styles.circleBottom]} />

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined} enabled={Platform.OS === "ios"}>
        <ScrollView ref={scrollViewRef} contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
          
          <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
            
            {/* Header */}
            <View style={styles.headerRow}>
              <TouchableOpacity style={styles.backBtn} onPress={() => step > 1 ? setStep(step - 1) : router.back()}>
                <Ionicons name="arrow-back" size={24} color={COLORS.text} />
              </TouchableOpacity>
              <Text style={styles.stepIndicator}>Step {step} of 3</Text>
            </View>

            <Text style={styles.title}>Join MedBook</Text>
            <Text style={styles.subtitle}>
              {step === 1 ? "Let's start with your personal info" : step === 2 ? "How can we reach you?" : "Secure your medical profile"}
            </Text>

            {renderProgress()}

            <View style={styles.glassCard}>
              
              {/* STEP 1: PERSONAL INFO */}
              {step === 1 && (
                <View>
                  <View style={styles.row}>
                    <View style={styles.flex1}>
                      <Text style={styles.inputLabel}>First Name</Text>
                      <View style={[styles.inputWrapper, focusedInput === "first" && styles.inputWrapperFocused]}>
                        <Ionicons name="person-outline" size={20} color={focusedInput === "first" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                        <TextInput
                          placeholder="e.g. Ahmed" style={styles.input} placeholderTextColor="#94A3B8"
                          value={firstName} onChangeText={setFirstName}
                          onFocus={() => setFocusedInput("first")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                    <View style={styles.flex1}>
                      <Text style={styles.inputLabel}>Last Name</Text>
                      <View style={[styles.inputWrapper, focusedInput === "last" && styles.inputWrapperFocused]}>
                        <TextInput
                          placeholder="e.g. Ali" style={styles.input} placeholderTextColor="#94A3B8"
                          value={lastName} onChangeText={setLastName}
                          onFocus={() => setFocusedInput("last")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                  </View>

                  <Text style={styles.inputLabel}>Date of Birth</Text>
                  <View style={[styles.inputWrapper, focusedInput === "dob" && styles.inputWrapperFocused]}>
                    <Ionicons name="calendar-outline" size={20} color={dateOfBirth.length === 10 ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                    <TextInput
                      placeholder="YYYY-MM-DD" placeholderTextColor="#94A3B8" keyboardType="numeric"
                      style={styles.input} value={dateOfBirth} onChangeText={handleDateChange} maxLength={10}
                      onFocus={() => setFocusedInput("dob")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>
                  <Text style={styles.inputHint}>Format: Year-Month-Day (e.g., 1990-05-24)</Text>

                  <Text style={styles.inputLabel}>Gender</Text>
                  <View style={styles.row}>
                    {["Male", "Female"].map((g) => (
                      <TouchableOpacity
                        key={g}
                        style={[styles.genderBtn, gender === g && styles.genderBtnActive]}
                        onPress={() => setGender(g)}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.genderText, gender === g && styles.genderTextActive]}>{g}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>

                  <TouchableOpacity style={styles.btnWrap} onPress={handleNextStep1} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={styles.btnGradient}>
                      <Text style={styles.btnText}>Continue</Text>
                      <Ionicons name="arrow-forward" size={20} color="#fff" style={{ marginLeft: 8 }} />
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              {/* STEP 2: CONTACT & LOCATION */}
              {step === 2 && (
                <View>
                  <Text style={styles.inputLabel}>Email Address</Text>
                  <View style={[styles.inputWrapper, focusedInput === "email" && styles.inputWrapperFocused]}>
                    <Ionicons name="mail-outline" size={20} color={focusedInput === "email" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                    <TextInput
                      placeholder="ahmed@example.com" keyboardType="email-address" autoCapitalize="none"
                      style={styles.input} value={email} onChangeText={setEmail} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("email")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>

                  <Text style={styles.inputLabel}>Phone Number</Text>
                  <View style={[styles.inputWrapper, focusedInput === "phone" && styles.inputWrapperFocused]}>
                    <Ionicons name="call-outline" size={20} color={focusedInput === "phone" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                    <TextInput
                      placeholder="01012345678" keyboardType="phone-pad" maxLength={11}
                      style={styles.input} value={phone} onChangeText={setPhone} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("phone")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>

                  <Text style={styles.inputLabel}>City</Text>
                  <View style={styles.cityGrid}>
                    {EGYPT_CITIES.map((item) => (
                      <TouchableOpacity
                        key={item}
                        style={[styles.cityOption, city === item && styles.cityOptionActive]}
                        onPress={() => setCity(item)}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.cityOptionText, city === item && styles.cityOptionTextActive]}>
                          {item}
                        </Text>
                      </TouchableOpacity>
                    ))}
                  </View>

                  <Text style={styles.inputLabel}>District / Area</Text>
                  <View style={[styles.inputWrapper, focusedInput === "district" && styles.inputWrapperFocused]}>
                    <Ionicons name="location-outline" size={20} color={focusedInput === "district" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                    <TextInput
                      placeholder="e.g. Helwan, Nasr City"
                      style={styles.input} value={district} onChangeText={setDistrict} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("district")} onBlur={() => setFocusedInput(null)}
                    />
                  </View>
                  <Text style={styles.inputHint}>Choose the city, then type your district or neighborhood.</Text>

                  <TouchableOpacity style={styles.btnWrap} onPress={handleNextStep2} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={styles.btnGradient}>
                      <Text style={styles.btnText}>Continue</Text>
                      <Ionicons name="arrow-forward" size={20} color="#fff" style={{ marginLeft: 8 }} />
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              {/* STEP 3: MEDICAL & SECURITY */}
              {step === 3 && (
                <View>
                  <View style={styles.row}>
                    <View style={styles.flex1}>
                      <Text style={styles.inputLabel}>Weight (kg)</Text>
                      <View style={[styles.inputWrapper, focusedInput === "weight" && styles.inputWrapperFocused]}>
                        <Ionicons name="scale-outline" size={20} color={focusedInput === "weight" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                        <TextInput
                          placeholder="75" keyboardType="numeric" maxLength={3}
                          style={styles.input} value={weight} onChangeText={setWeight} placeholderTextColor="#94A3B8"
                          onFocus={() => setFocusedInput("weight")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                    <View style={styles.flex1}>
                      <Text style={styles.inputLabel}>Height (cm)</Text>
                      <View style={[styles.inputWrapper, focusedInput === "height" && styles.inputWrapperFocused]}>
                        <Ionicons name="body-outline" size={20} color={focusedInput === "height" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                        <TextInput
                          placeholder="175" keyboardType="numeric" maxLength={3}
                          style={styles.input} value={height} onChangeText={setHeight} placeholderTextColor="#94A3B8"
                          onFocus={() => setFocusedInput("height")} onBlur={() => setFocusedInput(null)}
                        />
                      </View>
                    </View>
                  </View>

                  <Text style={styles.inputLabel}>Blood Type</Text>
                  <View style={styles.bloodTypesGrid}>
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

                  <Text style={[styles.inputLabel, { marginTop: 16 }]}>Create Password</Text>
                  <View style={[styles.inputWrapper, focusedInput === "password" && styles.inputWrapperFocused]}>
                    <Ionicons name="lock-closed-outline" size={20} color={focusedInput === "password" ? COLORS.primary : COLORS.textMuted} style={styles.inputIcon} />
                    <TextInput
                      placeholder="Min. 6 characters" secureTextEntry={!showPassword}
                      style={styles.input} value={password} onChangeText={setPassword} placeholderTextColor="#94A3B8"
                      onFocus={() => setFocusedInput("password")} onBlur={() => setFocusedInput(null)}
                    />
                    <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={{ padding: 8 }}>
                      <Ionicons name={showPassword ? "eye-outline" : "eye-off-outline"} size={20} color={COLORS.textMuted} />
                    </TouchableOpacity>
                  </View>

                  <TouchableOpacity style={[styles.btnWrap, loading && { opacity: 0.7 }]} onPress={handleRegister} disabled={loading} activeOpacity={0.8}>
                    <LinearGradient colors={[COLORS.primary, '#047857']} style={styles.btnGradient}>
                      {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnText}>Create Account</Text>}
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              )}

              <View style={styles.footer}>
                <Text style={styles.footerText}>Already have an account? </Text>
                <TouchableOpacity onPress={() => router.push("/(auth)/login")} activeOpacity={0.7}>
                  <Text style={styles.linkText}>Log In</Text>
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
  headerRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 20 },
  backBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: "rgba(255,255,255,0.8)", justifyContent: "center", alignItems: "center" },
  stepIndicator: { fontSize: 14, fontWeight: "600", color: COLORS.primary },
  title: { fontSize: 32, fontWeight: "800", color: COLORS.text, letterSpacing: -0.5 },
  subtitle: { fontSize: 15, color: COLORS.textMuted, marginTop: 8, marginBottom: 24 },
  progressContainer: { flexDirection: "row", alignItems: "center", marginBottom: 30, paddingHorizontal: 10 },
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
  row: { flexDirection: "row", gap: 12 },
  flex1: { flex: 1 },
  inputLabel: { fontSize: 12, fontWeight: "700", color: COLORS.text, marginBottom: 8, marginLeft: 4, textTransform: "uppercase", letterSpacing: 0.5 },
  inputHint: { fontSize: 11, color: COLORS.textMuted, marginTop: -10, marginBottom: 16, marginLeft: 4 },
  inputWrapper: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#F8FAFC", borderWidth: 1,
    borderColor: COLORS.border, borderRadius: 16, height: 56, paddingHorizontal: 16, marginBottom: 16
  },
  inputWrapperFocused: { borderColor: COLORS.primary, backgroundColor: "#fff" },
  inputIcon: { marginRight: 12 },
  input: { flex: 1, fontSize: 16, color: COLORS.text, height: "100%" },
  genderBtn: { flex: 1, height: 56, borderRadius: 16, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center", marginBottom: 20 },
  genderBtnActive: { backgroundColor: "rgba(16, 185, 129, 0.1)", borderColor: COLORS.primary },
  genderText: { fontSize: 15, fontWeight: "600", color: COLORS.textMuted },
  genderTextActive: { color: COLORS.primary },
  bloodTypesGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8, justifyContent: "space-between" },
  bloodBtn: { width: "23%", height: 45, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center" },
  bloodBtnActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  bloodText: { fontSize: 14, fontWeight: "600", color: COLORS.textMuted },
  bloodTextActive: { color: "#fff" },
  cityGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 14 },
  cityOption: { paddingHorizontal: 12, height: 38, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, justifyContent: "center", alignItems: "center", backgroundColor: "#F8FAFC" },
  cityOptionActive: { backgroundColor: "rgba(16, 185, 129, 0.12)", borderColor: COLORS.primary },
  cityOptionText: { fontSize: 12, fontWeight: "700", color: COLORS.textMuted },
  cityOptionTextActive: { color: COLORS.primary },
  btnWrap: { width: "100%", borderRadius: 16, overflow: "hidden", shadowColor: COLORS.primary, shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.25, shadowRadius: 12, elevation: 6, marginTop: 10, marginBottom: 24 },
  btnGradient: { flexDirection: "row", height: 56, justifyContent: "center", alignItems: "center" },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "700" },
  footer: { flexDirection: "row", justifyContent: "center", alignItems: "center" },
  footerText: { color: COLORS.textMuted, fontSize: 14 },
  linkText: { color: COLORS.primary, fontWeight: "700", fontSize: 14 }
});
