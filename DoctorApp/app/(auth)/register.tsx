import { useState, useRef } from "react";
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, TouchableOpacity, Keyboard, ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { registerApi, saveSession } from "../../services/authService";

// Medical Colors
const MED_COLORS = {
  primary: "#0EA5E9", // Sky Blue
  secondary: "#0284C7",
  accent: "#06B6D4", // Cyan
  bg: "#F8FAFC",
  text: "#0F172A",
  muted: "#64748B"
};

export default function RegisterScreen() {
  const router        = useRouter();
  const scrollViewRef = useRef<ScrollView>(null);
  const inputsRef     = useRef<(TextInput | null)[]>([]);

  const [firstName,           setFirstName]           = useState("");
  const [lastName,            setLastName]             = useState("");
  const [email,               setEmail]               = useState("");
  const [phone,               setPhone]               = useState("");
  const [password,            setPassword]            = useState("");
  const [confirmPassword,     setConfirmPassword]     = useState("");
  const [dateOfBirth,         setDateOfBirth]         = useState("");
  const [gender,              setGender]              = useState("Male");
  const [bloodType,           setBloodType]           = useState("O+");
  const [weight,              setWeight]              = useState("");
  const [height,              setHeight]              = useState("");
  const [smokingStatus,       setSmokingStatus]       = useState("Non-Smoker");
  const [showPassword,        setShowPassword]        = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading,             setLoading]             = useState(false);

  const focusNext   = (i: number) => inputsRef.current[i + 1]?.focus();
  const handleFocus = (i: number) =>
    setTimeout(() => scrollViewRef.current?.scrollTo({ y: i * 70, animated: true }), 100);

  const handleRegister = async () => {
    Keyboard.dismiss();

    if (!firstName.trim() || !lastName.trim() || !email.trim() || !phone.trim() || !password || !confirmPassword || !dateOfBirth.trim()) {
      Toast.show({ type: "error", text1: "Please fill all fields", position: "top", topOffset: 60 });
      return;
    }
    
    const phoneRegex = /^01[0125]\d{8}$/;
    if (!phoneRegex.test(phone.trim())) {
      Toast.show({ type: "error", text1: "Invalid Phone Number", text2: "Must be 11 digits starting with 01", position: "top", topOffset: 60 });
      return;
    }

    if (password.length < 6) {
      Toast.show({ type: "error", text1: "Password too short", text2: "At least 6 characters", position: "top", topOffset: 60 });
      return;
    }
    if (password !== confirmPassword) {
      Toast.show({ type: "error", text1: "Passwords don't match", position: "top", topOffset: 60 });
      return;
    }

    setLoading(true);
    try {
      const auth = await registerApi({
        fullName:     `${firstName.trim()} ${lastName.trim()}`,
        email:        email.toLowerCase().trim(),
        password:     password,
        role:         "Patient",
        phoneNumber:  phone.trim(),
        dateOfBirth:  dateOfBirth,
        gender:       gender,
        bloodType:    bloodType,
        weight:       Number(weight) || 0,
        height:       Number(height) || 0,
        smokingStatus: smokingStatus,
      });

      await saveSession(auth);
      Toast.show({ type: "success", text1: `Welcome ${firstName}! 👋`, text2: "Account created successfully", position: "top", topOffset: 60 });
      setTimeout(() => router.replace("/(patient)/home"), 1600);

    } catch (err: any) {
      Toast.show({ type: "error", text1: err.message || "Registration failed", position: "top", topOffset: 60 });
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: MED_COLORS.bg }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
    >
      <ScrollView
        ref={scrollViewRef}
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Join MedBook</Text>
        <Text style={styles.subTitle}>Create your luxury medical profile</Text>

        <View style={{ flexDirection: "row", gap: 12 }}>
          <TextInput
            ref={r => { inputsRef.current[0] = r; }}
            placeholder="First Name" style={[styles.input, { flex: 1 }]}
            value={firstName} onChangeText={setFirstName}
            returnKeyType="next" onSubmitEditing={() => focusNext(0)}
            onFocus={() => handleFocus(0)} editable={!loading}
          />
          <TextInput
            ref={r => { inputsRef.current[1] = r; }}
            placeholder="Last Name" style={[styles.input, { flex: 1 }]}
            value={lastName} onChangeText={setLastName}
            returnKeyType="next" onSubmitEditing={() => focusNext(1)}
            onFocus={() => handleFocus(1)} editable={!loading}
          />
        </View>

        <TextInput
          ref={r => { inputsRef.current[2] = r; }}
          placeholder="Email Address" keyboardType="email-address" autoCapitalize="none"
          style={styles.input} value={email} onChangeText={setEmail}
          returnKeyType="next" onSubmitEditing={() => focusNext(2)}
          onFocus={() => handleFocus(2)} editable={!loading}
        />

        <TextInput
          ref={r => { inputsRef.current[3] = r; }}
          placeholder="Phone Number (e.g. 01012345678)" keyboardType="phone-pad"
          style={styles.input} value={phone} onChangeText={setPhone}
          returnKeyType="next" onSubmitEditing={() => focusNext(3)}
          onFocus={() => handleFocus(3)} editable={!loading}
        />

        <TextInput
          ref={r => { inputsRef.current[4] = r; }}
          placeholder="Date of Birth (YYYY-MM-DD)"
          style={styles.input} value={dateOfBirth} onChangeText={setDateOfBirth}
          returnKeyType="next" onSubmitEditing={() => focusNext(4)}
          onFocus={() => handleFocus(4)} editable={!loading}
        />

        {/* Weight & Height */}
        <View style={{ flexDirection: "row", gap: 12 }}>
          <TextInput
            placeholder="Weight (kg)" keyboardType="numeric"
            style={[styles.input, { flex: 1 }]} value={weight} onChangeText={setWeight}
            editable={!loading}
          />
          <TextInput
            placeholder="Height (cm)" keyboardType="numeric"
            style={[styles.input, { flex: 1 }]} value={height} onChangeText={setHeight}
            editable={!loading}
          />
        </View>

        <View style={styles.selectorLabelRow}>
          <Text style={styles.selectorLabel}>Gender</Text>
          <Text style={styles.selectorLabel}>Blood Type</Text>
        </View>

        <View style={styles.selectorRow}>
          <View style={styles.genderWrap}>
            {["Male", "Female"].map(g => (
              <TouchableOpacity 
                key={g} 
                style={[styles.miniBtn, gender === g && styles.miniBtnActive]} 
                onPress={() => setGender(g)}
              >
                <Text style={[styles.miniBtnText, gender === g && styles.miniBtnTextActive]}>{g}</Text>
              </TouchableOpacity>
            ))}
          </View>
          <View style={styles.bloodWrap}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].map(bt => (
                <TouchableOpacity 
                  key={bt} 
                  style={[styles.bloodBtn, bloodType === bt && styles.bloodBtnActive]} 
                  onPress={() => setBloodType(bt)}
                >
                  <Text style={[styles.bloodBtnText, bloodType === bt && styles.bloodBtnTextActive]}>{bt}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>

        <Text style={styles.selectorLabel}>Smoking Status</Text>
        <View style={[styles.selectorRow, { marginTop: 8 }]}>
           {["Non-Smoker", "Smoker", "Ex-Smoker"].map(s => (
              <TouchableOpacity 
                key={s} 
                style={[styles.miniBtn, smokingStatus === s && styles.miniBtnActive, { marginRight: 8 }]} 
                onPress={() => setSmokingStatus(s)}
              >
                <Text style={[styles.miniBtnText, smokingStatus === s && styles.miniBtnTextActive]}>{s}</Text>
              </TouchableOpacity>
            ))}
        </View>

        <View style={styles.passWrap}>
          <TextInput
            ref={r => { inputsRef.current[5] = r; }}
            placeholder="Password" secureTextEntry={!showPassword}
            style={styles.passInput} value={password} onChangeText={setPassword}
            returnKeyType="next" onSubmitEditing={() => focusNext(5)}
            onFocus={() => handleFocus(5)} editable={!loading}
          />
          <TouchableOpacity onPress={() => setShowPassword(p => !p)} style={styles.eyeBtn}>
            <Ionicons name={showPassword ? "eye-off" : "eye"} size={22} color={MED_COLORS.muted} />
          </TouchableOpacity>
        </View>

        <View style={styles.passWrap}>
          <TextInput
            ref={r => { inputsRef.current[6] = r; }}
            placeholder="Confirm Password" secureTextEntry={!showConfirmPassword}
            style={styles.passInput} value={confirmPassword} onChangeText={setConfirmPassword}
            returnKeyType="done" onSubmitEditing={handleRegister}
            onFocus={() => handleFocus(6)} editable={!loading}
          />
          <TouchableOpacity onPress={() => setShowConfirmPassword(p => !p)} style={styles.eyeBtn}>
            <Ionicons name={showConfirmPassword ? "eye-off" : "eye"} size={22} color={MED_COLORS.muted} />
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={[styles.btn, loading && styles.btnDisabled]}
          onPress={handleRegister}
          disabled={loading}
          activeOpacity={0.8}
        >
          {loading
            ? <ActivityIndicator size="small" color="#fff" />
            : <Text style={styles.btnText}>Create Account</Text>
          }
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.push("/(auth)/login")} style={styles.loginLink}>
          <Text style={styles.loginTxt}>
            Already have an account?{" "}
            <Text style={{ color: MED_COLORS.primary, fontWeight: "700" }}>Login</Text>
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container:   { flexGrow: 1, padding: 25, paddingTop: 60, paddingBottom: 40, backgroundColor: MED_COLORS.bg },
  title:       { fontSize: 32, fontWeight: "800", color: MED_COLORS.primary, letterSpacing: -0.5 },
  subTitle:    { fontSize: 15, color: MED_COLORS.muted, marginBottom: 30, fontWeight: "500" },
  input: {
    height: 52, borderWidth: 1, borderColor: "#E2E8F0", borderRadius: 16,
    paddingHorizontal: 15, marginBottom: 15, backgroundColor: "#fff", fontSize: 16,
    color: MED_COLORS.text
  },
  passWrap:    { flexDirection: "row", alignItems: "center", borderWidth: 1, borderColor: "#E2E8F0", borderRadius: 16, paddingHorizontal: 15, marginBottom: 15, backgroundColor: "#fff" },
  passInput:   { flex: 1, height: 52, fontSize: 16, color: MED_COLORS.text },
  eyeBtn:      { padding: 10 },
  btn:         { height: 56, backgroundColor: MED_COLORS.primary, borderRadius: 16, alignItems: "center", justifyContent: "center", marginTop: 15, elevation: 4, shadowColor: MED_COLORS.primary, shadowOpacity: 0.2, shadowRadius: 8 },
  btnDisabled: { opacity: 0.6 },
  btnText:     { color: "#fff", fontSize: 17, fontWeight: "700" },
  loginLink:   { marginTop: 20, alignItems: "center" },
  loginTxt:    { textAlign: "center", color: MED_COLORS.muted, fontSize: 15 },

  selectorLabelRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8, paddingHorizontal: 5 },
  selectorLabel: { fontSize: 14, fontWeight: '700', color: MED_COLORS.text },
  selectorRow: { flexDirection: 'row', gap: 12, marginBottom: 15 },
  genderWrap: { flex: 1, flexDirection: 'row', gap: 8 },
  bloodWrap: { flex: 1.5 },
  miniBtn: { flex: 1, height: 45, borderRadius: 12, borderWidth: 1, borderColor: '#E2E8F0', justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
  miniBtnActive: { backgroundColor: MED_COLORS.primary, borderColor: MED_COLORS.primary },
  miniBtnText: { fontSize: 13, color: MED_COLORS.muted, fontWeight: '600' },
  miniBtnTextActive: { color: '#fff' },
  bloodBtn: { width: 45, height: 45, borderRadius: 12, borderWidth: 1, borderColor: '#E2E8F0', justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff', marginRight: 8 },
  bloodBtnActive: { backgroundColor: MED_COLORS.primary, borderColor: MED_COLORS.primary },
  bloodBtnText: { fontSize: 12, color: MED_COLORS.muted, fontWeight: '800' },
  bloodBtnTextActive: { color: '#fff' },
});
