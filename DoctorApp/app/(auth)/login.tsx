import { useState } from "react";
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, TouchableOpacity, Keyboard, ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { COLORS } from "../../constants/colors";
import { loginApi, saveSession } from "../../services/authService";

export default function LoginScreen() {
  const router = useRouter();
  const [email,        setEmail]        = useState("");
  const [password,     setPassword]     = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading,      setLoading]      = useState(false);

  const handleLogin = async () => {
    Keyboard.dismiss();

    if (!email.trim() || !password.trim()) {
      Toast.show({ type: "error", text1: "Please fill all fields", position: "top", topOffset: 60 });
      return;
    }

    setLoading(true);
    try {
      console.log('🚀 Starting login...');
      const auth = await loginApi({
        email:    email.toLowerCase().trim(),
        password: password,
      });
      console.log('✅ Login successful, got token');

      await saveSession(auth);
      console.log('✅ Session saved, preparing to navigate...');

      Toast.show({
        type: "success",
        text1: `Welcome back ${auth.name}! 👋`,
        position: "top",
        topOffset: 60,
        visibilityTime: 1500,
      });

      setTimeout(() => {
        const targetRoute = auth.role === "Doctor" ? "/(doctor)" : "/(patient)/home";
        console.log('🔄 Navigating to:', targetRoute);
        router.replace(targetRoute);
      }, 1600);

    } catch (err: any) {
      Toast.show({
        type: "error",
        text1: err.message || "Login failed",
        position: "top",
        topOffset: 60,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: COLORS.white }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Welcome Back! 👋</Text>
        <Text style={styles.subtitle}>Login to your account</Text>

        <View style={styles.inputContainer}>
          <TextInput
            placeholder="Email" style={styles.input}
            value={email} onChangeText={setEmail}
            keyboardType="email-address" autoCapitalize="none" editable={!loading}
          />
          <View style={styles.passWrap}>
            <TextInput
              placeholder="Password" secureTextEntry={!showPassword}
              style={styles.passInput} value={password} onChangeText={setPassword}
              editable={!loading} onSubmitEditing={handleLogin} returnKeyType="done"
            />
            <TouchableOpacity onPress={() => setShowPassword(p => !p)} style={styles.eyeBtn}>
              <Text>{showPassword ? "👁️" : "👁️‍🗨️"}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.btn, loading && styles.btnDisabled]}
          onPress={handleLogin}
          disabled={loading}
          activeOpacity={0.8}
        >
          {loading
            ? <ActivityIndicator size="small" color="#fff" />
            : <Text style={styles.btnText}>Login</Text>
          }
        </TouchableOpacity>

        <View style={styles.regRow}>
          <Text style={styles.regTxt}>Don&apos;t have an account? </Text>
          <TouchableOpacity onPress={() => router.push("/(auth)/register")}>
            <Text style={styles.regLink}>Create Account</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container:      { flexGrow: 1, padding: 25, justifyContent: "center", backgroundColor: COLORS.white },
  title:          { fontSize: 32, fontWeight: "bold", marginBottom: 10, color: COLORS.primary },
  subtitle:       { fontSize: 16, color: "#666", marginBottom: 40 },
  inputContainer: { marginBottom: 30 },
  input: {
    height: 52, borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 14,
    paddingHorizontal: 15, marginBottom: 15, backgroundColor: "#fafafa", fontSize: 16,
  },
  passWrap:    { flexDirection: "row", alignItems: "center", borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 14, paddingHorizontal: 15, backgroundColor: "#fafafa" },
  passInput:   { flex: 1, height: 52, fontSize: 16 },
  eyeBtn:      { padding: 10 },
  btn:         { height: 52, backgroundColor: COLORS.primary, borderRadius: 14, alignItems: "center", justifyContent: "center" },
  btnDisabled: { opacity: 0.6 },
  btnText:     { color: "#fff", fontSize: 16, fontWeight: "600" },
  regRow:      { flexDirection: "row", justifyContent: "center", marginTop: 20 },
  regTxt:      { color: "#666", fontSize: 16 },
  regLink:     { color: COLORS.primary, fontSize: 16, fontWeight: "600" },
});