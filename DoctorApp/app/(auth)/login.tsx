import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TouchableOpacity,
  Keyboard,
  ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Toast from "react-native-toast-message";
import CustomButton from "../../components/CustomButton";
import { COLORS } from "../../constants/colors";
import axios from "axios";

const API_URL = "http://192.168.43.216:5076/api";

export default function LoginScreen() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    Keyboard.dismiss();

    // 1. Validation
    if (!email.trim() || !password.trim()) {
      Toast.show({
        type: "error",
        text1: "Please fill all fields",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    setLoading(true);

    try {
      // 2. Call API
      const response = await axios.post(`${API_URL}/auth/login`, {
        email: email.toLowerCase().trim(),
        passwordHash: password,
      });

      const data = response.data;

      if (data.token) {
        // 3. Save token and user data
        await AsyncStorage.setItem("userToken", data.token);
        await AsyncStorage.setItem("user", JSON.stringify(data));
        await AsyncStorage.setItem("isLoggedIn", "true");
        await AsyncStorage.setItem("userRole", data.role ?? "Patient");

        Toast.show({
          type: "success",
          text1: `Welcome back ${data.name ?? ""}! 👋`,
          position: "top",
          topOffset: 60,
          visibilityTime: 1500,
        });

        // 4. Navigate based on role
        setTimeout(() => {
          if (data.role === "Doctor") {
            router.replace("/(doctor)");
          } else {
          
            router.replace("/(patient)/home");
          }
        }, 1600);
      }
    } catch (error: any) {
      const status = error.response?.status;
      const msg = error.response?.data;

      if (status === 401 || status === 400) {
        Toast.show({
          type: "error",
          text1: "Invalid credentials",
          text2: "Email or password is incorrect",
          position: "top",
          topOffset: 60,
        });
      } else if (!error.response) {
        Toast.show({
          type: "error",
          text1: "Cannot reach server",
          text2: "Check your connection and try again",
          position: "top",
          topOffset: 60,
        });
      } else {
        Toast.show({
          type: "error",
          text1: "Login failed",
          text2: typeof msg === "string" ? msg : "Something went wrong",
          position: "top",
          topOffset: 60,
        });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: COLORS.white }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Welcome Back! 👋</Text>
        <Text style={styles.subtitle}>Login to your account</Text>

        <View style={styles.inputContainer}>
          <TextInput
            placeholder="Email"
            style={styles.input}
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            editable={!loading}
          />

          <View style={styles.passwordContainer}>
            <TextInput
              placeholder="Password"
              secureTextEntry={!showPassword}
              style={styles.passwordInput}
              value={password}
              onChangeText={setPassword}
              editable={!loading}
              onSubmitEditing={handleLogin}
              returnKeyType="done"
            />
            <TouchableOpacity
              onPress={() => setShowPassword(!showPassword)}
              style={styles.eyeIcon}
            >
              <Text>{showPassword ? "👁️" : "👁️‍🗨️"}</Text>
            </TouchableOpacity>
          </View>
        </View>

        {loading ? (
          <ActivityIndicator
            size="large"
            color={COLORS.primary}
            style={{ marginVertical: 10 }}
          />
        ) : (
          <CustomButton title="Login" onPress={handleLogin} />
        )}

        <View style={styles.registerContainer}>
          <Text style={styles.registerText}>Don't have an account? </Text>
          <TouchableOpacity onPress={() => router.push("/(auth)/register")}>
            <Text style={styles.registerLink}>Create Account</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 25,
    justifyContent: "center",
    backgroundColor: COLORS.white,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    marginBottom: 10,
    color: COLORS.primary,
  },
  subtitle: {
    fontSize: 16,
    color: "#666",
    marginBottom: 40,
  },
  inputContainer: {
    marginBottom: 30,
  },
  input: {
    height: 52,
    borderWidth: 1,
    borderColor: "#e5e5e5",
    borderRadius: 14,
    paddingHorizontal: 15,
    marginBottom: 15,
    backgroundColor: "#fafafa",
    fontSize: 16,
  },
  passwordContainer: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e5e5e5",
    borderRadius: 14,
    paddingHorizontal: 15,
    backgroundColor: "#fafafa",
  },
  passwordInput: {
    flex: 1,
    height: 52,
    fontSize: 16,
  },
  eyeIcon: {
    padding: 10,
  },
  registerContainer: {
    flexDirection: "row",
    justifyContent: "center",
    marginTop: 20,
  },
  registerText: {
    color: "#666",
    fontSize: 16,
  },
  registerLink: {
    color: COLORS.primary,
    fontSize: 16,
    fontWeight: "600",
  },
});