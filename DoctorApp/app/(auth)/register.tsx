import { useState, useRef } from "react";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
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
import Toast from "react-native-toast-message";
import { Ionicons } from "@expo/vector-icons";
import CustomButton from "../../components/CustomButton";
import { COLORS } from "../../constants/colors";

const API_URL = "http://192.168.43.216:5076/api";

export default function RegisterScreen() {
  const router = useRouter();

  const scrollViewRef = useRef<ScrollView>(null);
  const inputsRef = useRef<(TextInput | null)[]>([]);

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const focusNextInput = (index: number) => {
    inputsRef.current[index + 1]?.focus();
  };

  const handleFocus = (index: number) => {
    setTimeout(() => {
      scrollViewRef.current?.scrollTo({ y: index * 70, animated: true });
    }, 100);
  };

  const handleRegister = async () => {
    Keyboard.dismiss();

    // 1. Check empty fields
    if (!firstName.trim() || !lastName.trim() || !email.trim() || !phone.trim() || !password || !confirmPassword) {
      Toast.show({
        type: "error",
        text1: "Please fill all fields",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // 2. Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      Toast.show({
        type: "error",
        text1: "Invalid email",
        text2: "Please enter a valid email address",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // 3. Validate password length
    if (password.length < 6) {
      Toast.show({
        type: "error",
        text1: "Password too short",
        text2: "Password must be at least 6 characters",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // 4. Check passwords match
    if (password !== confirmPassword) {
      Toast.show({
        type: "error",
        text1: "Passwords don't match",
        text2: "Please make sure both passwords are the same",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/auth/register`, {
        name: `${firstName.trim()} ${lastName.trim()}`,
        email: email.toLowerCase().trim(),
        passwordHash: password,
        role: "Patient",
      });

      const data = response.data;

      if (data.token) {
        await AsyncStorage.setItem("userToken", data.token);
        await AsyncStorage.setItem("user", JSON.stringify(data));
        await AsyncStorage.setItem("isLoggedIn", "true");
        await AsyncStorage.setItem("userRole", data.role ?? "Patient");

        Toast.show({
          type: "success",
          text1: `Welcome ${data.name}! 👋`,
          text2: "Account created successfully",
          position: "top",
          topOffset: 60,
          visibilityTime: 1500,
        });

        setTimeout(() => {
          router.replace("/(patient)/home");
        }, 1600);
      }
    } catch (error: any) {
      const status = error.response?.status;
      const msg = error.response?.data;

      if (status === 409 || (typeof msg === "string" && msg.toLowerCase().includes("exist"))) {
        Toast.show({
          type: "error",
          text1: "Email already registered",
          text2: "Try logging in instead",
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
          text1: "Registration failed",
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
      keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
    >
      <ScrollView
        ref={scrollViewRef}
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Create Account</Text>

        <TextInput
          ref={(ref) => { inputsRef.current[0] = ref; }}
          placeholder="First Name"
          style={styles.input}
          value={firstName}
          onChangeText={setFirstName}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(0)}
          onFocus={() => handleFocus(0)}
          blurOnSubmit={false}
          editable={!loading}
        />

        <TextInput
          ref={(ref) => { inputsRef.current[1] = ref; }}
          placeholder="Last Name"
          style={styles.input}
          value={lastName}
          onChangeText={setLastName}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(1)}
          onFocus={() => handleFocus(1)}
          blurOnSubmit={false}
          editable={!loading}
        />

        <TextInput
          ref={(ref) => { inputsRef.current[2] = ref; }}
          placeholder="Email"
          keyboardType="email-address"
          autoCapitalize="none"
          style={styles.input}
          value={email}
          onChangeText={setEmail}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(2)}
          onFocus={() => handleFocus(2)}
          blurOnSubmit={false}
          editable={!loading}
        />

        <TextInput
          ref={(ref) => { inputsRef.current[3] = ref; }}
          placeholder="Phone Number"
          keyboardType="phone-pad"
          style={styles.input}
          value={phone}
          onChangeText={setPhone}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(3)}
          onFocus={() => handleFocus(3)}
          blurOnSubmit={false}
          editable={!loading}
        />

        {/* Password */}
        <View style={styles.passwordContainer}>
          <TextInput
            ref={(ref) => { inputsRef.current[4] = ref; }}
            placeholder="Password"
            secureTextEntry={!showPassword}
            style={styles.passwordInput}
            value={password}
            onChangeText={setPassword}
            returnKeyType="next"
            onSubmitEditing={() => focusNextInput(4)}
            onFocus={() => handleFocus(4)}
            blurOnSubmit={false}
            editable={!loading}
          />
          <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eyeIcon}>
            <Ionicons name={showPassword ? "eye-off" : "eye"} size={22} color="#888" />
          </TouchableOpacity>
        </View>

        {/* Confirm Password */}
        <View style={styles.passwordContainer}>
          <TextInput
            ref={(ref) => { inputsRef.current[5] = ref; }}
            placeholder="Confirm Password"
            secureTextEntry={!showConfirmPassword}
            style={styles.passwordInput}
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            returnKeyType="done"
            onSubmitEditing={handleRegister}
            onFocus={() => handleFocus(5)}
            editable={!loading}
          />
          <TouchableOpacity onPress={() => setShowConfirmPassword(!showConfirmPassword)} style={styles.eyeIcon}>
            <Ionicons name={showConfirmPassword ? "eye-off" : "eye"} size={22} color="#888" />
          </TouchableOpacity>
        </View>

        <View style={styles.buttonContainer}>
          {loading ? (
            <ActivityIndicator size="large" color={COLORS.primary} style={{ marginVertical: 10 }} />
          ) : (
            <CustomButton title="Create Account" onPress={handleRegister} />
          )}
        </View>

        <TouchableOpacity
          onPress={() => router.push("/(auth)/login")}
          style={styles.loginContainer}
        >
          <Text style={styles.loginText}>
            Already have an account?{" "}
            <Text style={styles.loginBold}>Login</Text>
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 25,
    paddingTop: 60,
    paddingBottom: 40,
    backgroundColor: COLORS.white,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 30,
    color: COLORS.primary,
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
    marginBottom: 15,
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
  buttonContainer: {
    marginTop: 10,
    marginBottom: 10,
  },
  loginContainer: {
    marginTop: 20,
    alignItems: "center",
  },
  loginText: {
    textAlign: "center",
    color: "#666",
    fontSize: 16,
  },
  loginBold: {
    color: COLORS.primary,
    fontWeight: "600",
  },
});
