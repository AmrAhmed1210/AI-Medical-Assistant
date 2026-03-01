import { useState, useRef } from "react";
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
  ScrollView as ScrollViewType,
} from "react-native";
import { useRouter } from "expo-router";
import Toast from "react-native-toast-message";
import { Ionicons } from "@expo/vector-icons";
import CustomButton from "../../components/CustomButton";
import { COLORS } from "../../constants/colors";

export default function RegisterScreen() {
  const router = useRouter();
  
  // Refs with proper typing
  const scrollViewRef = useRef<ScrollViewType>(null);
  const inputsRef = useRef<(TextInput | null)[]>([]);

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const existingEmails = ["test@mail.com", "admin@medbook.com"];

  // Function to focus next input
  const focusNextInput = (index: number) => {
    if (index < inputsRef.current.length - 1) {
      inputsRef.current[index + 1]?.focus();
    }
  };

  // Function to scroll to focused input
  const handleFocus = (index: number) => {
    setTimeout(() => {
      scrollViewRef.current?.scrollTo({
        y: index * 70, // Adjust based on input height
        animated: true,
      });
    }, 100);
  };

  const handleRegister = async () => {
    Keyboard.dismiss(); // Dismiss keyboard when registering

    // Check for empty fields
    if (!firstName || !lastName || !email || !phone || !password || !confirmPassword) {
      Toast.show({
        type: "error",
        text1: "Please fill all fields",
        text2: "All fields are required to create an account",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // Check if email already exists
    if (existingEmails.includes(email.toLowerCase())) {
      Toast.show({
        type: "error",
        text1: "Account already exists",
        text2: "Please login instead",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // Check if passwords match
    if (password !== confirmPassword) {
      Toast.show({
        type: "error",
        text1: "Passwords do not match",
        text2: "Make sure both passwords are identical",
        position: "top",
        topOffset: 60,
      });
      return;
    }

    // Check password length
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

    // Save user data
    const user = {
      firstName,
      lastName,
      email,
      phone,
    };

    try {
      await AsyncStorage.setItem("user", JSON.stringify(user));
      await AsyncStorage.setItem("isLoggedIn", "true");
      await AsyncStorage.setItem("userName", firstName);

      Toast.show({
        type: "success",
        text1: `Welcome ${firstName}! 👋`,
        text2: "Your account has been created successfully",
        position: "top",
        topOffset: 60,
        visibilityTime: 2500,
      });

      setTimeout(() => {
        router.replace("/home");
      }, 2600);
    } catch (error) {
      Toast.show({
        type: "error",
        text1: "Something went wrong",
        text2: "Please try again",
        position: "top",
        topOffset: 60,
      });
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
          ref={ref => { inputsRef.current[0] = ref; }}
          placeholder="First Name"
          style={styles.input}
          value={firstName}
          onChangeText={setFirstName}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(0)}
          onFocus={() => handleFocus(0)}
          blurOnSubmit={false}
        />

        <TextInput
          ref={ref => { inputsRef.current[1] = ref; }}
          placeholder="Last Name"
          style={styles.input}
          value={lastName}
          onChangeText={setLastName}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(1)}
          onFocus={() => handleFocus(1)}
          blurOnSubmit={false}
        />

        <TextInput
          ref={ref => { inputsRef.current[2] = ref; }}
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
        />

        <TextInput
          ref={ref => { inputsRef.current[3] = ref; }}
          placeholder="Phone Number"
          keyboardType="phone-pad"
          style={styles.input}
          value={phone}
          onChangeText={setPhone}
          returnKeyType="next"
          onSubmitEditing={() => focusNextInput(3)}
          onFocus={() => handleFocus(3)}
          blurOnSubmit={false}
        />

        {/* Password */}
        <View style={styles.passwordContainer}>
          <TextInput
            ref={ref => { inputsRef.current[4] = ref; }}
            placeholder="Password"
            secureTextEntry={!showPassword}
            style={styles.passwordInput}
            value={password}
            onChangeText={setPassword}
            returnKeyType="next"
            onSubmitEditing={() => focusNextInput(4)}
            onFocus={() => handleFocus(4)}
            blurOnSubmit={false}
          />
          <TouchableOpacity
            onPress={() => setShowPassword(!showPassword)}
            style={styles.eyeIcon}
          >
            <Ionicons
              name={showPassword ? "eye-off" : "eye"}
              size={22}
              color="#888"
            />
          </TouchableOpacity>
        </View>

        {/* Confirm Password */}
        <View style={styles.passwordContainer}>
          <TextInput
            ref={ref => { inputsRef.current[5] = ref; }}
            placeholder="Confirm Password"
            secureTextEntry={!showConfirmPassword}
            style={styles.passwordInput}
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            returnKeyType="done"
            onSubmitEditing={handleRegister}
            onFocus={() => handleFocus(5)}
          />
          <TouchableOpacity
            onPress={() => setShowConfirmPassword(!showConfirmPassword)}
            style={styles.eyeIcon}
          >
            <Ionicons
              name={showConfirmPassword ? "eye-off" : "eye"}
              size={22}
              color="#888"
            />
          </TouchableOpacity>
        </View>

        <View style={styles.buttonContainer}>
          <CustomButton
            title="Create Account"
            onPress={handleRegister}
          />
        </View>

        <TouchableOpacity
          onPress={() => router.push("/(auth)/login")}
          style={styles.loginContainer}
        >
          <Text style={styles.loginText}>
            Already have an account? <Text style={styles.loginBold}>Login</Text>
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