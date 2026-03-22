import { View, Text, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import CustomButton from "../../components/CustomButton";
import { COLORS } from "../../constants/colors";

export default function Index() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      
      {/* Top Gradient Section */}
      <LinearGradient
        colors={["#1E9E84", "#159C7C"]}
        style={styles.topSection}
      >
        <View style={styles.iconContainer}>
          <Ionicons name="chatbubble-ellipses" size={40} color={COLORS.primary} />
        </View>

        <Text style={styles.title}>MedBook</Text>

        <Text style={styles.subtitle}>
          Book doctors easily & manage your health smartly
        </Text>

        <View style={styles.aiBadge}>
          <Ionicons name="sparkles" size={16} color="#fff" />
          <Text style={styles.aiText}>AI Chatbot Integrated</Text>
        </View>
      </LinearGradient>

      {/* Bottom Section */}
      <View style={styles.bottomSection}>
        <CustomButton
          title="Login"
          onPress={() => router.push("/(auth)/login")}
        />

        <CustomButton
          title="Create Account"
          onPress={() => router.push("/(auth)/register")}
          outline
        />
      </View>

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.white,
  },

  topSection: {
    height: "55%",
    justifyContent: "center",
    alignItems: "center",
    padding: 30,
    borderBottomLeftRadius: 50,
    borderBottomRightRadius: 50,
  },

  iconContainer: {
    backgroundColor: COLORS.white,
    padding: 20,
    borderRadius: 50,
    marginBottom: 20,
    elevation: 5,
  },

  title: {
    fontSize: 38,
    fontWeight: "bold",
    color: COLORS.white,
  },

  subtitle: {
    fontSize: 15,
    color: COLORS.white,
    marginTop: 10,
    textAlign: "center",
    opacity: 0.9,
  },

  aiBadge: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 15,
    backgroundColor: "rgba(255,255,255,0.2)",
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
  },

  aiText: {
    color: "#fff",
    marginLeft: 6,
    fontSize: 13,
    fontWeight: "600",
  },

  bottomSection: {
    flex: 1,
    padding: 30,
    justifyContent: "center",
  },
});