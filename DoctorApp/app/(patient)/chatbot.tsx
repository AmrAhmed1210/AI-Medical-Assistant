import { View, Text, StyleSheet, TouchableOpacity, StatusBar } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import Toast from "react-native-toast-message";

export default function ChatBotScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="chevron-back" size={24} color="#1E293B" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>AI Assistant</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Coming Soon Content */}
      <View style={styles.content}>
        <View style={styles.iconContainer}>
          <Ionicons name="chatbubble-ellipses-outline" size={80} color={COLORS.primary} />
        </View>
        
        <Text style={styles.title}>Coming Soon!</Text>
        
        <Text style={styles.subtitle}>
          Our AI Assistant is under development{'\n'}
          and will be available in the next update.
        </Text>

        <View style={styles.featuresContainer}>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={24} color={COLORS.primary} />
            <Text style={styles.featureText}>24/7 Medical Assistance</Text>
          </View>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={24} color={COLORS.primary} />
            <Text style={styles.featureText}>Symptom Checker</Text>
          </View>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={24} color={COLORS.primary} />
            <Text style={styles.featureText}>Appointment Booking</Text>
          </View>
        </View>

        <TouchableOpacity 
          style={styles.notifyButton}
          onPress={() => {
            Toast.show({
              type: "info",
              text1: "We'll notify you when ready!",
              position: "top",
              topOffset: 60,
            });
          }}
        >
          <Text style={styles.notifyButtonText}>Notify Me</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F7FA' },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 12,
    backgroundColor: '#F1F5F9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: { fontSize: 17, fontWeight: '600', color: '#1E293B' },
  content: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 24 },
  iconContainer: {
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: COLORS.primary + '10',
    justifyContent: 'center', alignItems: 'center',
    marginBottom: 24,
  },
  title: { fontSize: 24, fontWeight: '700', color: COLORS.primary, marginBottom: 12 },
  subtitle: {
    fontSize: 15, color: '#64748B', textAlign: 'center', lineHeight: 22, marginBottom: 32,
  },
  featuresContainer: { alignSelf: 'stretch', marginBottom: 32, backgroundColor: '#fff', padding: 20, borderRadius: 20, borderWidth: 1, borderColor: '#F1F5F9' },
  featureItem: { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  featureText: { fontSize: 14, color: '#1E293B', marginLeft: 12, fontWeight: '500' },
  notifyButton: { backgroundColor: COLORS.primary, paddingVertical: 14, borderRadius: 12, width: '100%', alignItems: 'center', shadowColor: COLORS.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.2, shadowRadius: 8, elevation: 4 },
  notifyButtonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});