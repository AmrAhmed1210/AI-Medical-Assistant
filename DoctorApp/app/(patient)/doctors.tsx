import { View, Text, StyleSheet } from "react-native";
import { COLORS } from "../../constants/colors";

export default function DoctorsScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Find Doctors</Text>
      <Text style={styles.subtitle}>Search for doctors...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
});