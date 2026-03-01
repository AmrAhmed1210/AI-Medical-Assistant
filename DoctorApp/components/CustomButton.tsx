import { TouchableOpacity, Text, StyleSheet } from "react-native";
import { COLORS } from "../constants/colors";

interface Props {
  title: string;
  onPress: () => void;
  outline?: boolean;
}

export default function CustomButton({ title, onPress, outline }: Props) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[
        styles.button,
        outline && {
          backgroundColor: "transparent",
          borderWidth: 2,
          borderColor: COLORS.primary,
        },
      ]}
    >
      <Text
        style={[
          styles.text,
          outline && { color: COLORS.primary },
        ]}
      >
        {title}
      </Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: COLORS.primary,
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 15,
  },
  text: {
    color: COLORS.white,
    fontWeight: "bold",
    fontSize: 16,
  },
});