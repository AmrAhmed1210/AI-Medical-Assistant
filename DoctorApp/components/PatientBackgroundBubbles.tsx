import React from "react";
import { StyleSheet, View } from "react-native";

type Props = {
  isDark?: boolean;
};

export default function PatientBackgroundBubbles({ isDark = false }: Props) {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <View
        style={[
          styles.bubble,
          styles.topLeft,
          { backgroundColor: isDark ? "rgba(16, 185, 129, 0.16)" : "rgba(16, 185, 129, 0.09)" },
        ]}
      />
      <View
        style={[
          styles.bubble,
          styles.bottomRight,
          { backgroundColor: isDark ? "rgba(14, 165, 233, 0.15)" : "rgba(14, 165, 233, 0.08)" },
        ]}
      />
      <View
        style={[
          styles.bubble,
          styles.center,
          { backgroundColor: isDark ? "rgba(20, 184, 166, 0.10)" : "rgba(20, 184, 166, 0.06)" },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  bubble: {
    position: "absolute",
    borderRadius: 999,
  },
  topLeft: {
    width: 320,
    height: 320,
    top: -120,
    left: -130,
  },
  bottomRight: {
    width: 380,
    height: 380,
    right: -170,
    bottom: -150,
  },
  center: {
    width: 230,
    height: 230,
    top: "38%",
    left: -110,
  },
});
