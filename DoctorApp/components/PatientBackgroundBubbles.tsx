import React from "react";
import { StyleSheet, Animated, View } from "react-native";

type Props = {
  isDark?: boolean;
  scrollY?: Animated.Value | Animated.AnimatedInterpolation<number>;
};

export default function PatientBackgroundBubbles({ isDark = false, scrollY }: Props) {
  // Continuous gentle floating animation when scrollY is not provided
  const floatAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    if (!scrollY) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(floatAnim, {
            toValue: 1,
            duration: 6000,
            useNativeDriver: true,
          }),
          Animated.timing(floatAnim, {
            toValue: 0,
            duration: 6000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [scrollY]);

  // Interpolations for smooth movement
  // If scrollY is passed, we move them dynamically based on scroll offset.
  // Otherwise, they float gently.
  const topLeftTranslateY = scrollY
    ? scrollY.interpolate({
        inputRange: [0, 500],
        outputRange: [0, 100],
        extrapolate: "clamp",
      })
    : floatAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [0, 15],
      });

  const bottomRightTranslateY = scrollY
    ? scrollY.interpolate({
        inputRange: [0, 500],
        outputRange: [0, -120],
        extrapolate: "clamp",
      })
    : floatAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [0, -20],
      });

  const centerTranslateY = scrollY
    ? scrollY.interpolate({
        inputRange: [0, 500],
        outputRange: [0, -60],
        extrapolate: "clamp",
      })
    : floatAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [0, -10],
      });

  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Animated.View
        style={[
          styles.bubble,
          styles.topLeft,
          {
            backgroundColor: isDark ? "rgba(16, 185, 129, 0.16)" : "rgba(16, 185, 129, 0.09)",
            transform: [{ translateY: topLeftTranslateY }],
          },
        ]}
      />
      <Animated.View
        style={[
          styles.bubble,
          styles.bottomRight,
          {
            backgroundColor: isDark ? "rgba(14, 165, 233, 0.15)" : "rgba(14, 165, 233, 0.08)",
            transform: [{ translateY: bottomRightTranslateY }],
          },
        ]}
      />
      <Animated.View
        style={[
          styles.bubble,
          styles.center,
          {
            backgroundColor: isDark ? "rgba(20, 184, 166, 0.10)" : "rgba(20, 184, 166, 0.06)",
            transform: [{ translateY: centerTranslateY }],
          },
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
