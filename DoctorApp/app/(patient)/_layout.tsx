import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { View, StyleSheet } from "react-native";

type TabIconProps = {
  color: string;
  size: number;
  focused: boolean;
};

function TabIcon({ name, color, size, focused }: { name: any; color: string; size: number; focused: boolean }) {
  return (
    <View style={[styles.iconWrap, focused && styles.iconWrapActive]}>
      <Ionicons name={name} size={focused ? size + 1 : size} color={color} />
    </View>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: "#BBBBC0",
        tabBarStyle: {
          height: 70,
          paddingBottom: 10,
          paddingTop: 6,
          backgroundColor: "#fff",
          borderTopWidth: 0,
          shadowColor: "#000",
          shadowOffset: { width: 0, height: -4 },
          shadowOpacity: 0.08,
          shadowRadius: 12,
          elevation: 12,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: "600",
          marginTop: 2,
        },
      }}
    >
      <Tabs.Screen
        name="home"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "home" : "home-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="doctors"
        options={{
          title: "Find",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "search" : "search-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="messages"
        options={{
          title: "Messages",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "chatbubbles" : "chatbubbles-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="chatbot"
        options={{
          title: "AI Bot",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "medical" : "medical-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      <Tabs.Screen
        name="profile"
        options={{
          title: "Profile",
          tabBarIcon: ({ color, size, focused }: TabIconProps) => (
            <TabIcon name={focused ? "person" : "person-outline"} color={color} size={size} focused={focused} />
          ),
        }}
      />

      {/* Hidden screens - مش هتظهر في الـ tab bar */}
      <Tabs.Screen
        name="doctor-details"
        options={{
          href: null, // يخفيها من الـ tab bar تماماً
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  iconWrap: {
    width: 36,
    height: 30,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 10,
  },
  iconWrapActive: {
    backgroundColor: COLORS.primary + "18",
  },
});