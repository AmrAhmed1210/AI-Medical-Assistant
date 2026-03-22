import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, Image, ScrollView, TextInput, StatusBar } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors"; 

const conversations = [
  {
    id: 1,
    name: "Dr. Eion Morgan",
    message: "Your test results look good. Let's schedule a follow-up...",
    time: "2m ago",
    image: "https://via.placeholder.com/150",
    unread: 2,
  },
  {
    id: 2,
    name: "Dr. Chloe Kelly",
    message: "Please remember to take your medication before the visit.",
    time: "1h ago",
    image: "https://via.placeholder.com/150",
    unread: 0,
  },
  {
    id: 3,
    name: "Dr. Lauren Hemp",
    message: "The MRI scan has been scheduled for next Monday.",
    time: "3h ago",
    image: "https://via.placeholder.com/150",
    unread: 1,
  },
  {
    id: 4,
    name: "Dr. James Patel",
    message: "Your cardiac report is ready for review.",
    time: "Yesterday",
    image: "https://via.placeholder.com/150",
    unread: 0,
  },
];

export default function MessagesScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Messages</Text>
          <Text style={styles.headerSubtitle}>Chat with your doctors</Text>
        </View>
        <TouchableOpacity style={styles.iconButton}>
          <Ionicons name="notifications-outline" size={24} color="#333" />
        </TouchableOpacity>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <View style={styles.searchBar}>
          <Ionicons name="search" size={20} color="#999" />
          <TextInput
            placeholder="Search conversations..."
            style={styles.searchInput}
            placeholderTextColor="#999"
          />
        </View>
      </View>

      {/* Conversations List */}
      <ScrollView style={styles.listContainer} showsVerticalScrollIndicator={false}>
        {conversations.map((convo) => (
          <TouchableOpacity 
            key={convo.id} 
            style={styles.convoItem}
            activeOpacity={0.7}
          >
            {/* Avatar Section */}
            <View style={styles.avatarContainer}>
              <Image source={{ uri: convo.image }} style={styles.avatar} />
              <View style={styles.onlineBadge} />
            </View>

            {/* Details Section - تم استبدال div بـ View هنا */}
            <View style={styles.convoDetails}>
              <View style={styles.convoHeader}>
                <Text style={styles.doctorName}>{convo.name}</Text>
                <Text style={styles.timeText}>{convo.time}</Text>
              </View>
              <Text style={styles.messageText} numberOfLines={1}>
                {convo.message}
              </Text>
            </View>

            {/* Unread Badge */}
            {convo.unread > 0 && (
              <View style={styles.unreadBadge}>
                <Text style={styles.unreadText}>{convo.unread}</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#333",
  },
  headerSubtitle: {
    fontSize: 14,
    color: "#666",
    marginTop: 2,
  },
  iconButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#f5f5f5",
    justifyContent: "center",
    alignItems: "center",
  },
  searchContainer: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f0f0f0",
    borderRadius: 15,
    paddingHorizontal: 15,
    height: 50,
  },
  searchInput: {
    flex: 1,
    marginLeft: 10,
    fontSize: 16,
    color: "#333",
  },
  listContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  convoItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  avatarContainer: {
    position: "relative",
  },
  avatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#eee",
  },
  onlineBadge: {
    position: "absolute",
    bottom: 2,
    right: 2,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: "#0D9488", // Teal Color
    borderWidth: 2,
    borderColor: "#fff",
  },
  convoDetails: {
    flex: 1,
    marginLeft: 15,
  },
  convoHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  doctorName: {
    fontSize: 16,
    fontWeight: "700",
    color: "#333",
  },
  timeText: {
    fontSize: 12,
    color: "#999",
  },
  messageText: {
    fontSize: 14,
    color: "#666",
  },
  unreadBadge: {
    backgroundColor: "#0D9488",
    width: 22,
    height: 22,
    borderRadius: 11,
    justifyContent: "center",
    alignItems: "center",
    marginLeft: 10,
  },
  unreadText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "bold",
  },
});