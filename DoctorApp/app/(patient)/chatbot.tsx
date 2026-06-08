import React, { useState, useEffect, useRef } from "react";
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  StatusBar, 
  TextInput, 
  FlatList, 
  KeyboardAvoidingView, 
  Platform,
  ActivityIndicator,
  Keyboard,
  Modal
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons, Feather } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { chatService, ChatMessage } from "../../services/chatService";
import Toast from "react-native-toast-message";
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import { useTheme } from "../../context/ThemeContext";
import { getRecommendedDoctorsForNeed, formatDoctorRecommendationsForAi } from "../../services/doctorService";
import AsyncStorage from "@react-native-async-storage/async-storage";

type ChatUiMessage = ChatMessage & { suggestedDoctors?: any[]; suggestedSpecialty?: string };
type LocalChatSession = {
  id: number;
  title: string;
  updatedAt: string;
  messages: ChatUiMessage[];
};

const CHAT_SESSIONS_KEY = "ai_chatbot_saved_sessions_v1";
const CURRENT_CHAT_SESSION_KEY = "ai_chatbot_current_session_v1";

export default function ChatBotScreen() {
  const router = useRouter();
  const { theme, isDark, colors } = useTheme();
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<number | undefined>(undefined);
  const [savedSessions, setSavedSessions] = useState<LocalChatSession[]>([]);
  const [historyVisible, setHistoryVisible] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  // AI medical chat is stateless on the server — no session restore needed.

  useEffect(() => {
    restoreSavedChats().catch(() => undefined);
  }, []);

  const restoreSavedChats = async () => {
    const raw = await AsyncStorage.getItem(CHAT_SESSIONS_KEY);
    const sessions: LocalChatSession[] = raw ? JSON.parse(raw) : [];
    const sorted = sessions.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    setSavedSessions(sorted);

    const currentRaw = await AsyncStorage.getItem(CURRENT_CHAT_SESSION_KEY);
    const currentId = currentRaw ? Number(currentRaw) : sorted[0]?.id;
    const current = sorted.find((session) => session.id === currentId) ?? sorted[0];
    if (current) {
      setCurrentSessionId(current.id);
      setMessages(current.messages);
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: false }), 100);
    }
  };

  const persistLocalChat = async (sessionId: number, nextMessages: ChatUiMessage[]) => {
    if (nextMessages.length === 0) return;
    const firstUserMessage = nextMessages.find((message) => message.role === "user")?.content ?? "Medical chat";
    const session: LocalChatSession = {
      id: sessionId,
      title: firstUserMessage.slice(0, 48),
      updatedAt: new Date().toISOString(),
      messages: nextMessages,
    };

    const nextSessions = [session, ...savedSessions.filter((item) => item.id !== sessionId)].slice(0, 20);
    setSavedSessions(nextSessions);
    await AsyncStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(nextSessions));
    await AsyncStorage.setItem(CURRENT_CHAT_SESSION_KEY, String(sessionId));
  };

  const openSavedSession = async (session: LocalChatSession) => {
    setCurrentSessionId(session.id);
    setMessages(session.messages);
    setHistoryVisible(false);
    await AsyncStorage.setItem(CURRENT_CHAT_SESSION_KEY, String(session.id));
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: false }), 100);
  };

  const loadMessages = async (sessionId: number) => {
    try {
      const msgs = await chatService.getMessages(sessionId);
      setMessages(msgs);
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    } catch (error) {
      console.error("Load messages error:", error);
    }
  };

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMsg = inputText.trim();
    setInputText("");
    Keyboard.dismiss();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    const localSessionId = currentSessionId ?? Date.now();
    if (!currentSessionId) setCurrentSessionId(localSessionId);

    const tempUserMsg: ChatUiMessage = {
      id: Date.now(),
      sessionId: localSessionId,
      role: "user",
      content: userMsg,
      timestamp: new Date().toISOString()
    };
    
    const optimisticMessages = [...messages, tempUserMsg];
    setMessages(optimisticMessages);
    persistLocalChat(localSessionId, optimisticMessages).catch(() => undefined);
    setIsLoading(true);

    try {
      const doctorRecommendations = await getRecommendedDoctorsForNeed(userMsg, 5).catch(() => ({ specialty: null, doctors: [] }));
      
      let contextPrompt = userMsg;
      if (doctorRecommendations.specialty && doctorRecommendations.doctors.length > 0) {
        // Case 1: Specialty inferred AND matching doctors exist → recommend them
        contextPrompt = `${userMsg}\n\n[Platform doctor recommendations context - use only if relevant]\nRecommended specialty: ${doctorRecommendations.specialty}\nAvailable doctors from our platform sorted by highest reviews/rating:\n${JSON.stringify(formatDoctorRecommendationsForAi(doctorRecommendations.doctors))}\nIf the patient needs a specialist, recommend these doctors by name and specialty.`;
      } else if (doctorRecommendations.specialty && doctorRecommendations.doctors.length === 0) {
        // Case 2: Specialty inferred BUT no matching doctors on platform → tell the user what specialty they need
        contextPrompt = `${userMsg}\n\n[Context: Based on the patient's message, the recommended medical specialty is "${doctorRecommendations.specialty}". However, there are currently no ${doctorRecommendations.specialty} doctors registered on our platform. Tell the patient that they should look for a "${doctorRecommendations.specialty}" specialist. Do NOT recommend any specific doctor names.]`;
      }
      // Case 3: No specialty inferred → just send the raw message, no doctor context

      const history = messages.slice(-8).map((m) => ({
        id: m.id,
        sessionId: m.sessionId,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
      }));
      const response = await chatService.ask(contextPrompt, currentSessionId, history);
      
      const assistantMsg: ChatUiMessage = {
        id: Date.now() + 1,
        sessionId: localSessionId,
        role: "assistant",
        content: response.reply,
        timestamp: new Date().toISOString(),
        suggestedDoctors: doctorRecommendations.doctors && doctorRecommendations.doctors.length > 0 ? doctorRecommendations.doctors : undefined,
        suggestedSpecialty: doctorRecommendations.specialty && doctorRecommendations.doctors.length === 0 ? doctorRecommendations.specialty : undefined,
      };

      const savedMessages = [...optimisticMessages, assistantMsg];
      setMessages(savedMessages);
      persistLocalChat(localSessionId, savedMessages).catch(() => undefined);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      Toast.show({
        type: "error",
        text1: "Error",
        text2: "Failed to get AI response."
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleImagePick = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0].uri) {
        uploadImage(result.assets[0].uri);
      }
    } catch (error) {
      Toast.show({ type: "error", text1: "Error selecting image" });
    }
  };

  const uploadImage = async (uri: string) => {
    setIsLoading(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    const localSessionId = currentSessionId ?? Date.now();
    if (!currentSessionId) setCurrentSessionId(localSessionId);

    // Optimistic image message
    const tempMsg: ChatUiMessage = {
      id: Date.now(),
      sessionId: localSessionId,
      role: "user",
      content: "[Analyzing image...]",
      timestamp: new Date().toISOString()
    };
    const optimisticMessages = [...messages, tempMsg];
    setMessages(optimisticMessages);
    persistLocalChat(localSessionId, optimisticMessages).catch(() => undefined);

    try {
      const result = await chatService.analyzeImage(uri, currentSessionId);
      const res = result as any;
      
      const analysisText = `${res.analysis_ar}\n\n[التفاصيل التقنية]:\n${res.technical_details}\n\n${res.disclaimer}`;
      
      const assistantMsg: ChatUiMessage = {
        id: Date.now() + 1,
        sessionId: localSessionId,
        role: "assistant",
        content: analysisText,
        timestamp: new Date().toISOString()
      };

      const savedMessages = [...optimisticMessages.filter(m => m.id !== tempMsg.id), 
        { ...tempMsg, content: "[Image sent]" },
        assistantMsg
      ];
      setMessages(savedMessages);
      persistLocalChat(localSessionId, savedMessages).catch(() => undefined);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      Toast.show({ type: "error", text1: "Analysis failed" });
      setMessages(prev => prev.filter(m => m.id !== tempMsg.id));
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: ChatUiMessage }) => {
    const isUser = item.role === "user";
    return (
      <View style={[styles.messageWrapper, isUser ? styles.userWrapper : styles.aiWrapper]}>
        {!isUser && (
          <View style={styles.aiAvatar}>
            <Ionicons name="sparkles" size={16} color="#fff" />
          </View>
        )}
        <View style={[styles.messageBubble, isUser ? styles.userBubble : [styles.aiBubble, { backgroundColor: colors.surface, borderColor: colors.border }]]}>
          <Text style={[styles.messageText, isUser ? styles.userText : [styles.aiText, { color: colors.text }]]}>
            {item.content}
          </Text>
          {item.suggestedDoctors && item.suggestedDoctors.length > 0 && (
            <View style={{ marginTop: 12, paddingTop: 8, borderTopWidth: 1, borderTopColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' }}>
              <Text style={{ fontSize: 12, color: colors.textMuted, marginBottom: 8, fontWeight: '700' }}>Recommended Specialists:</Text>
              {item.suggestedDoctors.map((doc: any, index: number) => (
                <TouchableOpacity key={index} style={[styles.suggestedDoctorCard, { backgroundColor: isDark ? 'rgba(16,185,129,0.1)' : '#F0FDF4', borderColor: isDark ? 'rgba(16,185,129,0.2)' : '#DCFCE7' }]} onPress={() => router.push({ pathname: "/(patient)/doctor-details", params: { id: doc.id } } as any)}>
                  <Ionicons name="medkit" size={20} color="#059669" style={{ marginRight: 8 }} />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.suggestedDoctorName}>Dr. {doc.name}</Text>
                    <Text style={styles.suggestedDoctorSpec}>{doc.specialty} • ⭐ {doc.rating || 'New'}</Text>
                  </View>
                  <Ionicons name="chevron-forward" size={16} color="#059669" />
                </TouchableOpacity>
              ))}
            </View>
          )}
          {item.suggestedSpecialty && !item.suggestedDoctors?.length && (
            <TouchableOpacity
              onPress={() => router.push({ pathname: "/(patient)/doctors", params: { specialty: item.suggestedSpecialty } } as any)}
              style={{ marginTop: 12, paddingTop: 8, paddingBottom: 4, borderTopWidth: 1, borderTopColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)', flexDirection: 'row', alignItems: 'center', gap: 8 }}
            >
              <View style={{ width: 32, height: 32, borderRadius: 10, backgroundColor: isDark ? 'rgba(234,179,8,0.15)' : '#FFFBEB', justifyContent: 'center', alignItems: 'center' }}>
                <Ionicons name="search" size={16} color="#D97706" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={{ fontSize: 12, color: '#D97706', fontWeight: '700' }}>التخصص المناسب: {item.suggestedSpecialty}</Text>
                <Text style={{ fontSize: 11, color: colors.textMuted }}>ابحث عن أطباء في هذا التخصص ←</Text>
              </View>
              <Ionicons name="chevron-forward" size={16} color="#D97706" />
            </TouchableOpacity>
          )}
          <Text style={[styles.timestamp, isUser ? styles.userTimestamp : styles.aiTimestamp]}>
            {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </Text>
        </View>
      </View>
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor="transparent" />
      
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        <TouchableOpacity onPress={() => router.back()} style={[styles.backButton, { backgroundColor: isDark ? "#0F172A" : "#F1F5F9" }]}>
          <Ionicons name="chevron-back" size={24} color={colors.text} />
        </TouchableOpacity>
        <View style={styles.headerInfo}>
          <Text style={[styles.headerTitle, { color: colors.text }]}>AI Assistant</Text>
          <View style={styles.statusDot} />
          <Text style={styles.statusText}>Online</Text>
        </View>
        <View style={styles.headerActions}>
          <TouchableOpacity 
            style={styles.newChatButton}
            onPress={() => setHistoryVisible(true)}
          >
            <Feather name="clock" size={23} color={COLORS.primary} />
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.newChatButton}
            onPress={() => {
              setCurrentSessionId(undefined);
              setMessages([]);
              AsyncStorage.removeItem(CURRENT_CHAT_SESSION_KEY).catch(() => undefined);
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            }}
          >
            <Feather name="plus-square" size={24} color={COLORS.primary} />
          </TouchableOpacity>
        </View>
      </View>

      <Modal visible={historyVisible} transparent animationType="fade">
        <View style={styles.historyOverlay}>
          <View style={[styles.historySheet, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.historyHeader}>
              <Text style={[styles.historyTitle, { color: colors.text }]}>Saved Chats</Text>
              <TouchableOpacity onPress={() => setHistoryVisible(false)}>
                <Ionicons name="close" size={22} color={colors.textMuted} />
              </TouchableOpacity>
            </View>
            <FlatList
              data={savedSessions}
              keyExtractor={(item) => String(item.id)}
              ListEmptyComponent={<Text style={styles.historyEmpty}>No old chats yet.</Text>}
              renderItem={({ item }) => (
                <TouchableOpacity style={[styles.historyItem, { borderColor: colors.border }]} onPress={() => openSavedSession(item)}>
                  <Text style={[styles.historyItemTitle, { color: colors.text }]} numberOfLines={1}>{item.title}</Text>
                  <Text style={styles.historyItemMeta}>{new Date(item.updatedAt).toLocaleString()}</Text>
                </TouchableOpacity>
              )}
            />
          </View>
        </View>
      </Modal>

      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderMessage}
        contentContainerStyle={styles.listContent}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <View style={styles.emptyIconContainer}>
              <Ionicons name="chatbubbles-outline" size={48} color={COLORS.primary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.text }]}>How can I help you today?</Text>
            <Text style={styles.emptySubtitle}>
              Ask me about your symptoms, medications,{'\n'}
              or general health advice.
            </Text>
          </View>
        }
      />

      {isLoading && (
        <View style={styles.typingIndicator}>
          <ActivityIndicator size="small" color={COLORS.primary} />
          <Text style={styles.typingText}>AI is thinking...</Text>
        </View>
      )}

      <KeyboardAvoidingView 
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
      >
        <View style={[styles.inputContainer, { backgroundColor: colors.surface, borderTopColor: colors.border }]}>
          <View style={[styles.inputWrapper, { backgroundColor: isDark ? "#0F172A" : "#F1F5F9" }]}>
            <TouchableOpacity 
              style={styles.attachButton}
              onPress={handleImagePick}
              disabled={isLoading}
            >
              <Ionicons name="attach" size={24} color={isDark ? "#94A3B8" : "#64748B"} />
            </TouchableOpacity>
            <TextInput
              style={[styles.input, { color: colors.text }]}
              placeholder="Type your message..."
              placeholderTextColor={isDark ? "#4B5563" : "#94A3B8"}
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={500}
            />
            <TouchableOpacity 
              style={[styles.sendButton, !inputText.trim() && { backgroundColor: isDark ? "#334155" : "#CBD5E1" }]}
              onPress={handleSend}
              disabled={!inputText.trim() || isLoading}
            >
              <Ionicons name="send" size={20} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8FAFC' },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 50,
    paddingBottom: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#E2E8F0',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
  },
  backButton: { width: 40, height: 40, borderRadius: 12, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F1F5F9' },
  headerInfo: { flex: 1, alignItems: 'center' },
  headerTitle: { fontSize: 18, fontWeight: '700', color: '#0F172A' },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#10B981', marginVertical: 4 },
  statusText: { fontSize: 12, color: '#64748B', fontWeight: '500' },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  newChatButton: { width: 40, height: 40, justifyContent: 'center', alignItems: 'center' },
  historyOverlay: { flex: 1, backgroundColor: 'rgba(15,23,42,0.45)', justifyContent: 'center', padding: 22 },
  historySheet: { maxHeight: '70%', borderRadius: 24, padding: 18, borderWidth: 1 },
  historyHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  historyTitle: { fontSize: 18, fontWeight: '900' },
  historyItem: { borderWidth: 1, borderRadius: 16, padding: 12, marginBottom: 10 },
  historyItemTitle: { fontSize: 14, fontWeight: '800' },
  historyItemMeta: { color: '#94A3B8', fontSize: 11, marginTop: 4, fontWeight: '600' },
  historyEmpty: { color: '#94A3B8', textAlign: 'center', paddingVertical: 24, fontWeight: '700' },
  
  listContent: { padding: 16, paddingBottom: 24 },
  messageWrapper: { flexDirection: 'row', marginBottom: 16, maxWidth: '85%' },
  userWrapper: { alignSelf: 'flex-end' },
  aiWrapper: { alignSelf: 'flex-start' },
  aiAvatar: { 
    width: 28, height: 28, borderRadius: 14, 
    backgroundColor: COLORS.primary, 
    justifyContent: 'center', alignItems: 'center',
    marginRight: 8, marginTop: 4 
  },
  messageBubble: { padding: 12, borderRadius: 16 },
  userBubble: { backgroundColor: COLORS.primary, borderBottomRightRadius: 4 },
  aiBubble: { backgroundColor: '#fff', borderBottomLeftRadius: 4, borderWidth: 1, borderColor: '#E2E8F0' },
  messageText: { fontSize: 15, lineHeight: 22 },
  userText: { color: '#fff' },
  aiText: { color: '#1E293B' },
  timestamp: { fontSize: 10, marginTop: 4, alignSelf: 'flex-end' },
  userTimestamp: { color: 'rgba(255,255,255,0.7)' },
  aiTimestamp: { color: '#94A3B8' },
  
  suggestedDoctorCard: { flexDirection: 'row', alignItems: 'center', padding: 10, borderRadius: 12, marginBottom: 6, borderWidth: 1 },
  suggestedDoctorName: { color: '#059669', fontWeight: 'bold', fontSize: 13 },
  suggestedDoctorSpec: { color: '#059669', fontSize: 11, opacity: 0.8, marginTop: 2 },

  emptyContainer: { flex: 1, alignItems: 'center', marginTop: 100 },
  emptyIconContainer: { width: 80, height: 80, borderRadius: 40, backgroundColor: COLORS.primary + '10', justifyContent: 'center', alignItems: 'center', marginBottom: 20 },
  emptyTitle: { fontSize: 20, fontWeight: '700', color: '#0F172A', marginBottom: 12 },
  emptySubtitle: { fontSize: 15, color: '#64748B', textAlign: 'center', lineHeight: 22 },

  typingIndicator: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingBottom: 8 },
  typingText: { fontSize: 13, color: '#64748B', marginLeft: 8, fontStyle: 'italic' },

  inputContainer: { padding: 12, backgroundColor: '#fff', borderTopWidth: 1, borderTopColor: '#E2E8F0', paddingBottom: Platform.OS === 'ios' ? 30 : 12 },
  inputWrapper: { flexDirection: 'row', alignItems: 'flex-end', backgroundColor: '#F1F5F9', borderRadius: 24, paddingHorizontal: 12, paddingVertical: 8 },
  attachButton: { width: 36, height: 36, justifyContent: 'center', alignItems: 'center', marginRight: 4, marginBottom: 2 },
  input: { flex: 1, minHeight: 40, maxHeight: 120, fontSize: 16, color: '#1E293B', paddingHorizontal: 8, paddingTop: 8 },
  sendButton: { width: 36, height: 36, borderRadius: 18, backgroundColor: COLORS.primary, justifyContent: 'center', alignItems: 'center', marginLeft: 8, marginBottom: 2 },
  sendButtonDisabled: { backgroundColor: '#CBD5E1' }
});
