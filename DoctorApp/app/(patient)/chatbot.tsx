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
  Keyboard
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons, Feather } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { chatService, ChatMessage } from "../../services/chatService";
import Toast from "react-native-toast-message";
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import { useTheme } from "../../context/ThemeContext";

export default function ChatBotScreen() {
  const router = useRouter();
  const { theme, isDark, colors } = useTheme();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<number | undefined>(undefined);
  const flatListRef = useRef<FlatList>(null);

  // AI medical chat is stateless on the server — no session restore needed.

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

    const tempUserMsg: ChatMessage = {
      id: Date.now(),
      sessionId: currentSessionId || 0,
      role: "user",
      content: userMsg,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, tempUserMsg]);
    setIsLoading(true);

    try {
      const response = await chatService.ask(userMsg, currentSessionId);
      
      if (!currentSessionId) {
        setCurrentSessionId(response.sessionId);
      }

      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        sessionId: response.sessionId,
        role: "assistant",
        content: response.reply,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMsg]);
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

    // Optimistic image message
    const tempMsg: ChatMessage = {
      id: Date.now(),
      sessionId: currentSessionId || 0,
      role: "user",
      content: "[Analyzing image...]",
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, tempMsg]);

    try {
      const result = await chatService.analyzeImage(uri, currentSessionId);
      const res = result as any;
      
      const analysisText = `${res.analysis_ar}\n\n[التفاصيل التقنية]:\n${res.technical_details}\n\n${res.disclaimer}`;
      
      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        sessionId: currentSessionId || 0,
        role: "assistant",
        content: analysisText,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev.filter(m => m.id !== tempMsg.id), 
        { ...tempMsg, content: "[Image sent]" },
        assistantMsg
      ]);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      Toast.show({ type: "error", text1: "Analysis failed" });
      setMessages(prev => prev.filter(m => m.id !== tempMsg.id));
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: ChatMessage }) => {
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
        <TouchableOpacity 
          style={styles.newChatButton}
          onPress={() => {
            setCurrentSessionId(undefined);
            setMessages([]);
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          }}
        >
          <Feather name="plus-square" size={24} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

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
  newChatButton: { width: 40, height: 40, justifyContent: 'center', alignItems: 'center' },
  
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
