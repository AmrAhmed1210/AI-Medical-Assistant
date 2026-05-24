import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Modal,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Image,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect, useLocalSearchParams } from "expo-router";
import { COLORS } from "../../constants/colors";
import PatientBackgroundBubbles from "@/components/PatientBackgroundBubbles";
import { useTheme } from "../../context/ThemeContext";
import { useLanguage } from "../../context/LanguageContext";
import { getDoctorById } from "../../services/doctorService";
import { getMyProfile, Profile } from "../../services/profileService";
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { uploadMessageFile } from "../../services/fileService";
import {
  getMySessions,
  getSessionDetails,
  parseDoctorIdFromSessionTitle,
  sendSessionMessage,
  SessionDetails,
  SessionItem,
  SessionMessage,
  startSession,
  deleteSession,
} from "../../services/sessionService";
import { useNotificationStore } from "../../store/notificationStore";

const getParam = (value: string | string[] | undefined): string => {
  if (Array.isArray(value)) return value[0] ?? "";
  return value ?? "";
};

const formatTimeAgo = (value?: string | null): string => {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  const diffMs = Date.now() - parsed.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return "Yesterday";
  return `${diffDays}d ago`;
};

const getSessionDate = (session: SessionItem): string => {
  return session.updatedAt || session.createdAt || "";
};

export default function MessagesScreen() {
  const { theme, isDark, colors } = useTheme();
  const { tr, isRTL } = useLanguage();
  const params = useLocalSearchParams<{
    doctorId?: string | string[];
    doctorName?: string | string[];
    initialMessage?: string | string[];
  }>();

  const targetDoctorId = Number(getParam(params.doctorId));
  const targetDoctorName = getParam(params.doctorName).trim();
  const initialMessage = getParam(params.initialMessage).trim();

  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [doctorDataBySession, setDoctorDataBySession] = useState<Record<number, { name: string, photoUrl?: string }>>({});
  const [lastMessageBySession, setLastMessageBySession] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [search, setSearch] = useState("");

  const [chatVisible, setChatVisible] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [selectedSession, setSelectedSession] = useState<SessionDetails | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [sending, setSending] = useState(false);
  const [openingSession, setOpeningSession] = useState(false);
  const [profile, setProfile] = useState<Profile | null>(null);

  const { unreadCounts, clearSessionMessages, incrementSessionMessage, latestMessagePayload } = useNotificationStore();

  const selectedSessionIdRef = useRef<number | null>(null);
  useEffect(() => {
    selectedSessionIdRef.current = selectedSessionId;
  }, [selectedSessionId]);

  const chatVisibleRef = useRef<boolean>(false);
  useEffect(() => {
    chatVisibleRef.current = chatVisible;
  }, [chatVisible]);

  useEffect(() => {
    if (!latestMessagePayload) return;
    const payload = latestMessagePayload;
    const sessionId = payload?.sessionId ?? payload?.SessionId;
    if (!sessionId) return;

    const content = payload?.message ?? payload?.Message ?? '';
    const timestamp = payload?.timestamp ?? new Date().toISOString();
    const doctorName = payload?.doctorName ?? payload?.DoctorName ?? 'Doctor';

    if (String(selectedSessionIdRef.current) === String(sessionId) && chatVisibleRef.current) {
      const msg: SessionMessage = {
        id: -Date.now(),
        sessionId: Number(sessionId),
        role: "doctor",
        content,
        timestamp,
        senderName: doctorName,
        messageType: "text",
      };
      setSelectedSession(prev => prev ? { ...prev, messages: [...(prev.messages || []), msg] } : prev);
      clearSessionMessages(sessionId);
    } else {
      incrementSessionMessage(sessionId);
    }

    setLastMessageBySession(prev => ({ ...prev, [sessionId]: content }));
    setSessions(prev => {
      const idx = prev.findIndex(s => s.id == sessionId);
      if (idx !== -1) {
        const updated = { ...prev[idx], updatedAt: timestamp, lastMessage: content };
        const copy = [...prev];
        copy.splice(idx, 1);
        return [updated, ...copy];
      } else {
        // New session not in list - add it
        const newSession: SessionItem = {
          id: Number(sessionId),
          userId: 0, // Not strictly needed for UI list
          title: `chat|d:${payload.doctorId || 0}|`,
          createdAt: timestamp,
          updatedAt: timestamp,
          lastMessage: content,
        };
        return [newSession, ...prev];
      }
    });

  }, [latestMessagePayload, clearSessionMessages, incrementSessionMessage]);

  const handledDoctorIdRef = useRef<number | null>(null);

  const getDisplayDoctorData = useCallback(
    (session: SessionItem): { name: string, photoUrl?: any } => {
      if (session.type === 'SupportChat' || session.title?.includes('Support')) {
        return { name: "الشكاوي والدعم", photoUrl: require("../../assets/images/support_avatar.png") };
      }
      const direct = doctorDataBySession[session.id];
      if (direct) return direct;
      const docId = parseDoctorIdFromSessionTitle(session.title);
      if (docId && Number.isFinite(targetDoctorId) && docId === targetDoctorId && targetDoctorName) {
        return { name: targetDoctorName };
      }
      return { name: docId ? `Doctor #${docId}` : "Doctor" };
    },
    [doctorDataBySession, targetDoctorId, targetDoctorName]
  );

  const loadSessionDetails = useCallback(async (sessionId: number) => {
    setOpeningSession(true);
    try {
      const details = await getSessionDetails(sessionId);
      setSelectedSession(details);
      setSelectedSessionId(sessionId);
      setChatVisible(true);
      clearSessionMessages(sessionId);
      const latest = details.messages?.[details.messages.length - 1]?.content ?? "";
      if (latest) {
        setLastMessageBySession((prev) => ({ ...prev, [sessionId]: latest }));
      }
    } finally {
      setOpeningSession(false);
    }
  }, []);

  const loadSessions = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true);
    else setLoading(true);

    try {
      const data = await getMySessions();
      const ordered = [...data].sort(
        (a, b) => new Date(getSessionDate(b)).getTime() - new Date(getSessionDate(a)).getTime()
      );
      setSessions(ordered);

      setLastMessageBySession(
        Object.fromEntries(ordered.map(s => [s.id, s.lastMessage ?? ""]))
      );

      const doctorIds = Array.from(
        new Set(
          ordered
            .map((session) => parseDoctorIdFromSessionTitle(session.title))
            .filter((id): id is number => Number.isFinite(id as number) && Number(id) > 0)
        )
      );

      const doctorPairs = await Promise.all(
        doctorIds.map(async (doctorId) => {
          try {
            const doc = await getDoctorById(doctorId);
            return [doctorId, { name: String(doc.name ?? `Doctor #${doctorId}`), photoUrl: doc.imageUrl || doc.photoUrl }] as const;
          } catch {
            return [doctorId, { name: `Doctor #${doctorId}` }] as const;
          }
        })
      );

      const byDoctorId = Object.fromEntries(doctorPairs);
      const bySessionId: Record<number, { name: string, photoUrl?: string }> = {};
      ordered.forEach((session) => {
        const doctorId = parseDoctorIdFromSessionTitle(session.title);
        if (doctorId && byDoctorId[doctorId]) {
          bySessionId[session.id] = byDoctorId[doctorId];
        }
      });

      setDoctorDataBySession(bySessionId);
    } finally {
      if (showRefreshing) setRefreshing(false);
      else setLoading(false);
    }
  }, []);

  const ensureDirectSession = useCallback(async () => {
    if (!Number.isFinite(targetDoctorId) || targetDoctorId <= 0) return;

    // If we already handled this doctor in this "navigation session", skip UNLESS there is a new initialMessage
    if (handledDoctorIdRef.current === targetDoctorId && !initialMessage) return;
    handledDoctorIdRef.current = targetDoctorId;

    const existing = sessions.find(
      (session) => parseDoctorIdFromSessionTitle(session.title) === targetDoctorId
    );

    if (existing) {
      if (initialMessage) {
        try {
          const saved = await sendSessionMessage(existing.id, initialMessage);
          setSelectedSession((prev) => {
            if (!prev || String(prev.id) !== String(existing.id)) return prev;
            const alreadyHas = (prev.messages || []).some(m => m.id === saved.id);
            if (alreadyHas) return prev;
            return { ...prev, messages: [...(prev.messages || []), saved] };
          });
        } catch {
          // ignore error
        }
      }
      await loadSessionDetails(existing.id);
      setSelectedSessionId(existing.id);
      setChatVisible(true);
      await loadSessions(true);
      return;
    }

    const created = await startSession(targetDoctorId, initialMessage || undefined);
    await loadSessions(true);
    await loadSessionDetails(created.id);
    setSelectedSessionId(created.id);
    setChatVisible(true);
  }, [initialMessage, loadSessionDetails, loadSessions, sessions, targetDoctorId]);

  useEffect(() => {
    loadSessions().catch(() => undefined);
    fetchProfile();
  }, [loadSessions]);

  const fetchProfile = async () => {
    try {
      const data = await getMyProfile();
      setProfile(data);
    } catch { }
  };

  const initialMessageSentRef = useRef(false);

  useEffect(() => {
    if (loading) return;
    if (initialMessage && initialMessageSentRef.current) return;

    ensureDirectSession().then(() => {
      if (initialMessage) initialMessageSentRef.current = true;
    }).catch(() => undefined);
  }, [ensureDirectSession, loading, initialMessage]);

  useFocusEffect(
    useCallback(() => {
      fetchProfile();
      loadSessions(true).catch(() => undefined);
    }, [loadSessions])
  );

  const filteredSessions = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return sessions;

    return sessions.filter((session) => {
      const docData = getDisplayDoctorData(session);
      const doctorName = docData.name.toLowerCase();
      const preview = String(lastMessageBySession[session.id] ?? "").toLowerCase();
      return doctorName.includes(term) || preview.includes(term);
    });
  }, [getDisplayDoctorData, lastMessageBySession, search, sessions]);

  const handleOpenConversation = async (session: SessionItem) => {
    await loadSessionDetails(session.id);
  };

  const handlePickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0]) {
      await handleSendMedia(result.assets[0].uri, 'image');
    }
  };

  const handlePickDocument = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: '*/*',
    });

    if (!result.canceled && result.assets[0]) {
      await handleSendMedia(result.assets[0].uri, 'file', result.assets[0].name);
    }
  };

  const handleSendMedia = async (uri: string, type: 'image' | 'file', name?: string) => {
    if (!selectedSessionId || sending) return;
    setSending(true);

    try {
      const { url, fileName } = await uploadMessageFile(uri, name);
      const saved = await sendSessionMessage(selectedSessionId, type === 'image' ? '[Image]' : `[File: ${fileName}]`, type, url, fileName);

      setSelectedSession((prev) => {
        if (!prev) return prev;
        return { ...prev, messages: [...(prev.messages || []), saved] };
      });

      setLastMessageBySession((prev) => ({ ...prev, [selectedSessionId]: type === 'image' ? 'Sent an image' : 'Sent a file' }));
      await loadSessions(true);
    } catch (e: any) {
      Alert.alert("Error", "Failed to upload file");
    } finally {
      setSending(false);
    }
  };

  const handleSendMessage = async () => {
    if (!selectedSessionId || !chatInput.trim() || sending) return;
    const content = chatInput.trim();
    setChatInput("");
    setSending(true);

    const tempMessage: SessionMessage = {
      id: -Date.now(),
      sessionId: selectedSessionId,
      role: "user",
      content,
      timestamp: new Date().toISOString(),
      messageType: "text",
    };

    setSelectedSession((prev) =>
      prev ? { ...prev, messages: [...(prev.messages || []), tempMessage] } : prev
    );
    setLastMessageBySession((prev) => ({ ...prev, [selectedSessionId]: content }));

    try {
      const saved = await sendSessionMessage(selectedSessionId, content);
      setSelectedSession((prev) => {
        if (!prev) return prev;
        // Remove all temporary messages (negative IDs) and append the saved one
        const confirmed = (prev.messages || []).filter((m) => Number(m.id) > 0);
        return { ...prev, messages: [...confirmed, saved] };
      });

      // Update session list preview immediately
      setSessions(prev => {
        const idx = prev.findIndex(s => s.id == selectedSessionId);
        if (idx !== -1) {
          const updated = { ...prev[idx], updatedAt: saved.timestamp, lastMessage: content };
          const copy = [...prev];
          copy.splice(idx, 1);
          return [updated, ...copy];
        }
        return prev;
      });

      // Then refresh from server to stay in sync
      await loadSessions(true);
    } catch (e: any) {
      Alert.alert("Error", e.message || "Failed to send message");
      setSelectedSession((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          messages: (prev.messages || []).filter((item) => item.id !== tempMessage.id),
        };
      });
      setChatInput(content);
    } finally {
      setSending(false);
    }
  };

  const handleDeleteChat = () => {
    if (!selectedSessionId) return;
    Alert.alert(
      "Delete Conversation",
      "Are you sure you want to delete this chat? This action cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              await deleteSession(selectedSessionId);
              setChatVisible(false);
              setSelectedSessionId(null);
              setSelectedSession(null);
              loadSessions(true);
            } catch (e: any) {
              Alert.alert("Error", e.message || "Failed to delete conversation");
            }
          }
        }
      ]
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor={isDark ? colors.background : "#fff"} />
      <PatientBackgroundBubbles isDark={isDark} />

      <View style={styles.magicHeader}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={StyleSheet.absoluteFill}>
          <View style={styles.headerTop}>
            <View>
              <Text style={styles.headerTitle}>Messages</Text>
              <Text style={styles.headerSubtitle}>Seamless care with your team</Text>
            </View>
            <TouchableOpacity style={styles.headerAction}>
              <Ionicons name="ellipsis-horizontal" size={20} color="rgba(255,255,255,0.8)" />
            </TouchableOpacity>
          </View>
          <View style={styles.searchBar}>
            <Ionicons name="search" size={18} color="rgba(255,255,255,0.7)" />
            <TextInput
              placeholder="Search conversations..."
              style={styles.searchInput}
              placeholderTextColor="rgba(255,255,255,0.5)"
              value={search}
              onChangeText={setSearch}
            />
          </View>
          <View style={[styles.liquidBlob, { top: -30, right: -30, width: 180, height: 180, backgroundColor: '#10B981', opacity: 0.1 }]} />
          <View style={[styles.liquidBlob, { bottom: -20, left: -20, width: 140, height: 140, backgroundColor: '#34D399', opacity: 0.1 }]} />
        </LinearGradient>
      </View>

      <ScrollView
        style={styles.listContainer}
        showsVerticalScrollIndicator={false}
        decelerationRate="fast"
        bounces={true}
      >
        {loading ? (
          <View style={styles.emptyWrap}>
            <ActivityIndicator color={COLORS.primary} />
          </View>
        ) : filteredSessions.length === 0 ? (
          <View style={styles.emptyWrap}>
            <Ionicons name="chatbubble-ellipses-outline" size={34} color={isDark ? "#4B5563" : "#9CA3AF"} />
            <Text style={[styles.emptyTitle, { color: colors.text }]}>No conversations yet</Text>
            <Text style={[styles.emptySub, { color: colors.textMuted }]}>Start a conversation from doctor profile.</Text>
          </View>
        ) : (
          filteredSessions.map((session) => {
            const docData = getDisplayDoctorData(session);
            const preview = lastMessageBySession[session.id] || "Tap to open conversation";
            const lastAt = getSessionDate(session);
            const unread = unreadCounts[session.id] || 0;

            return (
              <TouchableOpacity
                key={session.id}
                style={[styles.convoItem, { backgroundColor: colors.surface, borderColor: colors.border }]}
                activeOpacity={0.7}
                onPress={() => {
                  handleOpenConversation(session).catch(() => undefined);
                }}
              >
                <View style={styles.avatarContainer}>
                  <View style={styles.avatarFallback}>
                    {docData.photoUrl ? (
                      <Image source={typeof docData.photoUrl === 'string' ? { uri: docData.photoUrl } : docData.photoUrl} style={styles.avatarImg} />
                    ) : (
                      <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.avatarImg} />
                    )}
                  </View>
                  <View style={styles.onlineBadge} />
                </View>

                <View style={styles.convoDetails}>
                  <View style={styles.convoHeader}>
                    <Text style={[styles.doctorName, { color: colors.text }]} numberOfLines={1}>{docData.name}</Text>
                    <Text style={styles.timeText}>{formatTimeAgo(lastAt)}</Text>
                  </View>
                  <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Text style={[styles.messageText, { color: colors.textMuted }]} numberOfLines={1}>
                      {preview}
                    </Text>
                    {unread > 0 && (
                      <View style={styles.unreadBadge}>
                        <Text style={styles.unreadText}>{unread}</Text>
                      </View>
                    )}
                  </View>
                </View>
              </TouchableOpacity>
            );
          })
        )}
      </ScrollView>

      <Modal visible={chatVisible} animationType="slide" onRequestClose={() => setChatVisible(false)}>
        <KeyboardAvoidingView
          style={[styles.chatContainer, { backgroundColor: colors.background }]}
          behavior={Platform.OS === "ios" ? "padding" : undefined}
        >
          <PatientBackgroundBubbles isDark={isDark} />
          <View style={[styles.chatHeader, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
            <TouchableOpacity onPress={() => setChatVisible(false)} style={styles.backBtn}>
              <Ionicons name="chevron-back" size={22} color={colors.text} />
            </TouchableOpacity>
            <View style={{ flex: 1 }}>
              <Text style={[styles.chatTitle, { color: colors.text }]}>
                {selectedSession
                  ? getDisplayDoctorData(selectedSession).name
                  : targetDoctorName || "Conversation"}
              </Text>
              {openingSession && <Text style={styles.chatSub}>Loading...</Text>}
            </View>
            <TouchableOpacity onPress={handleDeleteChat} style={styles.deleteBtn}>
              <Ionicons name="trash-outline" size={20} color="#EF4444" />
            </TouchableOpacity>
          </View>

          <ScrollView
            style={styles.chatMessages}
            contentContainerStyle={{ paddingVertical: 15 }}
            decelerationRate="fast"
            bounces={true}
            maintainVisibleContentPosition={{ minIndexForVisible: 0 }}
          >
            {(selectedSession?.messages || []).map((msg) => {
              const isMine = msg.role?.toLowerCase() === "user" || msg.role?.toLowerCase() === "patient";
              const senderPhoto = isMine ? profile?.photoUrl : msg.senderPhotoUrl || getDisplayDoctorData(selectedSession!).photoUrl;

              return (
                <View key={String(msg.id)} style={[styles.bubbleRow, isMine ? styles.bubbleRowMine : styles.bubbleRowDoctor]}>
                  {!isMine && (
                    <View style={styles.bubbleAvatar}>
                      {senderPhoto ? (
                        <Image source={typeof senderPhoto === 'string' ? { uri: senderPhoto } : senderPhoto} style={styles.bubbleAvatarImg} />
                      ) : (
                        <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.bubbleAvatarImg} />
                      )}
                    </View>
                  )}

                  <LinearGradient
                    colors={isMine ? [COLORS.primary, "#047857"] : (isDark ? ["#1E293B", "#0F172A"] : ["#fff", "#F8FAFC"])}
                    style={[styles.bubble, isMine ? styles.bubbleMine : styles.bubbleDoctor, !isMine && { borderColor: colors.border }]}
                  >
                    <Text style={[styles.senderName, isMine ? styles.senderNameMine : [styles.senderNameDoctor, { color: colors.text }]]}>
                      {msg.senderName || msg.role}
                    </Text>
                    {msg.messageType === 'image' && msg.attachmentUrl ? (
                      <Image
                        source={{ uri: msg.attachmentUrl }}
                        style={styles.messageImage}
                        resizeMode="cover"
                      />
                    ) : msg.messageType === 'file' ? (
                      <View style={styles.fileContainer}>
                        <Ionicons name="document-text" size={24} color={isMine ? "#fff" : COLORS.primary} />
                        <View style={styles.fileInfo}>
                          <Text style={[styles.fileName, isMine ? styles.fileNameMine : styles.fileNameDoctor]} numberOfLines={1}>
                            {msg.fileName || 'Document'}
                          </Text>
                          <Text style={[styles.fileType, isMine ? styles.fileTypeMine : styles.fileTypeDoctor]}>File Attachment</Text>
                        </View>
                      </View>
                    ) : (
                      <Text style={[styles.bubbleTxt, isMine ? styles.bubbleTxtMine : [styles.bubbleTxtDoctor, { color: colors.text }]]}>
                        {msg.content}
                      </Text>
                    )}
                    <Text style={[styles.bubbleTime, isMine ? styles.bubbleTimeMine : styles.bubbleTimeDoctor]}>
                      {formatTimeAgo(msg.timestamp)}
                    </Text>
                  </LinearGradient>

                  {isMine && (
                    <View style={styles.bubbleAvatar}>
                      {senderPhoto ? (
                        <Image source={{ uri: senderPhoto }} style={styles.bubbleAvatarImg} />
                      ) : (
                        <View style={[styles.avatarMini, { backgroundColor: COLORS.primary + '20' }]}>
                          <Ionicons name="person" size={10} color={COLORS.primary} />
                        </View>
                      )}
                    </View>
                  )}
                </View>
              );
            })}
          </ScrollView>

          <View style={[styles.chatComposer, { backgroundColor: colors.surface, borderTopColor: colors.border }]}>
            <View style={styles.composerActions}>
              <TouchableOpacity style={styles.attachBtn} onPress={handlePickImage}>
                <LinearGradient colors={isDark ? ["#1E293B", "#0F172A"] : ["#F1F5F9", "#E2E8F0"]} style={[styles.attachIconWrap, { borderColor: colors.border }]}>
                  <Ionicons name="image" size={18} color={isDark ? "#9CA3AF" : "#475569"} />
                </LinearGradient>
              </TouchableOpacity>
              <TouchableOpacity style={styles.attachBtn} onPress={handlePickDocument}>
                <LinearGradient colors={isDark ? ["#1E293B", "#0F172A"] : ["#F1F5F9", "#E2E8F0"]} style={[styles.attachIconWrap, { borderColor: colors.border }]}>
                  <Ionicons name="document-attach" size={18} color={isDark ? "#9CA3AF" : "#475569"} />
                </LinearGradient>
              </TouchableOpacity>
            </View>
            <View style={[styles.inputWrapper, { backgroundColor: isDark ? "#0F172A" : "#F8FAFC", borderColor: colors.border }]}>
              <TextInput
                value={chatInput}
                onChangeText={setChatInput}
                placeholder="Type a message..."
                placeholderTextColor={isDark ? "#6B7280" : "#9CA3AF"}
                style={[styles.chatInput, { color: colors.text }]}
                multiline
              />
              <TouchableOpacity
                style={[styles.sendBtn, (!chatInput.trim() || sending) && styles.sendBtnDisabled]}
                disabled={!chatInput.trim() || sending}
                onPress={handleSendMessage}
              >
                <LinearGradient colors={["#059669", "#064E3B"]} style={styles.sendIconWrap}>
                  {sending ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Ionicons name="send" size={16} color="#fff" />
                  )}
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  magicHeader: { height: 220, borderBottomLeftRadius: 40, borderBottomRightRadius: 40, overflow: 'hidden', elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.2, shadowRadius: 20 },
  headerTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 20, paddingTop: 60, marginBottom: 20, zIndex: 10 },
  headerTitle: { fontSize: 24, fontWeight: "800", color: "#fff", letterSpacing: -0.5 },
  headerSubtitle: { fontSize: 13, color: "rgba(255,255,255,0.8)", marginTop: 2 },
  headerAction: { width: 40, height: 40, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: "center", alignItems: "center", borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  searchBar: { flexDirection: "row", alignItems: "center", backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 18, paddingHorizontal: 16, height: 52, marginHorizontal: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)' },
  searchInput: { flex: 1, marginLeft: 10, fontSize: 15, color: "#fff", fontWeight: '600' },
  liquidBlob: { position: 'absolute', borderRadius: 100 },
  listContainer: { flex: 1 },
  convoItem: { flexDirection: "row", alignItems: "center", paddingVertical: 16, paddingHorizontal: 16, backgroundColor: "#fff", borderBottomWidth: 1, borderBottomColor: "#F8FAFC" },
  avatarContainer: { position: "relative" },
  avatarFallback: { width: 56, height: 56, borderRadius: 24, backgroundColor: "#ECFDF5", alignItems: "center", justifyContent: "center", overflow: "hidden" },
  avatarImg: { width: "100%", height: "100%", borderRadius: 24 },
  onlineBadge: { position: "absolute", bottom: 0, right: 0, width: 14, height: 14, borderRadius: 7, backgroundColor: "#10B981", borderWidth: 3, borderColor: "#fff" },
  convoDetails: { flex: 1, marginLeft: 15 },
  convoHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 6 },
  doctorName: { fontSize: 15, fontWeight: "700", color: "#1E293B" },
  timeText: { fontSize: 11, color: "#94A3B8" },
  messageText: { fontSize: 13, color: "#64748B", flex: 1, marginRight: 8 },
  unreadBadge: { backgroundColor: "#059669", minWidth: 20, height: 20, borderRadius: 10, justifyContent: "center", alignItems: "center", paddingHorizontal: 6 },
  unreadText: { color: "#fff", fontSize: 10, fontWeight: "800" },
  emptyWrap: { alignItems: "center", justifyContent: "center", paddingVertical: 100, gap: 15 },
  emptyTitle: { fontSize: 18, fontWeight: "700", color: "#475569" },
  emptySub: { fontSize: 14, color: "#94A3B8", textAlign: "center", paddingHorizontal: 50 },

  chatContainer: { flex: 1, backgroundColor: "#F8FAFC" },
  chatHeader: { paddingTop: 60, paddingBottom: 15, paddingHorizontal: 16, backgroundColor: "#fff", borderBottomWidth: 1, borderBottomColor: "#F1F5F9", flexDirection: "row", alignItems: "center", gap: 12 },
  backBtn: { width: 40, height: 40, borderRadius: 14, backgroundColor: "#F1F5F9", alignItems: "center", justifyContent: "center" },
  chatTitle: { fontSize: 17, fontWeight: "700", color: "#1E293B" },
  chatSub: { fontSize: 11, color: "#10B981", fontWeight: "600", marginTop: 1 },
  deleteBtn: { width: 40, height: 40, borderRadius: 14, backgroundColor: "#FEF2F2", alignItems: "center", justifyContent: "center" },
  chatMessages: { flex: 1, paddingHorizontal: 16 },
  bubbleRow: { flexDirection: "row", marginVertical: 8 },
  bubbleRowMine: { justifyContent: "flex-end" },
  bubbleRowDoctor: { justifyContent: "flex-start" },
  bubble: { maxWidth: "80%", borderRadius: 24, paddingHorizontal: 18, paddingVertical: 12, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3 },
  bubbleMine: { borderBottomRightRadius: 4 },
  bubbleDoctor: { borderBottomLeftRadius: 4 },
  bubbleTxt: { fontSize: 14, lineHeight: 20 },
  bubbleTxtMine: { color: "#fff" },
  bubbleTxtDoctor: { color: "#1E293B" },
  senderName: { fontSize: 11, fontWeight: "800", marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 },
  bubbleAvatar: { width: 34, height: 34, borderRadius: 12, marginHorizontal: 8, alignSelf: "flex-end", overflow: "hidden" },
  bubbleAvatarImg: { width: "100%", height: "100%", borderRadius: 12 },
  avatarMini: { width: "100%", height: "100%", borderRadius: 12, alignItems: "center", justifyContent: "center" },
  senderNameMine: { color: "rgba(255,255,255,0.6)" },
  senderNameDoctor: { color: "#94A3B8" },
  bubbleTime: { fontSize: 10, marginTop: 6, textAlign: 'right' },
  bubbleTimeMine: { color: "rgba(255,255,255,0.6)" },
  bubbleTimeDoctor: { color: "#94A3B8" },
  chatComposer: { paddingHorizontal: 16, paddingVertical: 15, backgroundColor: "#fff", borderTopWidth: 1, borderTopColor: "#F1F5F9" },
  composerActions: { flexDirection: "row", gap: 10, marginBottom: 12 },
  attachBtn: { flex: 1, padding: 4 },
  attachIconWrap: { height: 40, borderRadius: 12, justifyContent: "center", alignItems: "center" },
  inputWrapper: { flexDirection: "row", alignItems: "center", gap: 10, backgroundColor: "#F1F5F9", borderRadius: 20, padding: 6 },
  chatInput: { flex: 1, paddingHorizontal: 12, paddingVertical: 8, fontSize: 14, color: "#1E293B", maxHeight: 100 },
  sendBtn: { width: 44, height: 44 },
  sendIconWrap: { width: "100%", height: "100%", borderRadius: 16, justifyContent: "center", alignItems: "center" },
  sendBtnDisabled: { opacity: 0.5 },
  messageImage: {
    width: 200,
    height: 150,
    borderRadius: 12,
    marginVertical: 4,
  },
  fileContainer: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 4 },
  fileInfo: { flex: 1 },
  fileName: { fontSize: 13, fontWeight: '600' },
  fileNameMine: { color: '#fff' },
  fileNameDoctor: { color: '#1E293B' },
  fileType: { fontSize: 11 },
  fileTypeMine: { color: 'rgba(255,255,255,0.7)' },
  fileTypeDoctor: { color: '#94A3B8' },
});
