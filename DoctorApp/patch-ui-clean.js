const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'app', '(patient)', 'ai-profile-assistant.tsx');
let content = fs.readFileSync(filePath, 'utf8');

const renderStart = content.indexOf('  /* ──────────────── RENDER ──────────────── */');
if (renderStart === -1) { console.error('Could not find render section'); process.exit(1); }

const before = content.substring(0, renderStart);

const newRender = `  /* ──────────────── RENDER ──────────────── */
  const renderSummaryCard = () => {
    if (!diseases.length && !meds.length && !allergies.length && !surgeries.length && !vitals.length) return null;
    const items = [
      { icon: "heart", count: diseases.length, label: t("أمراض", "Diseases") },
      { icon: "medkit", count: meds.length, label: t("أدوية", "Meds") },
      { icon: "alert-circle", count: allergies.length, label: t("حساسية", "Allergies") },
      { icon: "cut", count: surgeries.length, label: t("جراحات", "Surgeries") },
      { icon: "pulse", count: vitals.length, label: t("قياسات", "Vitals") },
    ];
    return (
      <View style={[styles.summaryCard, { backgroundColor: isDark ? colors.surface : "#fff", borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)" }]}>
        <View style={styles.summaryRow}>
          {items.map((it, i) => (
            <View key={i} style={[styles.summaryItem, { backgroundColor: isDark ? "rgba(255,255,255,0.04)" : "#F8FAF9" }]}>
              <Ionicons name={it.icon as any} size={16} color={COLORS.primary} />
              <Text style={[styles.summaryCount, { color: colors.text }]}>{it.count}</Text>
              <Text style={[styles.summaryLabel, { color: colors.textMuted }]}>{it.label}</Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const renderMessage = ({ item, index }: { item: ChatMsg; index: number }) => {
    const isUser = item.role === "user";
    const isLast = index === state.messages.length - 1;

    return (
      <AnimatedMessage index={index}>
        <View style={{ marginBottom: 14 }}>
        {/* Bubble row */}
        <View style={[styles.bubbleRow, isUser ? styles.bubbleRowUser : styles.bubbleRowAI]}>
          {!isUser && (
            <View style={styles.aiAvatar}>
              <Ionicons name="sparkles" size={14} color="#fff" />
            </View>
          )}
          {isUser ? (
            <LinearGradient
              colors={[COLORS.primary, "#047857"]}
              style={[styles.bubble, styles.userBubble]}
            >
              <Text style={[styles.bubbleText, { color: "#fff" }]}>{item.content}</Text>
              <Text style={[styles.bubbleTime, { color: "rgba(255,255,255,0.5)" }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </LinearGradient>
          ) : (
            <View style={[styles.bubble, styles.aiBubble, { backgroundColor: isDark ? "rgba(255,255,255,0.05)" : "#fff", borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)" }]}>
              <Text style={[styles.bubbleText, { color: colors.text }]}>{item.content}</Text>
              <Text style={[styles.bubbleTime, { color: colors.textMuted }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </View>
          )}
        </View>

        {/* Summary card on home */}
        {!isUser && (item.content.includes("مرحباً") || item.content.includes("Hi!") || item.content.includes("Welcome")) && (
          <View style={{ marginTop: 10, marginLeft: 34 }}>{renderSummaryCard()}</View>
        )}

        {/* Chips */}
        {isLast && !isUser && item.chips && item.chips.length > 0 && (
          <View style={styles.chipsWrap}>
            {item.chips.map((chip, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.chip, { 
                  backgroundColor: isDark ? "rgba(255,255,255,0.06)" : "#fff", 
                  borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.08)" 
                }]}
                onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); processInput(chip.label, chip.value); }}
                activeOpacity={0.6}
              >
                <Text style={[styles.chipLabel, { color: isDark ? "#D1D5DB" : "#374151" }]}>{chip.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>
      </AnimatedMessage>
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: isDark ? "#0F1419" : "#F5F7F6" }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor="transparent" translucent />

      {/* ── Header ── */}
      <View style={styles.headerWrap}>
        <LinearGradient colors={["#064E3B", "#059669"]} style={styles.headerGradient}>
          <View style={styles.headerContent}>
            <TouchableOpacity onPress={() => router.back()} style={styles.headerBtn} activeOpacity={0.7}>
              <Ionicons name="chevron-back" size={20} color="#fff" />
            </TouchableOpacity>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.headerTitle}>{t("المساعد الطبي", "Medical Assistant")}</Text>
              <View style={styles.statusRow}>
                <View style={styles.statusDot} />
                <Text style={styles.statusLabel}>{t("متصل • ذكاء اصطناعي", "Online • AI")}</Text>
              </View>
            </View>
            <TouchableOpacity
              onPress={() => { setLocalLang(isAr ? "en" : "ar"); Haptics.selectionAsync(); }}
              style={[styles.headerBtn, { marginRight: 6 }]}
              activeOpacity={0.7}
            >
              <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{isAr ? "EN" : "عربي"}</Text>
            </TouchableOpacity>
            {historyStack.length > 0 && (
              <TouchableOpacity onPress={handleUndo} style={styles.headerBtn} activeOpacity={0.7}>
                <Ionicons name="arrow-undo" size={16} color="#fff" />
              </TouchableOpacity>
            )}
          </View>
          <View style={[styles.blob, { top: -30, right: -30, width: 140, height: 140, backgroundColor: "#10B981", opacity: 0.08 }]} />
          <View style={[styles.blob, { bottom: -20, left: -20, width: 100, height: 100, backgroundColor: "#34D399", opacity: 0.06 }]} />
        </LinearGradient>
      </View>

      {/* ── Messages ── */}
      <FlatList
        ref={flatListRef}
        data={state.messages}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderMessage}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />

      {/* ── Typing indicator ── */}
      {isLoading && (
        <View style={styles.typingRow}>
          <TypingIndicator />
          <Text style={[styles.typingText, { color: colors.textMuted }]}>{t("جاري التفكير...", "Thinking...")}</Text>
        </View>
      )}

      {/* ── Input ── */}
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}>
        <View style={[styles.inputContainer, { backgroundColor: isDark ? "#1A1F2E" : "#fff", borderTopColor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)" }]}>
          <View style={[styles.inputRow, { backgroundColor: isDark ? "#252D3A" : "#F1F5F3" }]}>
            <TextInput
              style={[styles.input, { color: colors.text, textAlign: isAr ? "right" : "left" }]}
              placeholder={t("اكتب هنا...", "Type here...")}
              placeholderTextColor={isDark ? "#4B5563" : "#94A3B8"}
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={200}
            />
            <TouchableOpacity
              style={[styles.sendBtn, !inputText.trim() && { opacity: 0.35 }]}
              onPress={handleSendText}
              disabled={!inputText.trim() || isLoading}
              activeOpacity={0.7}
            >
              <Ionicons name="send" size={16} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
      <Toast />
    </View>
  );
}

/* ──────────────── STYLES ──────────────── */
const styles = StyleSheet.create({
  container: { flex: 1 },

  /* Header */
  headerWrap: { borderBottomLeftRadius: 24, borderBottomRightRadius: 24, overflow: "hidden", elevation: 6, shadowColor: "#064E3B", shadowOpacity: 0.1, shadowRadius: 10, shadowOffset: { width: 0, height: 3 } },
  headerGradient: { paddingTop: Platform.OS === "ios" ? 54 : 44, paddingBottom: 16, paddingHorizontal: 16 },
  headerContent: { flexDirection: "row", alignItems: "center", zIndex: 10 },
  headerBtn: { width: 36, height: 36, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.15)", justifyContent: "center", alignItems: "center" },
  headerTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  statusRow: { flexDirection: "row", alignItems: "center", marginTop: 3, gap: 5 },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: "#34D399" },
  statusLabel: { color: "rgba(255,255,255,0.65)", fontSize: 11, fontWeight: "500" },
  blob: { position: "absolute", borderRadius: 200 },

  /* Messages */
  listContent: { paddingHorizontal: 14, paddingTop: 16, paddingBottom: 16 },
  bubbleRow: { flexDirection: "row", maxWidth: "85%" },
  bubbleRowUser: { alignSelf: "flex-end" },
  bubbleRowAI: { alignSelf: "flex-start" },
  aiAvatar: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center", marginRight: 8, marginTop: 2 },
  bubble: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: 18, maxWidth: width * 0.74 },
  userBubble: { borderBottomRightRadius: 4 },
  aiBubble: { borderBottomLeftRadius: 4, borderWidth: 1 },
  bubbleText: { fontSize: 14.5, lineHeight: 22 },
  bubbleTime: { fontSize: 10, marginTop: 4, alignSelf: "flex-end" },

  /* Chips */
  chipsWrap: { flexDirection: "row", flexWrap: "wrap", paddingLeft: 36, paddingTop: 10, gap: 8 },
  chip: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 20, borderWidth: 1 },
  chipLabel: { fontSize: 13.5, fontWeight: "600" },

  /* Summary card */
  summaryCard: { padding: 14, borderRadius: 16, borderWidth: 1 },
  summaryRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, justifyContent: "center" },
  summaryItem: { minWidth: 58, flex: 1, paddingVertical: 10, borderRadius: 12, alignItems: "center" },
  summaryCount: { fontSize: 18, fontWeight: "800", marginTop: 4 },
  summaryLabel: { fontSize: 10, fontWeight: "600", marginTop: 2 },
  summarySubtext: { fontSize: 10, color: "#888", marginTop: 2, textAlign: "center" },
  chatContainer: { flex: 1, padding: 14 },

  /* Typing */
  typingRow: { flexDirection: "row", alignItems: "center", paddingHorizontal: 14, paddingBottom: 8, gap: 8, marginLeft: 42 },
  typingText: { fontSize: 12, fontStyle: "italic" },
  typingDot: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: COLORS.primary },

  /* Input */
  inputContainer: { paddingHorizontal: 12, paddingTop: 10, paddingBottom: Platform.OS === "ios" ? 28 : 10, borderTopWidth: 1 },
  inputRow: { flexDirection: "row", alignItems: "center", borderRadius: 24, paddingLeft: 14, paddingRight: 5, paddingVertical: 4 },
  input: { flex: 1, minHeight: 38, maxHeight: 100, fontSize: 15, paddingHorizontal: 6, paddingTop: Platform.OS === "ios" ? 10 : 7 },
  sendBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center", marginLeft: 6 },
});
`;

content = before + newRender;
fs.writeFileSync(filePath, content, 'utf8');
console.log('Clean UI applied!');
