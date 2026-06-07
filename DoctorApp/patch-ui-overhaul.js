const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'app', '(patient)', 'ai-profile-assistant.tsx');
let content = fs.readFileSync(filePath, 'utf8');

// Find the render section start and cut everything from there to end
const renderStart = content.indexOf('  /* ──────────────── RENDER ──────────────── */');
if (renderStart === -1) { console.error('Could not find render section'); process.exit(1); }

const before = content.substring(0, renderStart);

const newRender = `  /* ──────────────── RENDER ──────────────── */
  const renderSummaryCard = () => {
    if (!diseases.length && !meds.length && !allergies.length && !surgeries.length && !vitals.length) return null;
    const items = [
      { icon: "heart", color: "#EF4444", bg: "#FEF2F2", darkBg: "#3B1111", count: diseases.length, label: t("أمراض", "Diseases") },
      { icon: "medkit", color: "#10B981", bg: "#ECFDF5", darkBg: "#0D3326", count: meds.length, label: t("أدوية", "Meds") },
      { icon: "alert-circle", color: "#F59E0B", bg: "#FFFBEB", darkBg: "#3B2E0A", count: allergies.length, label: t("حساسية", "Allergies") },
      { icon: "cut", color: "#6366F1", bg: "#EEF2FF", darkBg: "#1E1B4B", count: surgeries.length, label: t("جراحات", "Surgeries") },
      { icon: "pulse", color: "#06B6D4", bg: "#ECFEFF", darkBg: "#0B3644", count: vitals.length, label: t("قياسات", "Vitals") },
    ];
    return (
      <View style={[styles.summaryCard, { backgroundColor: isDark ? colors.surface : "#fff", borderColor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }]}>
        <Text style={[styles.summaryTitle, { color: colors.textMuted }]}>{t("ملخص ملفك", "Your Profile")}</Text>
        <View style={styles.summaryRow}>
          {items.map((it, i) => (
            <View key={i} style={[styles.summaryItem, { backgroundColor: isDark ? it.darkBg : it.bg }]}>
              <View style={[styles.summaryIconCircle, { backgroundColor: it.color + "18" }]}>
                <Ionicons name={it.icon as any} size={15} color={it.color} />
              </View>
              <Text style={[styles.summaryCount, { color: it.color }]}>{it.count}</Text>
              <Text style={[styles.summaryLabel, { color: isDark ? "rgba(255,255,255,0.5)" : "#6B7280" }]}>{it.label}</Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const chipIconMap: Record<string, string> = {
    ADD: "add-circle-outline", EDIT: "create-outline", DELETE: "trash-outline",
    SETUP: "settings-outline", HOME: "home-outline",
    DISEASE: "heart-outline", MED: "medkit-outline", ALLERGY: "alert-circle-outline",
    SURGERY: "cut-outline", VITAL: "pulse-outline", GENERAL: "person-outline",
    DRUG: "medical-outline", FOOD: "restaurant-outline", OTHER: "ellipsis-horizontal",
    YES: "checkmark-circle-outline", NO: "close-circle-outline",
    SKIP: "play-skip-forward-outline", NONE: "remove-circle-outline",
    CONFIRM_DEL: "checkmark-done", CANCEL_DEL: "arrow-undo-outline",
  };

  const chipColorMap: Record<string, string> = {
    ADD: "#10B981", EDIT: "#6366F1", DELETE: "#EF4444",
    SETUP: "#F59E0B", HOME: "#64748B",
    DISEASE: "#EF4444", MED: "#10B981", ALLERGY: "#F59E0B",
    SURGERY: "#6366F1", VITAL: "#06B6D4", GENERAL: "#8B5CF6",
    CONFIRM_DEL: "#EF4444", CANCEL_DEL: "#64748B",
  };

  const renderMessage = ({ item, index }: { item: ChatMsg; index: number }) => {
    const isUser = item.role === "user";
    const isLast = index === state.messages.length - 1;

    return (
      <AnimatedMessage index={index}>
        <View style={{ marginBottom: 12 }}>
        {/* Bubble row */}
        <View style={[styles.bubbleRow, isUser ? styles.bubbleRowUser : styles.bubbleRowAI]}>
          {!isUser && (
            <LinearGradient colors={["#059669", "#10B981"]} style={styles.aiAvatar}>
              <Ionicons name="sparkles" size={13} color="#fff" />
            </LinearGradient>
          )}
          {isUser ? (
            <LinearGradient
              colors={["#059669", "#047857"]}
              start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
              style={[styles.bubble, styles.userBubble]}
            >
              <Text style={[styles.bubbleText, { color: "#fff" }]}>{item.content}</Text>
              <Text style={[styles.bubbleTime, { color: "rgba(255,255,255,0.55)" }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </LinearGradient>
          ) : (
            <View style={[styles.bubble, styles.aiBubble, { backgroundColor: isDark ? "rgba(255,255,255,0.05)" : "#fff", borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)" }]}>
              <Text style={[styles.bubbleText, { color: colors.text }]}>{item.content}</Text>
              <Text style={[styles.bubbleTime, { color: colors.textMuted }]}>
                {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </Text>
            </View>
          )}
        </View>

        {/* Summary card on home */}
        {!isUser && (item.content.includes("مرحباً") || item.content.includes("Hi!") || item.content.includes("Welcome")) && (
          <View style={{ marginTop: 10, marginLeft: 33 }}>{renderSummaryCard()}</View>
        )}

        {/* Chips */}
        {isLast && !isUser && item.chips && item.chips.length > 0 && (
          <View style={styles.chipsWrap}>
            {item.chips.map((chip, i) => {
              const accent = chipColorMap[chip.value] || COLORS.primary;
              const iconName = chipIconMap[chip.value];
              return (
                <TouchableOpacity
                  key={i}
                  style={[styles.chip, {
                    backgroundColor: isDark ? accent + "12" : accent + "0A",
                    borderColor: isDark ? accent + "30" : accent + "28",
                  }]}
                  onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); processInput(chip.label, chip.value); }}
                  activeOpacity={0.6}
                >
                  {iconName && <Ionicons name={iconName as any} size={13} color={accent} style={{ marginRight: 4 }} />}
                  <Text style={[styles.chipLabel, { color: accent }]}>{chip.label}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        )}
      </View>
      </AnimatedMessage>
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: isDark ? "#0C1222" : "#F6F8FB" }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor="transparent" translucent />

      {/* ── Header ── */}
      <View style={styles.headerWrap}>
        <LinearGradient colors={["#064E3B", "#059669"]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.headerGradient}>
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
              <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{isAr ? "EN" : "عر"}</Text>
            </TouchableOpacity>
            {historyStack.length > 0 && (
              <TouchableOpacity onPress={handleUndo} style={styles.headerBtn} activeOpacity={0.7}>
                <Ionicons name="arrow-undo" size={16} color="#fff" />
              </TouchableOpacity>
            )}
          </View>
          <View style={[styles.blob, { top: -40, right: -40, width: 140, height: 140, backgroundColor: "#10B981", opacity: 0.08 }]} />
          <View style={[styles.blob, { bottom: -30, left: -30, width: 100, height: 100, backgroundColor: "#34D399", opacity: 0.06 }]} />
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
        <View style={[styles.typingRow, { backgroundColor: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)" }]}>
          <TypingIndicator />
          <Text style={[styles.typingText, { color: colors.textMuted }]}>{t("جاري التفكير...", "Thinking...")}</Text>
        </View>
      )}

      {/* ── Input ── */}
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}>
        <View style={[styles.inputContainer, { backgroundColor: isDark ? "#111827" : "#fff", borderTopColor: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)" }]}>
          <View style={[styles.inputRow, { backgroundColor: isDark ? "#1E293B" : "#F1F5F9" }]}>
            <TextInput
              style={[styles.input, { color: colors.text, textAlign: isAr ? "right" : "left" }]}
              placeholder={t("اكتب رسالتك...", "Type your message...")}
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
              <LinearGradient colors={["#059669", "#10B981"]} style={styles.sendBtnGradient}>
                <Ionicons name="send" size={14} color="#fff" style={{ marginLeft: 1 }} />
              </LinearGradient>
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
  headerWrap: { borderBottomLeftRadius: 22, borderBottomRightRadius: 22, overflow: "hidden", elevation: 6, shadowColor: "#064E3B", shadowOpacity: 0.12, shadowRadius: 10, shadowOffset: { width: 0, height: 3 } },
  headerGradient: { paddingTop: Platform.OS === "ios" ? 52 : 40, paddingBottom: 14, paddingHorizontal: 14 },
  headerContent: { flexDirection: "row", alignItems: "center", zIndex: 10 },
  headerBtn: { width: 34, height: 34, borderRadius: 11, backgroundColor: "rgba(255,255,255,0.15)", justifyContent: "center", alignItems: "center" },
  headerTitle: { color: "#fff", fontSize: 16, fontWeight: "700", letterSpacing: -0.2 },
  statusRow: { flexDirection: "row", alignItems: "center", marginTop: 2, gap: 4 },
  statusDot: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: "#34D399" },
  statusLabel: { color: "rgba(255,255,255,0.6)", fontSize: 10, fontWeight: "500" },
  blob: { position: "absolute", borderRadius: 200 },

  /* Messages */
  listContent: { paddingHorizontal: 12, paddingTop: 14, paddingBottom: 10 },
  bubbleRow: { flexDirection: "row", maxWidth: "84%" },
  bubbleRowUser: { alignSelf: "flex-end" },
  bubbleRowAI: { alignSelf: "flex-start" },
  aiAvatar: { width: 24, height: 24, borderRadius: 12, justifyContent: "center", alignItems: "center", marginRight: 7, marginTop: 2 },
  bubble: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 16, maxWidth: width * 0.72 },
  userBubble: { borderBottomRightRadius: 4 },
  aiBubble: { borderBottomLeftRadius: 4, borderWidth: StyleSheet.hairlineWidth },
  bubbleText: { fontSize: 13.5, lineHeight: 20 },
  bubbleTime: { fontSize: 8.5, marginTop: 3, alignSelf: "flex-end", letterSpacing: 0.2 },

  /* Chips */
  chipsWrap: { flexDirection: "row", flexWrap: "wrap", paddingLeft: 31, paddingTop: 8, gap: 6 },
  chip: { flexDirection: "row", alignItems: "center", paddingHorizontal: 11, paddingVertical: 6, borderRadius: 14, borderWidth: StyleSheet.hairlineWidth },
  chipLabel: { fontSize: 12, fontWeight: "600" },

  /* Summary card */
  summaryCard: { padding: 12, borderRadius: 14, borderWidth: StyleSheet.hairlineWidth },
  summaryTitle: { fontSize: 9, fontWeight: "700", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 },
  summaryRow: { flexDirection: "row", flexWrap: "wrap", gap: 5, justifyContent: "center" },
  summaryItem: { minWidth: 54, flex: 1, paddingVertical: 8, paddingHorizontal: 2, borderRadius: 10, alignItems: "center" },
  summaryIconCircle: { width: 26, height: 26, borderRadius: 13, justifyContent: "center", alignItems: "center" },
  summaryCount: { fontSize: 15, fontWeight: "800", marginTop: 2 },
  summaryLabel: { fontSize: 8.5, fontWeight: "600", marginTop: 1 },
  summarySubtext: { fontSize: 9, color: "#888", marginTop: 1, textAlign: "center" },
  chatContainer: { flex: 1, padding: 12 },

  /* Typing */
  typingRow: { flexDirection: "row", alignItems: "center", paddingHorizontal: 12, paddingVertical: 6, marginHorizontal: 12, marginBottom: 4, borderRadius: 12, gap: 7, marginLeft: 43 },
  typingText: { fontSize: 11, fontStyle: "italic" },
  typingDot: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: COLORS.primary },

  /* Input */
  inputContainer: { paddingHorizontal: 10, paddingTop: 8, paddingBottom: Platform.OS === "ios" ? 26 : 8, borderTopWidth: StyleSheet.hairlineWidth },
  inputRow: { flexDirection: "row", alignItems: "center", borderRadius: 20, paddingLeft: 12, paddingRight: 4, paddingVertical: 3 },
  input: { flex: 1, minHeight: 34, maxHeight: 90, fontSize: 13.5, paddingHorizontal: 4, paddingTop: Platform.OS === "ios" ? 9 : 6 },
  sendBtn: { width: 32, height: 32, borderRadius: 16, overflow: "hidden", marginLeft: 5 },
  sendBtnGradient: { flex: 1, justifyContent: "center", alignItems: "center" },
});
`;

content = before + newRender;
fs.writeFileSync(filePath, content, 'utf8');
console.log('UI overhaul complete!');
