import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  ActivityIndicator, Alert, Modal, TextInput, KeyboardAvoidingView,
  Platform, Animated,
} from "react-native";
import { useLocalSearchParams, useRouter, useFocusEffect } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";

const MOCK_DOCTORS: any[] = [
  { id: "1", name: "Dr. Sarah Ahmed",   specialty: "Cardiology",   rating: 4.8, reviewCount: 124, location: "Cairo, Egypt",      experience: "12 yrs", consultationFee: 50,  isAvailable: true,  bio: "Specialist in cardiovascular diseases with over 12 years of experience." },
  { id: "2", name: "Dr. Mohamed Ali",   specialty: "Dermatology",  rating: 4.6, reviewCount: 89,  location: "Giza, Egypt",       experience: "8 yrs",  consultationFee: 40,  isAvailable: true,  bio: "Expert in skin conditions, cosmetic dermatology, and hair treatment." },
  { id: "3", name: "Dr. Nour Hassan",   specialty: "Pediatrics",   rating: 4.9, reviewCount: 210, location: "Alexandria, Egypt", experience: "15 yrs", consultationFee: 60,  isAvailable: false, bio: "Dedicated pediatrician with a passion for child healthcare." },
  { id: "4", name: "Dr. Ahmed Karim",   specialty: "Neurology",    rating: 4.7, reviewCount: 67,  location: "Cairo, Egypt",      experience: "10 yrs", consultationFee: 70,  isAvailable: true,  bio: "Neurologist specializing in epilepsy, migraines, and neurodegenerative diseases." },
  { id: "5", name: "Dr. Layla Mostafa", specialty: "General",      rating: 4.5, reviewCount: 145, location: "Cairo, Egypt",      experience: "6 yrs",  consultationFee: 30,  isAvailable: true,  bio: "General practitioner providing comprehensive primary care for all ages." },
  { id: "6", name: "Dr. Omar Farouk",   specialty: "Cardiology",   rating: 4.3, reviewCount: 55,  location: "Giza, Egypt",       experience: "9 yrs",  consultationFee: 55,  isAvailable: false, bio: "Cardiologist with expertise in echocardiography and cardiac rehabilitation." },
];

const TIME_SLOTS = [
  "09:00 AM","09:30 AM","10:00 AM","10:30 AM",
  "11:00 AM","11:30 AM","12:00 PM","02:00 PM",
  "02:30 PM","03:00 PM","03:30 PM","04:00 PM",
];

function getNextDays(count: number) {
  const dayNames   = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
  const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return Array.from({ length: count }, (_, i) => {
    const d = new Date(); d.setDate(d.getDate() + i);
    return {
      label:    i === 0 ? "Today" : dayNames[d.getDay()],
      date:     `${d.getDate()} ${monthNames[d.getMonth()]}`,
      fullDate: d.toISOString().split("T")[0],
    };
  });
}

function formatCardNumber(v: string) {
  return v.replace(/\D/g, "").slice(0, 16).replace(/(.{4})/g, "$1 ").trim();
}
function formatExpiry(v: string) {
  const digits = v.replace(/\D/g, "").slice(0, 4);
  return digits.length > 2 ? digits.slice(0, 2) + "/" + digits.slice(2) : digits;
}

type PayMethod = "visa" | "cash" | null;
type ModalStep = "payment" | "visa_form" | null;

interface Review { author: string; rating: number; comment: string; date: string; }

// ── Star Row ──────────────────────────────────────────
function StarRow({ value, onChange, size = 28 }: { value: number; onChange?: (n: number) => void; size?: number }) {
  return (
    <View style={{ flexDirection: "row", gap: 4 }}>
      {[1, 2, 3, 4, 5].map((n) => (
        <TouchableOpacity key={n} onPress={() => onChange?.(n)} disabled={!onChange}>
          <Ionicons
            name={n <= value ? "star" : "star-outline"}
            size={size} color={n <= value ? "#FFB300" : "#CCC"}
          />
        </TouchableOpacity>
      ))}
    </View>
  );
}

export default function DoctorDetailsScreen() {
  const { doctorId } = useLocalSearchParams<{ doctorId: string }>();
  const router = useRouter();

  const [doctor,       setDoctor]       = useState<any>(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState("");
  const [selectedDay,  setSelectedDay]  = useState(0);
  const [selectedTime, setSelectedTime] = useState<string | null>(null);
  const [payMethod,    setPayMethod]    = useState<PayMethod>(null);
  const [booking,      setBooking]      = useState(false);
  const [modalStep,    setModalStep]    = useState<ModalStep>(null);
  const [showSuccess,  setShowSuccess]  = useState(false);

  // Visa form
  const [cardName,   setCardName]   = useState("");
  const [cardNumber, setCardNumber] = useState("");
  const [expiry,     setExpiry]     = useState("");
  const [cvv,        setCvv]        = useState("");
  const [cardErrors, setCardErrors] = useState<any>({});

  // Reviews
  const [reviews,      setReviews]      = useState<Review[]>([]);
  const [showAddReview,setShowAddReview]= useState(false);
  const [myRating,     setMyRating]     = useState(0);
  const [myComment,    setMyComment]    = useState("");
  const [savingReview, setSavingReview] = useState(false);

  const days = getNextDays(7);


  useEffect(() => {
    if (!doctorId) return;
    AsyncStorage.removeItem("reviews_doctor_undefined");
    fetchDoctor();
  }, [doctorId]);

  useFocusEffect(
    useCallback(() => {
      loadReviews();
    }, [doctorId])
  );

  const fetchDoctor = async () => {
    try {
      await new Promise((r) => setTimeout(r, 300));
      const found = MOCK_DOCTORS.find((d) => d.id === String(doctorId));
      if (!found) throw new Error("Doctor not found");
      setDoctor(found);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const loadReviews = async () => {
    if (!doctorId || doctorId === "undefined") return;
    setReviews([]); // امسح الـ reviews القديمة الأول
    try {
      const key = `reviews_doctor_${doctorId}`;
      const stored = await AsyncStorage.getItem(key);
      setReviews(stored ? JSON.parse(stored) : []);
    } catch {}
  };

  // ── Visa validation ──
  const validateCard = () => {
    const errs: any = {};
    if (!cardName.trim())                     errs.cardName   = "Name is required";
    if (cardNumber.replace(/\s/g,"").length < 16) errs.cardNumber = "Enter full 16-digit card number";
    if (expiry.length < 5)                    errs.expiry     = "Enter valid expiry MM/YY";
    if (cvv.length < 3)                       errs.cvv        = "Enter valid CVV";
    setCardErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleProceed = () => {
    if (!selectedTime) { Alert.alert("Select a time", "Please choose a time slot first."); return; }
    setModalStep("payment");
  };

  const handlePaymentChoice = (method: PayMethod) => {
    setPayMethod(method);
    if (method === "visa") { setModalStep("visa_form"); }
    else { setModalStep(null); confirmBooking("cash"); }
  };

  const handleVisaConfirm = () => {
    if (!validateCard()) return;
    setModalStep(null);
    confirmBooking("visa");
  };

  const confirmBooking = async (method: PayMethod) => {
    setBooking(true);
    try {
      await new Promise((r) => setTimeout(r, 900));

      // ── حفظ الحجز في AsyncStorage ──
      const newBooking = {
        id:            Date.now().toString(),
        doctorName:    doctor.name,
        specialty:     doctor.specialty,
        date:          days[selectedDay].date,
        time:          selectedTime,
        paymentMethod: method,
        bookedAt:      new Date().toISOString(),
      };
      const raw      = await AsyncStorage.getItem("my_bookings");
      const existing = raw ? JSON.parse(raw) : [];
      await AsyncStorage.setItem("my_bookings", JSON.stringify([newBooking, ...existing]));
      // ────────────────────────────────

      setPayMethod(method);
      setShowSuccess(true);
    } catch (e: any) {
      Alert.alert("Booking Failed", e.message);
    } finally { setBooking(false); }
  };

  // ── Save Review ──
  const submitReview = async () => {
    if (myRating === 0)    { Alert.alert("Rating required", "Please select a star rating."); return; }
    if (!myComment.trim()) { Alert.alert("Comment required", "Please write a comment."); return; }
    setSavingReview(true);
    try {
      const userName = await AsyncStorage.getItem("userName") || "Anonymous";
      const newReview: Review = {
        author:  userName,
        rating:  myRating,
        comment: myComment.trim(),
        date:    new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }),
      };
      const updated = [newReview, ...reviews];
      setReviews(updated);
      await AsyncStorage.setItem(`reviews_doctor_${doctorId}`, JSON.stringify(updated));
      setMyRating(0); setMyComment(""); setShowAddReview(false);
    } finally { setSavingReview(false); }
  };

  const avgRating = reviews.length
    ? (reviews.reduce((s, r) => s + r.rating, 0) / reviews.length).toFixed(1)
    : doctor?.rating?.toFixed(1);

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color={COLORS.primary} /></View>;
  if (error || !doctor) return (
    <View style={styles.center}>
      <Text style={styles.errorTxt}>⚠️ {error || "Doctor not found"}</Text>
      <TouchableOpacity style={styles.retryBtn} onPress={fetchDoctor}>
        <Text style={styles.retryTxt}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Back */}
      <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
        <Ionicons name="chevron-back" size={22} color="#1A1A1A" />
      </TouchableOpacity>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Profile Header */}
        <View style={styles.profileCard}>
          <View style={styles.avatarWrap}>
            <View style={styles.avatarFallback}>
              <Text style={styles.avatarTxt}>{doctor.name?.charAt(4)?.toUpperCase() ?? "D"}</Text>
            </View>
            {doctor.isAvailable && <View style={styles.onlineDot} />}
          </View>
          <Text style={styles.docName}>{doctor.name}</Text>
          <Text style={styles.docSpec}>{doctor.specialty}</Text>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Ionicons name="star" size={14} color="#FFB300" />
              <Text style={styles.statVal}>{avgRating}</Text>
              <Text style={styles.statLbl}>Rating</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="time-outline" size={14} color={COLORS.primary} />
              <Text style={styles.statVal}>{doctor.experience}</Text>
              <Text style={styles.statLbl}>Experience</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="people-outline" size={14} color={COLORS.primary} />
              <Text style={styles.statVal}>{doctor.reviewCount + reviews.length}</Text>
              <Text style={styles.statLbl}>Reviews</Text>
            </View>
          </View>
        </View>

        {/* About */}
        {doctor.bio && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>About</Text>
            <Text style={styles.bio}>{doctor.bio}</Text>
          </View>
        )}

        {/* Info pills */}
        <View style={styles.pills}>
          <View style={styles.pill}>
            <Ionicons name="location-outline" size={14} color={COLORS.primary} />
            <Text style={styles.pillTxt}>{doctor.location}</Text>
          </View>
          <View style={styles.pill}>
            <Ionicons name="cash-outline" size={14} color={COLORS.primary} />
            <Text style={styles.pillTxt}>${doctor.consultationFee} per visit</Text>
          </View>
        </View>

        {/* Date */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Select Date</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10 }}>
            {days.map((day, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.dayChip, selectedDay === i && styles.dayChipActive]}
                onPress={() => setSelectedDay(i)}
              >
                <Text style={[styles.dayLabel, selectedDay === i && styles.dayLabelActive]}>{day.label}</Text>
                <Text style={[styles.dayDate,  selectedDay === i && styles.dayDateActive]}>{day.date}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Time */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Select Time</Text>
          <View style={styles.timeGrid}>
            {TIME_SLOTS.map((slot) => (
              <TouchableOpacity
                key={slot}
                style={[styles.timeChip, selectedTime === slot && styles.timeChipActive]}
                onPress={() => setSelectedTime(slot)}
              >
                <Text style={[styles.timeTxt, selectedTime === slot && styles.timeTxtActive]}>{slot}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* ── Reviews ── */}
        <View style={styles.section}>
          <View style={styles.reviewHeader}>
            <Text style={styles.sectionTitle}>Reviews ({reviews.length})</Text>
            <TouchableOpacity style={styles.addReviewBtn} onPress={() => setShowAddReview(true)}>
              <Ionicons name="add" size={14} color="#fff" />
              <Text style={styles.addReviewTxt}>Add Review</Text>
            </TouchableOpacity>
          </View>

          {reviews.length === 0 ? (
            <View style={styles.noReviews}>
              <Ionicons name="chatbubble-outline" size={32} color="#DDD" />
              <Text style={styles.noReviewsTxt}>No reviews yet — be the first!</Text>
            </View>
          ) : (
            reviews.map((r, i) => (
              <View key={i} style={styles.reviewCard}>
                <View style={styles.reviewTop}>
                  <View style={styles.reviewAvatar}>
                    <Text style={styles.reviewAvatarTxt}>{r.author.charAt(0).toUpperCase()}</Text>
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.reviewAuthor}>{r.author}</Text>
                    <Text style={styles.reviewDate}>{r.date}</Text>
                  </View>
                  <StarRow value={r.rating} size={13} />
                </View>
                <Text style={styles.reviewComment}>{r.comment}</Text>
              </View>
            ))
          )}
        </View>

        <View style={{ height: 100 }} />
      </ScrollView>

      {/* Book Button */}
      <View style={styles.bottomBar}>
        <View style={styles.feePill}>
          <Text style={styles.feeLbl}>Fee</Text>
          <Text style={styles.feeAmt}>${doctor.consultationFee}</Text>
        </View>
        <TouchableOpacity
          style={[styles.bookBtn, !selectedTime && styles.bookBtnOff]}
          onPress={handleProceed}
          disabled={booking || !selectedTime}
        >
          {booking
            ? <ActivityIndicator color="#fff" />
            : <>
                <Text style={styles.bookBtnTxt}>{selectedTime ? "Continue" : "Pick a Time"}</Text>
                {selectedTime && <Ionicons name="arrow-forward" size={15} color="#fff" />}
              </>
          }
        </TouchableOpacity>
      </View>

      {/* ── Payment Choice Modal ── */}
      <Modal visible={modalStep === "payment"} transparent animationType="slide">
        <TouchableOpacity style={styles.overlay} activeOpacity={1} onPress={() => setModalStep(null)}>
          <View style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Choose Payment</Text>
            <Text style={styles.sheetSub}>{days[selectedDay].date} · {selectedTime} · ${doctor.consultationFee}</Text>

            <TouchableOpacity style={styles.payOption} onPress={() => handlePaymentChoice("visa")}>
              <View style={[styles.payIconBg, { backgroundColor: "#EEF2FF" }]}>
                <Ionicons name="card" size={22} color="#4F46E5" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.payTitle}>Pay with Visa</Text>
                <Text style={styles.paySub}>Credit / Debit card</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color="#CCC" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.payOption} onPress={() => handlePaymentChoice("cash")}>
              <View style={[styles.payIconBg, { backgroundColor: "#ECFDF5" }]}>
                <Ionicons name="cash" size={22} color={COLORS.primary} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.payTitle}>Pay on Arrival</Text>
                <Text style={styles.paySub}>Cash at clinic</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color="#CCC" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.cancelBtn} onPress={() => setModalStep(null)}>
              <Text style={styles.cancelTxt}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      {/* ── Visa Form Modal ── */}
      <Modal visible={modalStep === "visa_form"} transparent animationType="slide">
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.overlay}>
          <View style={[styles.sheet, { paddingBottom: 36 }]}>
            <View style={styles.sheetHandle} />
            <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 6 }}>
              <TouchableOpacity onPress={() => setModalStep("payment")} style={{ marginRight: 10 }}>
                <Ionicons name="chevron-back" size={22} color="#333" />
              </TouchableOpacity>
              <Text style={styles.sheetTitle}>Card Details</Text>
            </View>
            <Text style={styles.sheetSub}>All transactions are secure & encrypted</Text>

            {/* Card Name */}
            <Text style={styles.fieldLabel}>Cardholder Name</Text>
            <TextInput
              style={[styles.input, cardErrors.cardName && styles.inputError]}
              placeholder="Name on card"
              placeholderTextColor="#BBB"
              value={cardName}
              onChangeText={setCardName}
              autoCapitalize="words"
            />
            {cardErrors.cardName && <Text style={styles.fieldError}>{cardErrors.cardName}</Text>}

            {/* Card Number */}
            <Text style={styles.fieldLabel}>Card Number</Text>
            <View style={[styles.inputRow, cardErrors.cardNumber && styles.inputError]}>
              <Ionicons name="card-outline" size={18} color="#AAA" />
              <TextInput
                style={[styles.input, { flex: 1, borderWidth: 0, marginBottom: 0, padding: 0 }]}
                placeholder="0000 0000 0000 0000"
                placeholderTextColor="#BBB"
                value={cardNumber}
                onChangeText={(v) => setCardNumber(formatCardNumber(v))}
                keyboardType="numeric"
                maxLength={19}
              />
            </View>
            {cardErrors.cardNumber && <Text style={styles.fieldError}>{cardErrors.cardNumber}</Text>}

            {/* Expiry + CVV */}
            <View style={{ flexDirection: "row", gap: 12 }}>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLabel}>Expiry Date</Text>
                <TextInput
                  style={[styles.input, cardErrors.expiry && styles.inputError]}
                  placeholder="MM/YY"
                  placeholderTextColor="#BBB"
                  value={expiry}
                  onChangeText={(v) => setExpiry(formatExpiry(v))}
                  keyboardType="numeric"
                  maxLength={5}
                />
                {cardErrors.expiry && <Text style={styles.fieldError}>{cardErrors.expiry}</Text>}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLabel}>CVV</Text>
                <TextInput
                  style={[styles.input, cardErrors.cvv && styles.inputError]}
                  placeholder="•••"
                  placeholderTextColor="#BBB"
                  value={cvv}
                  onChangeText={(v) => setCvv(v.replace(/\D/g, "").slice(0, 4))}
                  keyboardType="numeric"
                  secureTextEntry
                  maxLength={4}
                />
                {cardErrors.cvv && <Text style={styles.fieldError}>{cardErrors.cvv}</Text>}
              </View>
            </View>

            <TouchableOpacity style={styles.confirmPayBtn} onPress={handleVisaConfirm}>
              <Ionicons name="lock-closed" size={15} color="#fff" />
              <Text style={styles.confirmPayTxt}>Pay ${doctor.consultationFee} Securely</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* ── Add Review Modal ── */}
      <Modal visible={showAddReview} transparent animationType="slide">
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.overlay}>
          <View style={[styles.sheet, { paddingBottom: 36 }]}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Write a Review</Text>
            <Text style={styles.sheetSub}>Share your experience with {doctor.name}</Text>

            <Text style={styles.fieldLabel}>Your Rating</Text>
            <StarRow value={myRating} onChange={setMyRating} size={32} />
            {myRating > 0 && (
              <Text style={[styles.fieldLabel, { color: "#FFB300", marginTop: 6 }]}>
                {["","Poor","Fair","Good","Very Good","Excellent"][myRating]}
              </Text>
            )}

            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Your Comment</Text>
            <TextInput
              style={[styles.input, styles.textarea]}
              placeholder="Tell others about your experience..."
              placeholderTextColor="#BBB"
              value={myComment}
              onChangeText={setMyComment}
              multiline
              numberOfLines={4}
              textAlignVertical="top"
            />

            <View style={{ flexDirection: "row", gap: 10, marginTop: 8 }}>
              <TouchableOpacity
                style={[styles.confirmPayBtn, { flex: 1, backgroundColor: "#F0F0F0" }]}
                onPress={() => setShowAddReview(false)}
              >
                <Text style={[styles.confirmPayTxt, { color: "#555" }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.confirmPayBtn, { flex: 2 }]}
                onPress={submitReview}
                disabled={savingReview}
              >
                {savingReview
                  ? <ActivityIndicator color="#fff" size="small" />
                  : <Text style={styles.confirmPayTxt}>Submit Review</Text>
                }
              </TouchableOpacity>
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* ── Success Modal ── */}
      <Modal visible={showSuccess} transparent animationType="fade">
        <View style={styles.successOverlay}>
          <View style={styles.successCard}>
            <View style={styles.successIconWrap}>
              <Ionicons name="checkmark-circle" size={64} color={COLORS.primary} />
            </View>
            <Text style={styles.successTitle}>Booking Confirmed!</Text>
            <Text style={styles.successDoc}>{doctor.name}</Text>
            <View style={styles.successDetails}>
              <View style={styles.successRow}>
                <Ionicons name="calendar-outline" size={15} color={COLORS.primary} />
                <Text style={styles.successDetailTxt}>{days[selectedDay].date} · {selectedTime}</Text>
              </View>
              <View style={styles.successRow}>
                <Ionicons name={payMethod === "visa" ? "card-outline" : "cash-outline"} size={15} color={COLORS.primary} />
                <Text style={styles.successDetailTxt}>
                  {payMethod === "visa" ? "Paid by Visa" : "Pay on Arrival"}
                </Text>
              </View>
            </View>
            <TouchableOpacity
              style={styles.doneBtn}
              onPress={() => { setShowSuccess(false); router.back(); }}
            >
              <Text style={styles.doneBtnTxt}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: "#F4F6FA" },
  center:     { flex: 1, justifyContent: "center", alignItems: "center", gap: 12 },
  errorTxt:   { color: "#e53935", fontSize: 14, textAlign: "center" },
  retryBtn:   { backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 9, borderRadius: 18 },
  retryTxt:   { color: "#fff", fontWeight: "600" },

  backBtn: {
    position: "absolute", top: 48, left: 16, zIndex: 10,
    backgroundColor: "#fff", borderRadius: 14, padding: 10,
    shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 6, elevation: 4,
  },

  profileCard: {
    backgroundColor: "#fff", alignItems: "center",
    paddingTop: 88, paddingBottom: 24, paddingHorizontal: 20,
    borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
    shadowColor: "#000", shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.06, shadowRadius: 10, elevation: 4,
  },
  avatarWrap:     { position: "relative", marginBottom: 12 },
  avatarFallback: {
    width: 84, height: 84, borderRadius: 42,
    backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center",
    borderWidth: 3, borderColor: COLORS.primary + "30",
  },
  avatarTxt: { fontSize: 32, fontWeight: "800", color: COLORS.primary },
  onlineDot: {
    position: "absolute", bottom: 4, right: 4,
    width: 15, height: 15, borderRadius: 8,
    backgroundColor: "#4CAF50", borderWidth: 3, borderColor: "#fff",
  },
  docName:    { fontSize: 20, fontWeight: "800", color: "#1A1A1A" },
  docSpec:    { fontSize: 13, color: COLORS.primary, fontWeight: "600", marginTop: 3 },
  statsRow:   {
    flexDirection: "row", alignItems: "center",
    marginTop: 18, paddingTop: 18, borderTopWidth: 1, borderTopColor: "#F0F0F0",
    width: "100%", justifyContent: "center",
  },
  statItem:   { flex: 1, alignItems: "center", gap: 3 },
  statDivider:{ width: 1, height: 32, backgroundColor: "#F0F0F0" },
  statVal:    { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  statLbl:    { fontSize: 10, color: "#AAA" },

  section:     { marginTop: 20, paddingHorizontal: 18 },
  sectionTitle:{ fontSize: 16, fontWeight: "700", color: "#1A1A1A", marginBottom: 12 },
  bio:         { fontSize: 13, color: "#666", lineHeight: 20 },

  pills: { flexDirection: "row", flexWrap: "wrap", gap: 10, paddingHorizontal: 18, marginTop: 14 },
  pill:  {
    flexDirection: "row", alignItems: "center", gap: 6,
    backgroundColor: COLORS.primary + "12", paddingHorizontal: 12, paddingVertical: 7, borderRadius: 20,
  },
  pillTxt: { fontSize: 12, color: "#444", fontWeight: "500" },

  dayChip:       { alignItems: "center", paddingHorizontal: 14, paddingVertical: 10, borderRadius: 14, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#EEE", minWidth: 68 },
  dayChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  dayLabel:      { fontSize: 10, color: "#AAA", fontWeight: "500" },
  dayLabelActive:{ color: "rgba(255,255,255,0.8)" },
  dayDate:       { fontSize: 13, color: "#1A1A1A", fontWeight: "700", marginTop: 2 },
  dayDateActive: { color: "#fff" },

  timeGrid:      { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  timeChip:      { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 12, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#EEE" },
  timeChipActive:{ backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  timeTxt:       { fontSize: 12, color: "#333", fontWeight: "500" },
  timeTxtActive: { color: "#fff" },

  /* Reviews */
  reviewHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  addReviewBtn: {
    flexDirection: "row", alignItems: "center", gap: 4,
    backgroundColor: COLORS.primary, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 18,
  },
  addReviewTxt: { color: "#fff", fontSize: 12, fontWeight: "600" },
  noReviews:    { alignItems: "center", paddingVertical: 24, gap: 8 },
  noReviewsTxt: { fontSize: 13, color: "#BBB" },
  reviewCard: {
    backgroundColor: "#fff", borderRadius: 14, padding: 14, marginBottom: 10,
    shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 4, elevation: 1,
  },
  reviewTop:      { flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 8 },
  reviewAvatar:   { width: 36, height: 36, borderRadius: 18, backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center" },
  reviewAvatarTxt:{ fontSize: 15, fontWeight: "700", color: COLORS.primary },
  reviewAuthor:   { fontSize: 13, fontWeight: "700", color: "#1A1A1A" },
  reviewDate:     { fontSize: 11, color: "#BBB", marginTop: 1 },
  reviewComment:  { fontSize: 13, color: "#555", lineHeight: 20 },

  /* Bottom bar */
  bottomBar: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    backgroundColor: "#fff", paddingHorizontal: 18, paddingTop: 12, paddingBottom: 28,
    borderTopWidth: 1, borderTopColor: "#F0F0F0",
    flexDirection: "row", alignItems: "center", gap: 12,
  },
  feePill:   { backgroundColor: COLORS.primary + "12", paddingHorizontal: 14, paddingVertical: 10, borderRadius: 14, alignItems: "center" },
  feeLbl:    { fontSize: 9, color: COLORS.primary, fontWeight: "600" },
  feeAmt:    { fontSize: 15, fontWeight: "800", color: COLORS.primary },
  bookBtn:   { flex: 1, backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 15, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 },
  bookBtnOff:{ backgroundColor: "#DDD" },
  bookBtnTxt:{ color: "#fff", fontSize: 15, fontWeight: "700" },

  /* Sheets */
  overlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.45)", justifyContent: "flex-end" },
  sheet: {
    backgroundColor: "#fff", borderTopLeftRadius: 28, borderTopRightRadius: 28,
    paddingHorizontal: 22, paddingTop: 14, paddingBottom: 28,
  },
  sheetHandle:{ width: 40, height: 4, borderRadius: 2, backgroundColor: "#DDD", alignSelf: "center", marginBottom: 18 },
  sheetTitle: { fontSize: 19, fontWeight: "800", color: "#1A1A1A", marginBottom: 3 },
  sheetSub:   { fontSize: 12, color: "#AAA", marginBottom: 20 },

  payOption: {
    flexDirection: "row", alignItems: "center", gap: 14,
    backgroundColor: "#FAFAFA", borderRadius: 16, padding: 14, marginBottom: 10,
    borderWidth: 1.5, borderColor: "#F0F0F0",
  },
  payIconBg: { width: 44, height: 44, borderRadius: 14, justifyContent: "center", alignItems: "center" },
  payTitle:  { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  paySub:    { fontSize: 11, color: "#AAA", marginTop: 2 },
  cancelBtn: { alignItems: "center", paddingVertical: 12, marginTop: 4 },
  cancelTxt: { color: "#AAA", fontSize: 14, fontWeight: "500" },

  /* Visa form */
  fieldLabel: { fontSize: 12, fontWeight: "600", color: "#555", marginBottom: 6, marginTop: 12 },
  input: {
    backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12,
    fontSize: 14, color: "#1A1A1A", borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2,
  },
  inputRow: {
    flexDirection: "row", alignItems: "center", gap: 10,
    backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12,
    borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2,
  },
  inputError: { borderColor: "#e53935" },
  fieldError: { fontSize: 11, color: "#e53935", marginBottom: 4 },
  textarea:   { height: 90, paddingTop: 12 },

  confirmPayBtn: {
    backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 15,
    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 16,
  },
  confirmPayTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },

  /* Success */
  successOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "center", alignItems: "center", padding: 24 },
  successCard:    { backgroundColor: "#fff", borderRadius: 28, padding: 28, alignItems: "center", width: "100%" },
  successIconWrap:{ width: 90, height: 90, borderRadius: 45, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center", marginBottom: 14 },
  successTitle:   { fontSize: 20, fontWeight: "800", color: "#1A1A1A" },
  successDoc:     { fontSize: 14, color: COLORS.primary, fontWeight: "600", marginTop: 3, marginBottom: 18 },
  successDetails: { backgroundColor: "#F8F8F8", borderRadius: 14, padding: 14, width: "100%", gap: 10 },
  successRow:     { flexDirection: "row", alignItems: "center", gap: 10 },
  successDetailTxt:{ fontSize: 13, color: "#555", fontWeight: "500" },
  doneBtn:        { backgroundColor: COLORS.primary, borderRadius: 18, paddingHorizontal: 44, paddingVertical: 13, marginTop: 18 },
  doneBtnTxt:     { color:"#fff", fontSize: 15, fontWeight: "700" },
});