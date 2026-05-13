import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Image,
  ActivityIndicator, Platform, StatusBar, Modal, TextInput,
  KeyboardAvoidingView, Animated, Alert, Dimensions
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { COLORS } from "../../constants/colors";
import { getDoctorById, getReviewsByDoctor, addReview, updateMyReview, deleteMyReview, DoctorDetails, Review } from "../../services/doctorService";
import { bookAppointment, updateAppointment } from "../../services/appointmentService";
import { apiFetch } from "../../services/http";
import { BASE_URL } from "../../constants/api";
import { startSignalRConnection, onDoctorUpdated, onScheduleReady, onScheduleUpdated, subscribeToDoctorSchedule } from "../../services/signalr";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { addNotification } from "../../services/notificationService";
import { checkIfFollowed, setFollowed, toggleFollowed, checkIfSubscribed, setSubscribed } from "../../services/followService";

const { width: SCREEN_WIDTH } = Dimensions.get("window");

const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

const toLocalIsoDate = (date: Date): string => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

const normalizeToLocalIsoDate = (raw: unknown): string => {
  const text = raw?.toString?.().trim() ?? ""
  if (!text) return ""
  const isoMatch = text.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (isoMatch) return `${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}`
  const parsed = new Date(text)
  return !Number.isNaN(parsed.getTime()) ? toLocalIsoDate(parsed) : ""
}

const getDayIndexFromLocalIsoDate = (isoDate: string): number => {
  const [year, month, day] = isoDate.split("-").map(Number)
  return new Date(year, month - 1, day).getDay()
}

const getNormalizedDayName = (value: unknown): string => {
  if (typeof value === 'number') return DAY_NAMES[((value % 7) + 7) % 7].toLowerCase()
  if (typeof value === 'string') {
    const numeric = Number(value)
    if (!Number.isNaN(numeric)) return DAY_NAMES[((numeric % 7) + 7) % 7].toLowerCase()
    return value.trim().toLowerCase()
  }
  return ''
}

const parseTimeToMinutes = (value: unknown): number | null => {
  const text = value?.toString?.().trim() ?? ""
  if (!text) return null
  const amPmMatch = text.match(/^(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)$/i)
  if (amPmMatch) {
    let h = Number(amPmMatch[1])
    const m = Number(amPmMatch[2])
    const marker = amPmMatch[3].toUpperCase()
    if (marker === "PM" && h !== 12) h += 12
    if (marker === "AM" && h === 12) h = 0
    return h * 60 + m
  }
  const hhmmMatch = text.match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/)
  if (hhmmMatch) return Number(hhmmMatch[1]) * 60 + Number(hhmmMatch[2])
  return null
}

const toDisplaySlot = (raw: unknown): string => {
  const text = raw?.toString?.() ?? ""
  if (text.includes("AM") || text.includes("PM")) return text
  const minutes = parseTimeToMinutes(text)
  if (minutes == null) return text
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  const ampm = h >= 12 ? "PM" : "AM"
  const h12 = h % 12 || 12
  return `${h12.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")} ${ampm}`
}

const getTimeSlotsForDay = (date: string, availability: any[], bookedSlots: any[] = []): string[] => {
  if (!date || !availability || availability.length === 0) return []
  const dayIndex = getDayIndexFromLocalIsoDate(date)
  const dayOfWeek = DAY_NAMES[dayIndex].toLowerCase()
  
  const dayAvail = availability.find(a => {
    const rawDow = a.dayName ?? a.DayName ?? a.dayOfWeek ?? a.DayOfWeek ?? a.day ?? a.Day ?? ''
    return getNormalizedDayName(rawDow) === dayOfWeek && (a.isAvailable ?? a.IsAvailable ?? true)
  })

  if (!dayAvail) return []

  const isBooked = (slot: string): boolean => {
    const slotMins = parseTimeToMinutes(slot)
    return bookedSlots.some(bs => {
      const bDate = normalizeToLocalIsoDate(bs.date ?? bs.Date ?? "")
      const bMins = parseTimeToMinutes(bs.time ?? bs.Time ?? "")
      return bDate === date && bMins === slotMins
    })
  }

  const rawSlots = dayAvail.timeSlots ?? dayAvail.TimeSlots
  if (Array.isArray(rawSlots) && rawSlots.length > 0) {
    return rawSlots.map(s => toDisplaySlot(s)).filter(s => !isBooked(s))
  }

  const start = parseTimeToMinutes(dayAvail.startTime ?? dayAvail.StartTime ?? '09:00')
  const end = parseTimeToMinutes(dayAvail.endTime ?? dayAvail.EndTime ?? '17:00')
  const dur = Number(dayAvail.slotDurationMinutes ?? dayAvail.SlotDurationMinutes ?? 30)

  if (start == null || end == null) return []
  const slots: string[] = []
  for (let m = start; m < end; m += dur) {
    const s = toDisplaySlot(`${Math.floor(m / 60)}:${m % 60}`)
    if (!isBooked(s)) {
      const isToday = date === toLocalIsoDate(new Date())
      const nowMins = new Date().getHours() * 60 + new Date().getMinutes()
      if (!isToday || m > nowMins + 15) slots.push(s)
    }
  }
  return slots
}

function getNextAvailableDays(availability: any[], count: number) {
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const result: { label: string; date: string; isoDate: string }[] = []
  const today = new Date()

  for (let i = 0; i < 30 && result.length < count; i++) {
    const d = new Date(today); d.setDate(today.getDate() + i)
    const dayName = DAY_NAMES[d.getDay()].toLowerCase()
    const hasAvail = availability.some(a => getNormalizedDayName(a.dayName ?? a.DayName ?? a.dayOfWeek ?? a.DayOfWeek ?? a.day ?? a.Day) === dayName && (a.isAvailable ?? a.IsAvailable ?? true))
    if (!hasAvail) continue
    result.push({
      label: i === 0 ? "Today" : DAY_NAMES[d.getDay()].slice(0, 3),
      date: `${d.getDate()} ${monthNames[d.getMonth()]}`,
      isoDate: toLocalIsoDate(d)
    })
  }
  return result
}

function getNextDays(count: number) {
  const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return Array.from({ length: count }, (_, i) => {
    const d = new Date(); d.setDate(d.getDate() + i);
    return {
      label: i === 0 ? "Today" : dayNames[d.getDay()],
      date: `${d.getDate()} ${monthNames[d.getMonth()]}`,
      isoDate: toLocalIsoDate(d)
    };
  });
}

const StarRow = ({ value, size = 16 }: { value: number, size?: number }) => (
  <View style={{ flexDirection: "row", gap: 2 }}>
    {[1, 2, 3, 4, 5].map(n => (
      <Ionicons key={n} name={n <= value ? "star" : "star-outline"} size={size} color={n <= value ? "#FBBF24" : "#CBD5E1"} />
    ))}
  </View>
);

export default function DoctorDetailsScreen() {
  const { id, doctorId: doctorIdParam, editAppointmentId, initialDate, initialTime } = useLocalSearchParams<{ 
    id: string, 
    doctorId: string,
    editAppointmentId?: string,
    initialDate?: string,
    initialTime?: string
  }>();
  const doctorId = (id || doctorIdParam) as string;
  const router = useRouter();

  const [doctor, setDoctor] = useState<DoctorDetails | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedDay, setSelectedDay] = useState(() => {
    if (initialDate) {
      // Try to find the day index for initialDate
      return 0; // Defaulting to 0 for now as 'days' is memoized and might not be ready
    }
    return 0;
  });
  const [selectedTime, setSelectedTime] = useState<string | null>(initialTime || null);
  const [booking, setBooking] = useState(false);
  const [modalStep, setModalStep] = useState<"payment" | "visa_form" | null>(null);
  const [payMethod, setPayMethod] = useState<"visa" | "cash" | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  const [availability, setAvailability] = useState<any[]>([]);
  const [bookedSlots, setBookedSlots] = useState<any[]>([]);
  const [hasSchedule, setHasSchedule] = useState(false);
  const [loadingAvailability, setLoadingAvailability] = useState(true);
  const [notifyEnabled, setNotifyEnabled] = useState(false);

  // Visa form
  const [cardName, setCardName] = useState("");
  const [cardNumber, setCardNumber] = useState("");
  const [expiry, setExpiry] = useState("");
  const [cvv, setCvv] = useState("");

  // Review
  const [showAddReview, setShowAddReview] = useState(false);
  const [myRating, setMyRating] = useState(0);
  const [myComment, setMyComment] = useState("");
  const [savingReview, setSavingReview] = useState(false);

  const [isFollowed, setIsFollowed] = useState(false);
  const [following, setFollowing] = useState(false);

  const days = useMemo(() => {
    if (!hasSchedule) return getNextDays(7)
    return getNextAvailableDays(availability, 7)
  }, [availability, hasSchedule])

  const currentSlots = useMemo(() => {
    const day = days[selectedDay]
    if (!day) return []
    return getTimeSlotsForDay(day.isoDate, availability, bookedSlots)
  }, [selectedDay, days, availability, bookedSlots])

  useEffect(() => {
    if (!doctorId) return;
    fetchAll();
    refreshFollowState();
  }, [doctorId]);

  useEffect(() => {
    if (!doctorId) return;

    let cleanupReady: (() => void) | undefined;
    let cleanupUpdated: (() => void) | undefined;
    let cleanupDoctor: (() => void) | undefined;
    let cancelled = false;

    const connectToScheduleUpdates = async () => {
      await startSignalRConnection();
      if (cancelled) return;
      await subscribeToDoctorSchedule(Number(doctorId));
      if (cancelled) return;

      const refreshThisDoctor = (payload: any) => {
        const updatedDoctorId = Number(payload?.doctorId ?? payload?.DoctorId ?? doctorId);
        if (updatedDoctorId === Number(doctorId)) {
          fetchAll(); // Refresh EVERYTHING (Profile + Availability)
          setSelectedTime(null);
        }
      };

      cleanupReady = onScheduleReady(refreshThisDoctor);
      cleanupUpdated = onScheduleUpdated(refreshThisDoctor);
      cleanupDoctor = onDoctorUpdated(refreshThisDoctor); // Refresh on profile update too
    };

    connectToScheduleUpdates().catch(() => undefined);

    return () => {
      cancelled = true;
      cleanupReady?.();
      cleanupUpdated?.();
      cleanupDoctor?.();
    };
  }, [doctorId]);

  const fetchAll = async () => {
    try {
      setLoading(true);
      const [doc, revs] = await Promise.all([
        getDoctorById(doctorId),
        getReviewsByDoctor(doctorId),
      ]);
      setDoctor(doc);
      setReviews(revs);
      fetchAvailability();
    } catch (e: any) {
      setError(e.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailability = async () => {
    try {
      setLoadingAvailability(true);
      const data = await apiFetch<any>(`${BASE_URL}/api/doctors/${doctorId}/availability`, { method: 'GET' }, false);
      const avail = data?.days ?? data?.Days ?? (Array.isArray(data) ? data : []);
      const booked = data?.bookedSlots ?? data?.BookedSlots ?? [];
      setAvailability(avail);
      setBookedSlots(booked);
      setHasSchedule(avail.some((a: any) => (a.isAvailable ?? a.IsAvailable ?? true)));
    } catch {
      setHasSchedule(false);
    } finally {
      setLoadingAvailability(false);
    }
  };

  const refreshFollowState = async () => {
    const f = await checkIfFollowed(Number(doctorId));
    const s = await checkIfSubscribed(Number(doctorId));
    setIsFollowed(f);
    setNotifyEnabled(f || s);
  };

  const toggleFollowDoctor = async () => {
    setFollowing(true);
    try {
      const next = !isFollowed;
      await setFollowed(Number(doctorId), next);
      setIsFollowed(next);
      if (next) {
        await startSignalRConnection();
        await subscribeToDoctorSchedule(Number(doctorId));
      }
    } finally {
      setFollowing(false);
    }
  };

  const handleNotifyMe = async () => {
    try {
      const next = !notifyEnabled;
      await setSubscribed(Number(doctorId), next);
      setNotifyEnabled(next);
      if (next) Alert.alert("Notifications Enabled", "We will notify you when Dr. " + doctor?.name + " updates their schedule.");
    } catch { Alert.alert("Error", "Failed to update notification settings."); }
  };

  const handleProceed = () => {
    if (!selectedTime) { Alert.alert("Select Time", "Please choose a preferred time slot first."); return; }
    setModalStep("payment");
  };

  const confirmBooking = async (method: "visa" | "cash") => {
    setBooking(true);
    try {
      const day = days[selectedDay];
      if (!day || !selectedTime) {
        setModalStep(null);
        Alert.alert("Slot Unavailable", "Please choose another available time slot.");
        return;
      }
      const time = selectedTime;
      const payload = {
        doctorId: Number(doctorId),
        date: day.isoDate,
        time,
        paymentMethod: method,
      };

      if (editAppointmentId) {
        await updateAppointment(Number(editAppointmentId), payload);
        Toast.show({ type: "success", text1: "Appointment Updated", text2: "Your booking has been rescheduled." });
      } else {
        await bookAppointment(payload);
        Toast.show({ type: "success", text1: "Booking Successful", text2: "Your appointment has been confirmed." });
      }

      setBookedSlots(prev => [...prev, { date: day.isoDate, time }]);
      setSelectedTime(null);
      setModalStep(null);
      setShowSuccess(true);
      fetchAvailability();
    } catch (e: any) {
      Alert.alert("Booking Failed", e.message);
    } finally {
      setBooking(false);
    }
  };

  const calculatedRating = useMemo(() => {
    if (reviews.length === 0) return doctor?.rating || 0;
    return parseFloat((reviews.reduce((a, b) => a + b.rating, 0) / reviews.length).toFixed(1));
  }, [reviews, doctor]);

  const scrollY = useRef(new Animated.Value(0)).current;

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color="#059669" /><Text style={styles.retryTxt}>Loading Luxury Profile...</Text></View>;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* TOP GLASS NAV */}
      <View style={styles.fixedHeaderNav}>
        <TouchableOpacity onPress={() => router.back()} style={styles.glassBtn}>
          <Ionicons name="chevron-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <TouchableOpacity onPress={() => router.push({ pathname: "/(patient)/messages", params: { doctorId, doctorName: doctor?.name } } as any)} style={styles.glassBtn}>
            <Ionicons name="chatbubble-ellipses-outline" size={22} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity onPress={toggleFollowDoctor} style={styles.glassBtn} disabled={following}>
            <Ionicons name={isFollowed ? "heart" : "heart-outline"} size={22} color={isFollowed ? "#EF4444" : "#fff"} />
          </TouchableOpacity>
        </View>
      </View>

      {/* LUXURY EMERALD HERO */}
      <View style={styles.magicHeader}>
        <LinearGradient colors={["#064E3B", "#065F46"]} style={StyleSheet.absoluteFill}>
          <View style={[styles.liquidBlob, { top: -40, right: -40, width: 250, height: 250, backgroundColor: '#059669', opacity: 0.15 }]} />
          <View style={[styles.liquidBlob, { bottom: -20, left: -20, width: 200, height: 200, backgroundColor: '#10B981', opacity: 0.1 }]} />

          <View style={styles.heroContent}>
            <View style={styles.avatarWrapper}>
              <View style={styles.avatarGlass}>
                <Image source={{ uri: doctor?.imageUrl || doctor?.photoUrl || "https://cdn-icons-png.flaticon.com/512/3774/3774299.png" }} style={styles.profileImg} />
              </View>
              {doctor?.isAvailable && <View style={styles.activePulse} />}
            </View>
            <View style={styles.heroText}>
              <View style={styles.specBadge}><Text style={styles.specBadgeText}>{doctor?.specialty?.toUpperCase()}</Text></View>
              <Text style={styles.heroName}>Dr. {doctor?.name}</Text>
              <View style={styles.ratingRow}>
                <Ionicons name="star" size={16} color="#FBBF24" />
                <Text style={styles.ratingText}>{calculatedRating}</Text>
                <View style={styles.ratingDot} />
                <Text style={styles.reviewCount}>{reviews.length} Reviews</Text>
              </View>
            </View>
          </View>
        </LinearGradient>
      </View>

      <Animated.ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingTop: 280, paddingBottom: 120 }}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { y: scrollY } } }], { useNativeDriver: false })}
      >
        <View style={styles.contentCard}>
          {/* STATS */}
          <View style={styles.statsBar}>
            <View style={styles.statBox}>
              <Ionicons name="ribbon" size={18} color="#059669" />
              <Text style={styles.statVal}>{doctor?.experience || 5}+ Yrs</Text>
              <Text style={styles.statLab}>Exp</Text>
            </View>
            <View style={styles.statLine} />
            <View style={styles.statBox}>
              <Ionicons name="wallet" size={18} color="#2563EB" />
              <Text style={styles.statVal}>${doctor?.consultationFee}</Text>
              <Text style={styles.statLab}>Fee</Text>
            </View>
            <View style={styles.statLine} />
            <View style={styles.statBox}>
              <Ionicons name="heart" size={18} color="#EF4444" />
              <Text style={styles.statVal}>{(doctor?.reviewCount || 0) + reviews.length}</Text>
              <Text style={styles.statLab}>Fans & Reviews</Text>
            </View>
          </View>

          {/* CLINIC ADDRESS - REQUESTED FEATURE */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Clinic Location</Text>
            <TouchableOpacity style={styles.addressCard}>
              <LinearGradient colors={["#F8FAFC", "#F1F5F9"]} style={styles.addressGradient}>
                <View style={styles.addressIconBox}>
                  <Ionicons name="location" size={24} color="#059669" />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.addressTitle}>Main Clinic Address</Text>
                  <Text style={styles.addressText} numberOfLines={2}>
                    {(doctor as any)?.location || (doctor as any)?.address || "123 Medical Plaza, Emerald District, Cairo"}
                  </Text>
                </View>
                <View style={styles.mapCircle}>
                  <Ionicons name="map-outline" size={18} color="#059669" />
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {/* BIO */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>About Doctor</Text>
            <View style={styles.bioCard}>
              <Text style={styles.bioText}>{doctor?.bio || "Highly experienced specialist providing patient-centered care with state-of-the-art medical technology."}</Text>
            </View>
          </View>

          {/* SCHEDULE */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Select Schedule</Text>
              <TouchableOpacity onPress={fetchAvailability}><Ionicons name="refresh" size={20} color="#059669" /></TouchableOpacity>
            </View>

            {loadingAvailability ? (
              <ActivityIndicator color="#059669" style={{ padding: 40 }} />
            ) : hasSchedule ? (
              <>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.daysScroll}>
                  {days.map((d, i) => (
                    <TouchableOpacity key={i} style={[styles.dayBtn, selectedDay === i && styles.dayBtnActive]} onPress={() => { setSelectedDay(i); setSelectedTime(null); }}>
                      <Text style={[styles.dayName, selectedDay === i && { color: '#fff' }]}>{d.label}</Text>
                      <Text style={[styles.dayNum, selectedDay === i && { color: '#fff' }]}>{d.date.split(' ')[0]}</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
                <View style={styles.slotsGrid}>
                  {currentSlots.length === 0 ? (
                    <View style={styles.emptySlots}><Text style={styles.emptySlotsText}>No slots available for today</Text></View>
                  ) : (
                    currentSlots.map(s => (
                      <TouchableOpacity key={s} style={[styles.slotBtn, selectedTime === s && styles.slotBtnActive]} onPress={() => setSelectedTime(s)}>
                        <Text style={[styles.slotTxt, selectedTime === s && { color: '#059669', fontWeight: '900' }]}>{s}</Text>
                      </TouchableOpacity>
                    ))
                  )}
                </View>
              </>
            ) : (
              <TouchableOpacity style={styles.notifyMeCard} onPress={handleNotifyMe}>
                <Ionicons name="notifications" size={32} color="#059669" />
                <Text style={styles.notifyTitle}>Notify Me</Text>
                <Text style={styles.notifySub}>Get an alert when Dr. {doctor?.name} updates their schedule.</Text>
                <View style={[styles.notifyToggle, notifyEnabled && { backgroundColor: '#059669' }]}>
                  <Text style={[styles.notifyToggleText, notifyEnabled && { color: '#fff' }]}>{notifyEnabled ? "Enabled" : "Enable Alerts"}</Text>
                </View>
              </TouchableOpacity>
            )}
          </View>

          {/* REVIEWS */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Reviews</Text>
              <TouchableOpacity onPress={() => setShowAddReview(true)}><Text style={styles.addRevTxt}>Add Review</Text></TouchableOpacity>
            </View>
            <View style={styles.reviewList}>
              {reviews.slice(0, 3).map((r, i) => (
                <View key={i} style={styles.reviewCard}>
                  <View style={styles.reviewHead}>
                    <View style={styles.revAvatar}><Text style={styles.revInitial}>{r.author[0]}</Text></View>
                    <View style={{ flex: 1 }}><Text style={styles.revName}>{r.author}</Text><StarRow value={r.rating} /></View>
                  </View>
                  <Text style={styles.revComment}>{r.comment}</Text>
                </View>
              ))}
            </View>
          </View>
        </View>
      </Animated.ScrollView>

      {/* BOTTOM BAR */}
      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.msgBtn} onPress={() => router.push({ pathname: "/(patient)/messages", params: { doctorId, doctorName: doctor?.name } } as any)}>
          <Ionicons name="chatbubble-ellipses" size={20} color="#059669" />
        </TouchableOpacity>
        <TouchableOpacity style={[styles.bookBtn, !selectedTime && { opacity: 0.6 }]} disabled={!selectedTime || booking} onPress={handleProceed}>
          <LinearGradient colors={["#059669", "#047857"]} style={styles.bookGradient}>
            <Text style={styles.payBtnText}>
              {booking ? "Processing..." : editAppointmentId ? "Update Appointment" : "Confirm Appointment"}
            </Text>
            <Ionicons name="arrow-forward" size={18} color="#fff" />
          </LinearGradient>
        </TouchableOpacity>
      </View>

      {/* MODALS */}
      <Modal visible={modalStep === "payment"} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            <View style={styles.modalIndicator} />
            <Text style={styles.modalTitle}>Payment Method</Text>
            <TouchableOpacity style={styles.payOption} onPress={() => { setPayMethod("visa"); setModalStep("visa_form"); }}>
              <Ionicons name="card" size={24} color="#2563EB" />
              <View style={{ flex: 1, marginLeft: 15 }}><Text style={styles.payOptionTitle}>Credit Card</Text><Text style={styles.payOptionSub}>Visa or Mastercard</Text></View>
              <Ionicons name="chevron-forward" size={20} color="#CBD5E1" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.payOption} onPress={() => confirmBooking("cash")}>
              <Ionicons name="cash" size={24} color="#059669" />
              <View style={{ flex: 1, marginLeft: 15 }}><Text style={styles.payOptionTitle}>Cash</Text><Text style={styles.payOptionSub}>Pay at clinic</Text></View>
              <Ionicons name="chevron-forward" size={20} color="#CBD5E1" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.closeModal} onPress={() => setModalStep(null)}><Text style={styles.closeModalText}>Cancel</Text></TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal visible={showSuccess} transparent animationType="fade">
        <View style={styles.successScreen}>
          <View style={styles.successIconBox}><Ionicons name="checkmark" size={80} color="#fff" /></View>
          <Text style={styles.successTitle}>Booking Successful!</Text>
          <Text style={styles.successDesc}>Your appointment with Dr. {doctor?.name} has been requested.</Text>
          <TouchableOpacity style={styles.successBtn} onPress={() => { setShowSuccess(false); router.back(); }}><Text style={styles.successBtnText}>Done</Text></TouchableOpacity>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  retryTxt: { color: "#64748B", marginTop: 10, fontWeight: '600' },

  fixedHeaderNav: { position: 'absolute', top: 50, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 20, zIndex: 1000 },
  glassBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)' },

  magicHeader: { position: 'absolute', top: 0, left: 0, right: 0, height: 320, zIndex: 0 },
  liquidBlob: { position: 'absolute', borderRadius: 150 },
  heroContent: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 24, paddingTop: 110 },
  avatarWrapper: { position: 'relative' },
  avatarGlass: { padding: 4, borderRadius: 58, backgroundColor: 'rgba(255,255,255,0.2)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)' },
  profileImg: { width: 104, height: 104, borderRadius: 52 },
  activePulse: { position: 'absolute', bottom: 6, right: 6, width: 18, height: 18, borderRadius: 9, backgroundColor: '#10B981', borderWidth: 3, borderColor: '#064E3B' },
  heroText: { marginLeft: 20, flex: 1 },
  specBadge: { alignSelf: 'flex-start', backgroundColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, marginBottom: 6 },
  specBadgeText: { color: '#fff', fontSize: 10, fontWeight: '800' },
  heroName: { fontSize: 26, fontWeight: '900', color: '#fff' },
  ratingRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  ratingText: { color: '#fff', fontSize: 14, fontWeight: '800', marginLeft: 4 },
  ratingDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: 'rgba(255,255,255,0.4)', marginHorizontal: 8 },
  reviewCount: { color: 'rgba(255,255,255,0.8)', fontSize: 13 },

  contentCard: { backgroundColor: '#F8FAFC', borderTopLeftRadius: 40, borderTopRightRadius: 40, paddingHorizontal: 20, paddingTop: 30, marginTop: -40, minHeight: 800 },
  statsBar: { flexDirection: 'row', backgroundColor: '#fff', borderRadius: 24, padding: 20, marginBottom: 30, elevation: 8, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15 },
  statBox: { flex: 1, alignItems: 'center' },
  statVal: { fontSize: 15, fontWeight: '900', color: '#1E293B', marginTop: 4 },
  statLab: { fontSize: 10, color: '#94A3B8', fontWeight: '800', textTransform: 'uppercase' },
  statLine: { width: 1, height: '60%', backgroundColor: '#F1F5F9', alignSelf: 'center' },

  section: { marginBottom: 30 },
  sectionTitle: { fontSize: 18, fontWeight: '900', color: '#1E293B', marginBottom: 15 },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },

  addressCard: { borderRadius: 24, overflow: 'hidden', elevation: 4, shadowColor: '#000', shadowOpacity: 0.05 },
  addressGradient: { flexDirection: 'row', alignItems: 'center', padding: 16, gap: 12 },
  addressIconBox: { width: 48, height: 48, borderRadius: 16, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', elevation: 2 },
  addressTitle: { fontSize: 12, color: '#94A3B8', fontWeight: '800' },
  addressText: { fontSize: 14, color: '#1E293B', fontWeight: '700', marginTop: 2 },
  mapCircle: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#ECFDF5', justifyContent: 'center', alignItems: 'center' },

  bioCard: { backgroundColor: '#fff', padding: 18, borderRadius: 24, borderWidth: 1, borderColor: '#F1F5F9' },
  bioText: { fontSize: 14, color: '#64748B', lineHeight: 22 },

  daysScroll: { marginBottom: 20 },
  dayBtn: { width: 64, height: 80, backgroundColor: '#fff', borderRadius: 20, justifyContent: 'center', alignItems: 'center', marginRight: 12, borderWidth: 1, borderColor: '#F1F5F9' },
  dayBtnActive: { backgroundColor: '#059669', borderColor: '#059669', elevation: 8 },
  dayName: { fontSize: 11, color: '#94A3B8', fontWeight: '700' },
  dayNum: { fontSize: 18, fontWeight: '900', color: '#1E293B', marginTop: 4 },

  slotsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  slotBtn: { paddingHorizontal: 12, paddingVertical: 12, borderRadius: 14, backgroundColor: '#fff', borderWidth: 1, borderColor: '#F1F5F9', flex: 1, minWidth: '30%', alignItems: 'center' },
  slotBtnActive: { backgroundColor: '#ECFDF5', borderColor: '#059669' },
  slotTxt: { fontSize: 13, fontWeight: '700', color: '#64748B' },
  emptySlots: { flex: 1, padding: 30, alignItems: 'center', backgroundColor: '#F1F5F9', borderRadius: 20, borderStyle: 'dashed', borderWidth: 1, borderColor: '#CBD5E1' },
  emptySlotsText: { color: '#94A3B8', fontSize: 13 },

  notifyMeCard: { backgroundColor: '#F0FDF4', padding: 25, borderRadius: 24, alignItems: 'center', borderWidth: 1, borderColor: '#DCFCE7' },
  notifyTitle: { fontSize: 18, fontWeight: '900', color: '#065F46', marginTop: 10 },
  notifySub: { fontSize: 13, color: '#059669', textAlign: 'center', marginVertical: 10, opacity: 0.7 },
  notifyToggle: { backgroundColor: '#fff', paddingHorizontal: 20, paddingVertical: 10, borderRadius: 12, borderWidth: 1, borderColor: '#059669' },
  notifyToggleText: { color: '#059669', fontWeight: '800' },

  reviewCard: { backgroundColor: '#fff', padding: 16, borderRadius: 24, marginBottom: 12, borderWidth: 1, borderColor: '#F1F5F9' },
  reviewHead: { flexDirection: 'row', gap: 12, marginBottom: 10 },
  revAvatar: { width: 36, height: 36, borderRadius: 12, backgroundColor: '#F1F5F9', justifyContent: 'center', alignItems: 'center' },
  revInitial: { color: '#059669', fontWeight: '900' },
  revName: { fontSize: 14, fontWeight: '800', color: '#1E293B' },
  revComment: { fontSize: 13, color: '#64748B', lineHeight: 18 },
  addRevTxt: { color: '#059669', fontWeight: '800', fontSize: 13 },

  bottomBar: { position: 'absolute', bottom: 0, left: 0, right: 0, backgroundColor: '#fff', padding: 20, paddingBottom: Platform.OS === 'ios' ? 35 : 20, flexDirection: 'row', gap: 12, borderTopLeftRadius: 30, borderTopRightRadius: 30, elevation: 20 },
  msgBtn: { width: 56, height: 56, borderRadius: 18, backgroundColor: '#F0FDF4', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#059669' },
  bookBtn: { flex: 1, borderRadius: 18, overflow: 'hidden' },
  bookGradient: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10 },
  bookBtnText: { color: '#fff', fontSize: 16, fontWeight: '900' },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  modalSheet: { backgroundColor: '#fff', borderTopLeftRadius: 30, borderTopRightRadius: 30, padding: 24 },
  modalIndicator: { width: 40, height: 5, backgroundColor: '#F1F5F9', borderRadius: 5, alignSelf: 'center', marginBottom: 20 },
  modalTitle: { fontSize: 20, fontWeight: '900', color: '#1E293B', marginBottom: 20 },
  payOption: { flexDirection: 'row', alignItems: 'center', padding: 16, backgroundColor: '#F8FAFC', borderRadius: 20, marginBottom: 12 },
  payOptionTitle: { fontSize: 16, fontWeight: '800', color: '#1E293B' },
  payOptionSub: { fontSize: 12, color: '#94A3B8' },
  closeModal: { padding: 15, alignItems: 'center' },
  closeModalText: { color: '#64748B', fontWeight: '700' },

  successScreen: { flex: 1, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', padding: 30 },
  successIconBox: { width: 120, height: 120, borderRadius: 60, backgroundColor: '#059669', justifyContent: 'center', alignItems: 'center', marginBottom: 30 },
  successTitle: { fontSize: 24, fontWeight: '900', color: '#1E293B' },
  successDesc: { fontSize: 15, color: '#64748B', textAlign: 'center', marginVertical: 15 },
  successBtn: { backgroundColor: '#059669', paddingHorizontal: 50, paddingVertical: 15, borderRadius: 20, marginTop: 20 },
  successBtnText: { color: '#fff', fontWeight: '900' },
  reviewList: { gap: 12 },
});
