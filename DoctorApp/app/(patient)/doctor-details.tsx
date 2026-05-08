import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Image,
  ActivityIndicator, Platform, StatusBar, Modal, TextInput,
  KeyboardAvoidingView, Animated, Alert
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { COLORS } from "../../constants/colors";
import { getDoctorById, getReviewsByDoctor, addReview, updateMyReview, deleteMyReview, DoctorDetails, Review } from "../../services/doctorService";
import { bookAppointment } from "../../services/appointmentService";
import { apiFetch } from "../../services/http";
import { BASE_URL } from "../../constants/api";
import { startSignalRConnection, onDoctorUpdated, onScheduleReady, onScheduleUpdated, onNewConsultation, onNewMedication, subscribeToDoctorSchedule } from "../../services/signalr";
import RatingStars from "../../components/RatingStars";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { addNotification } from "../../services/notificationService";

import { checkIfFollowed, setFollowed, toggleFollowed, checkIfSubscribed, setSubscribed } from "../../services/followService";

const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

const normalizeText = (value: unknown): string => {
  const text = value?.toString?.()
  return typeof text === "string" ? text.trim().toLowerCase() : ""
}

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
  if (isoMatch) {
    return `${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}`
  }

  const numericMatch = text.match(/^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})$/)
  if (numericMatch) {
    const first = Number(numericMatch[1])
    const second = Number(numericMatch[2])
    const year = Number(numericMatch[3])
    if (Number.isFinite(first) && Number.isFinite(second) && Number.isFinite(year)) {
      const dayFirst = first > 12 || second <= 12
      const day = dayFirst ? first : second
      const month = dayFirst ? second : first
      if (day >= 1 && day <= 31 && month >= 1 && month <= 12) {
        return `${year.toString().padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`
      }
    }
  }

  const parsed = new Date(text)
  if (!Number.isNaN(parsed.getTime())) {
    return toLocalIsoDate(parsed)
  }

  return ""
}

const getDayIndexFromLocalIsoDate = (isoDate: string): number => {
  const [year, month, day] = isoDate.split("-").map(Number)
  if (
    !Number.isFinite(year) ||
    !Number.isFinite(month) ||
    !Number.isFinite(day)
  ) {
    return new Date(isoDate).getDay()
  }
  return new Date(year, month - 1, day).getDay()
}

const normalizeToDisplayDate = (isoDate: string): string => {
  if (!isoDate) return ""
  const [y, m, d] = isoDate.split("-").map(Number)
  const date = new Date(y, m - 1, d)
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
  return `${d} ${monthNames[date.getMonth()]}`
}

const normalizeBoolean = (value: unknown): boolean => {
  if (typeof value === "boolean") return value
  if (typeof value === "number") return value === 1
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    return normalized === "true" || normalized === "1" || normalized === "yes"
  }
  return false
}

const getNormalizedDayName = (value: unknown): string => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    const index = ((value % 7) + 7) % 7
    return DAY_NAMES[index].toLowerCase()
  }

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    const numeric = Number(trimmed)
    if (!Number.isNaN(numeric)) {
      const index = ((numeric % 7) + 7) % 7
      return DAY_NAMES[index].toLowerCase()
    }
    return trimmed.toLowerCase()
  }

  return ''
}

const isDayEnabled = (day: any): boolean => normalizeBoolean(day?.isAvailable ?? day?.IsAvailable)

const parseTimeToMinutes = (value: unknown): number | null => {
  const text = value?.toString?.().trim() ?? ""
  if (!text) return null

  const amPmMatch = text.match(/^(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)$/i)
  if (amPmMatch) {
    let hours = Number(amPmMatch[1])
    const minutes = Number(amPmMatch[2])
    const marker = amPmMatch[3].toUpperCase()
    if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null
    if (hours < 1 || hours > 12 || minutes < 0 || minutes > 59) return null
    if (marker === "PM" && hours !== 12) hours += 12
    if (marker === "AM" && hours === 12) hours = 0
    return hours * 60 + minutes
  }

  const hhmmMatch = text.match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/)
  if (!hhmmMatch) return null

  const hours = Number(hhmmMatch[1])
  const minutes = Number(hhmmMatch[2])
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null
  if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) return null
  return hours * 60 + minutes
}

const slotToKey = (value: unknown): string => {
  const minutes = parseTimeToMinutes(value)
  if (minutes != null) return `m-${minutes}`
  const fallback = value?.toString?.().trim().toLowerCase() ?? ""
  return fallback ? `t-${fallback}` : ""
}

const toDisplaySlot = (raw: unknown): string => {
  const text = raw?.toString?.() ?? ""
  if (!text) return ""

  if (text.includes("AM") || text.includes("PM")) {
    return text
  }

  const minutes = parseTimeToMinutes(text)
  if (minutes == null) return text

  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  const ampm = h >= 12 ? "PM" : "AM"
  const h12 = h % 12 || 12
  return `${h12.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")} ${ampm}`
}

const getTimeSlotsForDay = (date: string, availability: any[], bookedSlots: any[] = []): string[] => {
  if (!date || !availability || availability.length === 0) {
    return []
  }

  const dayIndex = getDayIndexFromLocalIsoDate(date)
  const dayOfWeek = DAY_NAMES[dayIndex]

  const dayAvail = availability.find(a => {
    const rawDow = a.dayName ?? a.DayName ?? a.dayOfWeek ?? a.DayOfWeek ?? a.day ?? a.Day ?? ''
    const normalizedDay = getNormalizedDayName(rawDow)
    return normalizedDay === dayOfWeek.toLowerCase() && isDayEnabled(a)
  })

  if (!dayAvail) return []

  const isBookedForDateTime = (slotValue: unknown): boolean => {
    const slotKey = slotToKey(slotValue)
    if (!slotKey) return false

    return bookedSlots.some((bs) => {
      const bookedDate = normalizeToLocalIsoDate(bs?.date ?? bs?.Date ?? "")
      if (!bookedDate || bookedDate !== date) return false
      const bookedKey = slotToKey(bs?.time ?? bs?.Time ?? "")
      return bookedKey === slotKey
    })
  }

  const rawBackendSlots = dayAvail.timeSlots ?? dayAvail.TimeSlots
  if (Array.isArray(rawBackendSlots) && rawBackendSlots.length > 0) {
    return rawBackendSlots
      .map((slot: unknown) => toDisplaySlot(slot))
      .filter((slot: string) => {
        if (!slot) return false
        if (isBookedForDateTime(slot)) return false

        // Check if slot is in the past (today only)
        const isToday = date === toLocalIsoDate(new Date())
        if (isToday) {
          const slotMins = parseTimeToMinutes(slot)
          const nowMins = new Date().getHours() * 60 + new Date().getMinutes()
          if (slotMins != null && slotMins <= nowMins) {
            return false
          }
        }
        return true
      })
  }

  const startTimeRaw = dayAvail.startTime ?? dayAvail.StartTime ?? '09:00'
  const endTimeRaw = dayAvail.endTime ?? dayAvail.EndTime ?? '17:00'
  const slotDurationRaw = Number(dayAvail.slotDurationMinutes ?? dayAvail.SlotDurationMinutes ?? 30)
  const slotDuration = Number.isFinite(slotDurationRaw) && slotDurationRaw > 0 ? slotDurationRaw : 30

  const slots: string[] = []
  const startMinutes = parseTimeToMinutes(startTimeRaw)
  const endMinutes = parseTimeToMinutes(endTimeRaw)

  if (startMinutes == null || endMinutes == null || endMinutes < startMinutes) {
    return []
  }

  let currentMinutes = startMinutes

  while (currentMinutes <= endMinutes) {
    const h = Math.floor(currentMinutes / 60)
    const m = currentMinutes % 60
    const ampm = h >= 12 ? 'PM' : 'AM'
    const h12 = h % 12 || 12
    const slotStr = `${h12.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')} ${ampm}`

    if (!isBookedForDateTime(slotStr)) {
      const isToday = date === toLocalIsoDate(new Date())
      const nowMins = new Date().getHours() * 60 + new Date().getMinutes()
      if (!isToday || currentMinutes > nowMins) {
        slots.push(slotStr)
      }
    }
    currentMinutes += slotDuration
  }

  return slots
}

function getNextAvailableDays(availability: any[], count: number) {
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const result: { label: string; date: string; isoDate: string; dateObj: Date }[] = []
  const today = new Date()

  for (let offset = 0; offset < 30 && result.length < count; offset += 1) {
    const d = new Date(today)
    d.setDate(today.getDate() + offset)
    const dayName = DAY_NAMES[d.getDay()].toLowerCase()
    const hasAvailability = availability.some((a) => {
      const rawDow = a.dayName ?? a.DayName ?? a.dayOfWeek ?? a.DayOfWeek ?? a.day ?? a.Day ?? ''
      const normalizedDay = getNormalizedDayName(rawDow)
      return normalizedDay === dayName && isDayEnabled(a)
    })

    if (!hasAvailability) continue

    const iso = toLocalIsoDate(d)
    result.push({
      label: offset === 0 ? "Today" : DAY_NAMES[d.getDay()].slice(0, 3),
      date: `${d.getDate()} ${monthNames[d.getMonth()]}`,
      isoDate: iso,
      dateObj: d,
    })
  }

  return result
}

function getNextDays(count: number) {
  const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return Array.from({ length: count }, (_, i) => {
    const d = new Date(); d.setDate(d.getDate() + i);
    const iso = toLocalIsoDate(d);
    return {
      label: i === 0 ? "Today" : dayNames[d.getDay()],
      date: `${d.getDate()} ${monthNames[d.getMonth()]}`,
      isoDate: iso,
      dateObj: d
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

function StarRow({ value, onChange, size = 28 }: { value: number; onChange?: (n: number) => void; size?: number }) {
  return (
    <View style={{ flexDirection: "row", gap: 4 }}>
      {[1, 2, 3, 4, 5].map((n) => {
        const isFull = n <= Math.floor(value);
        const isHalf = !isFull && n <= Math.ceil(value) && (value % 1 >= 0.3 && value % 1 <= 0.8);
        return (
          <TouchableOpacity key={n} onPress={() => onChange?.(n)} disabled={!onChange}>
            <Ionicons
              name={isFull ? "star" : (isHalf ? "star-half" : "star-outline")}
              size={size} color={(isFull || isHalf) ? "#FFB300" : "#CCC"}
            />
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

export default function DoctorDetailsScreen() {
  const { doctorId } = useLocalSearchParams<{ doctorId: string }>();
  const router = useRouter();

  const [doctor, setDoctor] = useState<DoctorDetails | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedDay, setSelectedDay] = useState(0);
  const [selectedTime, setSelectedTime] = useState<string | null>(null);
  const [payMethod, setPayMethod] = useState<PayMethod>(null);
  const [booking, setBooking] = useState(false);
  const [modalStep, setModalStep] = useState<ModalStep>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  const [availability, setAvailability] = useState<any[]>([])
  const [bookedSlots, setBookedSlots] = useState<any[]>([])
  const [hasSchedule, setHasSchedule] = useState(false)
  const [notifyEnabled, setNotifyEnabled] = useState(false)
  const [loadingAvailability, setLoadingAvailability] = useState(true)

  // Visa form
  const [cardName, setCardName] = useState("");
  const [cardNumber, setCardNumber] = useState("");
  const [expiry, setExpiry] = useState("");
  const [cvv, setCvv] = useState("");
  const [cardErrors, setCardErrors] = useState<any>({});

  // Review
  const [showAddReview, setShowAddReview] = useState(false);
  const [myRating, setMyRating] = useState(0);
  const [myComment, setMyComment] = useState("");
  const [savingReview, setSavingReview] = useState(false);
  const [myDisplayName, setMyDisplayName] = useState("");

  // Follow
  const [isFollowed, setIsFollowed] = useState(false);
  const [following, setFollowing] = useState(false);

  const days = useMemo(() => {
    if (!hasSchedule) return getNextDays(7)
    return getNextAvailableDays(availability, 7)
  }, [availability, hasSchedule])

  const selectedDayData = useMemo(() => days[selectedDay] || null, [days, selectedDay]);

  const currentSlots = useMemo(() => {
    if (!selectedDayData) return []
    return getTimeSlotsForDay(selectedDayData.isoDate, availability, bookedSlots)
  }, [selectedDayData, availability, bookedSlots])

  useEffect(() => {
    if (selectedDay >= days.length) {
      setSelectedDay(0)
      setSelectedTime(null)
    }
  }, [days.length, selectedDay])

  useEffect(() => {
    if (!doctorId) return;
    fetchAll();
  }, [doctorId]);

  useEffect(() => {
    const id = Number(doctorId)
    checkIfFollowed(id)
      .then((followed) => {
        setIsFollowed(followed)
        setNotifyEnabled(followed)
      })
      .catch(() => {
        setIsFollowed(false)
        setNotifyEnabled(false)
      })
  }, [doctorId]);

  const fetchAll = async () => {
    try {
      setLoading(true);
      setError("");
      const [doc, revs] = await Promise.all([
        getDoctorById(doctorId!),
        getReviewsByDoctor(doctorId!),
      ]);
      const currentName = await AsyncStorage.getItem("userName");
      setDoctor(doc);
      setReviews(revs);
      setMyDisplayName((currentName ?? "").trim());
      await Promise.all([
        fetchAvailability(),
        refreshFollowState()
      ]);
    } catch (e: any) {
      setError(e.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  const checkFollowStatus = async (targetDoctorId: number): Promise<boolean> => {
    return await checkIfFollowed(targetDoctorId)
  }

  const refreshFollowState = async () => {
    const id = Number(doctorId)
    if (!Number.isFinite(id) || id <= 0) {
      setIsFollowed(false)
      setNotifyEnabled(false)
      return
    }

    const [followed, subscribed] = await Promise.all([
      checkIfFollowed(id),
      checkIfSubscribed(id)
    ])
    setIsFollowed(followed)
    setNotifyEnabled(followed || subscribed)
  }

  const toggleFollow = async (targetDoctorId: number) => {
    if (!Number.isFinite(targetDoctorId) || targetDoctorId <= 0) return

    setFollowing(true)
    try {
      const followed = await checkIfFollowed(targetDoctorId)
      const next = !followed
      await setFollowed(targetDoctorId, next)
      setIsFollowed(next)

      const subscribed = await checkIfSubscribed(targetDoctorId)
      setNotifyEnabled(next || subscribed)

      if (next) {
        await startSignalRConnection()
        await subscribeToDoctorSchedule(targetDoctorId)
        Alert.alert("✅ Following!", "You will be notified of schedule changes.")
      }
    } catch {
      Alert.alert("Error", "Failed to update followed list.")
    } finally {
      setFollowing(false)
    }
  }

  const handleNotifyMe = async () => {
    const id = Number(doctorId)
    if (!Number.isFinite(id) || id <= 0) return

    try {
      const currentlyEnabled = notifyEnabled
      const next = !currentlyEnabled

      await setSubscribed(id, next)
      setNotifyEnabled(next || isFollowed)

      if (next) {
        await apiFetch(`${BASE_URL}/api/doctors/${doctorId}/notify-schedule`, { method: 'POST' }, true)
        await startSignalRConnection()
        await subscribeToDoctorSchedule(id)
        Alert.alert("🔔 Notifications On", "You will be notified when Dr. " + (doctor?.name || "the doctor") + " updates their schedule.")
      } else {
        Alert.alert("🔕 Notifications Off", "You will no longer receive schedule alerts unless you follow this doctor.")
      }
    } catch {
      Alert.alert('Error', 'Failed to update notification settings.')
    }
  }

  const fetchAvailability = async () => {
    try {
      setLoadingAvailability(true)
      const data = await apiFetch<any>(
        `${BASE_URL}/api/doctors/${doctorId}/availability`,
        { method: 'GET', allowedStatusCodes: [404] },
        false
      )
      const availData = data?.days ?? data?.Days ?? (Array.isArray(data) ? data : [])
      const bookedData = data?.bookedSlots ?? data?.BookedSlots ?? []

      setBookedSlots(Array.isArray(bookedData) ? bookedData : [])
      if (availData.length > 0) {
        setAvailability(availData)
        const hasAny = availData.some((a: any) => {
          const daySlots = a?.timeSlots ?? a?.TimeSlots
          const hasWindow =
            parseTimeToMinutes(a?.startTime ?? a?.StartTime) != null &&
            parseTimeToMinutes(a?.endTime ?? a?.EndTime) != null
          const dayEnabled = isDayEnabled(a)
          return dayEnabled && ((Array.isArray(daySlots) && daySlots.length > 0) || hasWindow)
        })
        setHasSchedule(hasAny)
      } else {
        setAvailability([])
        setHasSchedule(false)
      }
    } catch (err) {
      console.error('Failed to fetch availability:', err)
      setAvailability([])
      setBookedSlots([])
      setHasSchedule(false)
    } finally {
      setLoadingAvailability(false)
    }
  }

  useEffect(() => {
    let unsubReady: (() => void) | undefined
    let unsubUpdated: (() => void) | undefined
    let mounted = true

    const bindScheduleUpdates = async () => {
      await startSignalRConnection()
      if (!mounted) return

      const handleScheduleEvent = async (data: any, isUpdatedEvent: boolean) => {
        const payload = data ?? {}
        const eventDoctorId = Number(payload?.doctorId ?? payload?.DoctorId)
        const eventDoctorName = String(payload?.doctorName ?? payload?.DoctorName ?? "Doctor")
        if (eventDoctorId === Number(doctorId)) {
          const followed = await checkFollowStatus(eventDoctorId)
          if (followed) {
            fetchAvailability()
            Alert.alert(
              isUpdatedEvent ? "Schedule Updated" : "Schedule Ready",
              isUpdatedEvent
                ? `Dr. ${eventDoctorName}'s schedule has been updated.`
                : `Dr. ${eventDoctorName}'s schedule is now available!`
            )
          }
        }
      }

      unsubReady = onScheduleReady((data) => {
        handleScheduleEvent(data, false).catch(() => undefined)
      })

      unsubUpdated = onScheduleUpdated((data) => {
        handleScheduleEvent(data, true).catch(() => undefined)
      })

      try {
        const id = Number(doctorId)
        if (Number.isFinite(id) && id > 0) {
          const followed = await checkFollowStatus(id)
          if (followed) {
            await subscribeToDoctorSchedule(id)
          }
        }
      } catch {
        // ignore
      }
    }

    bindScheduleUpdates()

    return () => {
      mounted = false
      unsubReady?.()
      unsubUpdated?.()
    }
  }, [doctorId])

  const validateCard = () => {
    const errs: any = {};
    if (!cardName.trim()) errs.cardName = "Name is required";
    if (cardNumber.replace(/\s/g, "").length < 16) errs.cardNumber = "Enter full 16-digit card number";
    if (expiry.length < 5) errs.expiry = "Enter valid expiry MM/YY";
    if (cvv.length < 3) errs.cvv = "Enter valid CVV";
    setCardErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleProceed = () => {
    if (!selectedDayData) { Alert.alert("No available date", "This doctor has no selectable dates yet."); return; }
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
    if (!doctor || !selectedTime || !selectedDayData) return;
    setBooking(true);
    try {
      const newAppointment = await bookAppointment({
        doctorId: doctor.id,
        date: selectedDayData.isoDate,
        time: selectedTime,
        paymentMethod: method as "visa" | "cash",
      });

      await addNotification({
        id: `booking_${newAppointment.id ?? Date.now()}`,
        type: "confirmed",
        title: "✅ Booking Received!",
        message: `Your appointment request has been sent to Dr. ${doctor.name}. Waiting for doctor approval.`,
        timestamp: Date.now(),
        doctorId: doctor.id,
        doctorName: doctor.name,
        appointmentId: Number(newAppointment.id),
        icon: "✅",
      });

      Alert.alert(
        "✅ Request Sent!",
        `Your appointment with Dr. ${doctor.name} is pending doctor approval. You will be notified when confirmed.`,
        [{ text: "OK" }]
      );

      setPayMethod(method);
      setShowSuccess(true);
      await fetchAvailability();
    } catch (e: any) {
      Alert.alert("Booking Failed", e.message);
    } finally {
      setBooking(false);
    }
  };

  const submitReview = async () => {
    if (myRating === 0) { Alert.alert("Rating required", "Please select a star rating."); return; }
    if (!myComment.trim()) { Alert.alert("Comment required", "Please write a comment."); return; }
    setSavingReview(true);
    try {
      await addReview(Number(doctorId), myRating, myComment.trim(), myDisplayName || "Anonymous")
      await fetchAll();
      setMyRating(0); setMyComment(""); setShowAddReview(false);
    } catch (e: any) {
      Alert.alert("Error", e.message);
    } finally {
      setSavingReview(false);
    }
  };

  const handleUpdateExistingReview = async () => {
    if (myRating === 0 || !myComment.trim()) return;
    setSavingReview(true);
    try {
      await updateMyReview(Number(doctorId), myRating, myComment.trim(), myExistingReview?.id);
      await fetchAll();
      setShowAddReview(false);
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? "Failed to update review.");
    } finally {
      setSavingReview(false);
    }
  };

  const handleDeleteExistingReview = async () => {
    setSavingReview(true);
    try {
      await deleteMyReview(Number(doctorId), myExistingReview?.id);
      await fetchAll();
      setMyRating(0); setMyComment(""); setShowAddReview(false);
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? "Failed to delete review.");
    } finally {
      setSavingReview(false);
    }
  };

  const calculatedRating = useMemo(() => {
    if (reviews.length === 0) return doctor?.rating || 0;
    const sum = reviews.reduce((acc, r) => acc + r.rating, 0);
    return parseFloat((sum / reviews.length).toFixed(1));
  }, [reviews, doctor?.rating]);

  const myExistingReview = useMemo(() => {
    const byMineFlag = reviews.find((review) => review.isMine === true)
    if (byMineFlag) return byMineFlag
    const myName = normalizeText(myDisplayName)
    if (!myName) return undefined
    return reviews.find((review) => normalizeText(review.author) === myName)
  }, [reviews, myDisplayName])

  const scrollY = useRef(new Animated.Value(0)).current;

  const headerHeight = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [280, 200],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 150],
    outputRange: [1, 0.9],
    extrapolate: 'clamp',
  });

  const avatarScale = scrollY.interpolate({
    inputRange: [0, 100],
    outputRange: [1, 0.8],
    extrapolate: 'clamp',
  });

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#059669" />
        <Text style={styles.retryTxt}>Loading Doctor Details...</Text>
      </View>
    );
  }

  if (error || !doctor) {
    return (
      <View style={styles.center}>
        <Ionicons name="alert-circle-outline" size={64} color="#EF4444" />
        <Text style={styles.errorTxt}>{error || "Doctor not found"}</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={fetchAll}>
          <Text style={styles.retryTxt}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* TOP NAVIGATION BUTTONS (ALWAYS ON TOP) */}
      <View style={styles.fixedHeaderNav}>
        <TouchableOpacity onPress={() => router.back()} style={styles.glassBtn}>
          <Ionicons name="chevron-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <TouchableOpacity 
            onPress={() => router.push({ pathname: "/(patient)/messages", params: { doctorId: doctorId, doctorName: doctor?.name } } as any)} 
            style={styles.glassBtn}
          >
            <Ionicons name="chatbubble-ellipses-outline" size={22} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => toggleFollow(Number(doctorId))} style={styles.glassBtn} disabled={following}>
            <Ionicons name={isFollowed ? "heart" : "heart-outline"} size={22} color={isFollowed ? "#EF4444" : "#fff"} />
          </TouchableOpacity>
        </View>
      </View>

      {/* FIXED GREEN BACKGROUND */}
      <View style={styles.magicHeader}>
        <LinearGradient 
          colors={["#064E3B", "#065F46"]} 
          start={{ x: 0, y: 0 }} 
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFill}
        >
          <View style={[styles.liquidBlob, { top: -60, right: -60, width: 300, height: 300, backgroundColor: '#059669', opacity: 0.15 }]} />
          <View style={[styles.liquidBlob, { bottom: -40, left: -40, width: 250, height: 250, backgroundColor: '#10B981', opacity: 0.1 }]} />

          <View style={styles.doctorHeroSection}>
            <Animated.View style={[styles.profileImgContainer, { transform: [{ scale: avatarScale }] }]}>
              <View style={styles.avatarGlassBorder}>
                <Image
                  source={{ uri: doctor?.imageUrl || doctor?.photoUrl || "https://cdn-icons-png.flaticon.com/512/3774/3774299.png" }}
                  style={styles.profileImg}
                />
              </View>
              {doctor?.isAvailable && <View style={styles.activePulse} />}
            </Animated.View>

            <View style={styles.heroInfo}>
              <View style={styles.specBadge}>
                <Text style={styles.specBadgeText}>{doctor?.specialty?.toUpperCase()}</Text>
              </View>
              <Text style={styles.heroName}>Dr. {doctor?.name}</Text>
              <View style={styles.heroRatingRow}>
                <Ionicons name="star" size={16} color="#FDE047" />
                <Text style={styles.heroRatingText}>{calculatedRating}</Text>
                <View style={styles.ratingDot} />
                <Text style={styles.heroReviewCount}>{reviews.length || doctor?.reviewCount || 0} Reviews</Text>
              </View>
            </View>
          </View>
        </LinearGradient>
      </View>

      <Animated.ScrollView
        showsVerticalScrollIndicator={false}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        scrollEventThrottle={16}
        contentContainerStyle={{ paddingTop: 300, paddingBottom: 120 }}
      >
        <View style={styles.contentCard}>
          {/* Stats Bar */}
          <View style={styles.statsContainer}>
            <View style={styles.statItem}>
              <View style={[styles.statIconWrap, { backgroundColor: '#F0FDF4' }]}>
                <Ionicons name="ribbon-outline" size={20} color="#059669" />
              </View>
              <Text style={styles.statLabel}>Experience</Text>
              <Text style={styles.statValue}>{doctor?.experience || (doctor as any)?.yearsExperience || 0} Yrs</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <View style={[styles.statIconWrap, { backgroundColor: '#EFF6FF' }]}>
                <Ionicons name="wallet-outline" size={20} color="#2563EB" />
              </View>
              <Text style={styles.statLabel}>Consultation</Text>
              <Text style={styles.statValue}>${doctor?.consultationFee}</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <View style={[styles.statIconWrap, { backgroundColor: '#FFF7ED' }]}>
                <Ionicons name="chatbubble-ellipses-outline" size={20} color="#EA580C" />
              </View>
              <Text style={styles.statLabel}>Reviews</Text>
              <Text style={styles.statValue}>{doctor?.reviewCount || 0}</Text>
            </View>
          </View>

          {/* Bio Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>About Doctor</Text>
            <View style={styles.bioCard}>
              <Text style={styles.bioText}>
                {doctor?.bio || "Dr. " + doctor?.name + " is a highly skilled " + doctor?.specialty + " dedicated to providing exceptional care to all patients."}
              </Text>
            </View>
          </View>

          {/* Booking / Schedule Section */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Select Schedule</Text>
              <TouchableOpacity onPress={fetchAvailability}>
                <Ionicons name="refresh-circle-outline" size={24} color="#059669" />
              </TouchableOpacity>
            </View>

            {loadingAvailability ? (
              <View style={styles.loaderContainer}>
                <ActivityIndicator color="#059669" size="large" />
              </View>
            ) : hasSchedule ? (
              <>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.daysScroll}>
                  {days.map((d, i) => (
                    <TouchableOpacity
                      key={i}
                      activeOpacity={0.8}
                      style={[styles.dayButton, selectedDay === i && styles.dayButtonActive]}
                      onPress={() => { setSelectedDay(i); setSelectedTime(null); }}
                    >
                      <Text style={[styles.dayName, selectedDay === i && styles.dayTextActive]}>{d.label}</Text>
                      <Text style={[styles.dayNum, selectedDay === i && styles.dayTextActive]}>{d.date.split(' ')[0]}</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>

                <View style={styles.slotsContainer}>
                  {currentSlots.length === 0 ? (
                    <View style={styles.emptyState}>
                      <Ionicons name="calendar-outline" size={40} color="#CBD5E1" />
                      <Text style={styles.emptyStateText}>No available slots for this day</Text>
                    </View>
                  ) : (
                    <View style={styles.slotsGrid}>
                      {currentSlots.map((s) => (
                        <TouchableOpacity
                          key={s}
                          activeOpacity={0.7}
                          style={[styles.slotCard, selectedTime === s && styles.slotCardActive]}
                          onPress={() => setSelectedTime(s)}
                        >
                          <Text style={[styles.slotTime, selectedTime === s && styles.slotTimeActive]}>{s}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  )}
                </View>
              </>
            ) : (
              <View style={styles.waitingCard}>
                <LinearGradient colors={["#F0FDF4", "#DCFCE7"]} style={styles.waitingGradient}>
                  <Ionicons name="notifications-outline" size={40} color="#059669" />
                  <Text style={styles.waitingTitle}>Schedule Pending</Text>
                  <Text style={styles.waitingSub}>Be the first to know when Dr. {doctor?.name} opens their schedule.</Text>
                  <TouchableOpacity
                    style={[styles.notifyAction, notifyEnabled && styles.notifyActionActive]}
                    onPress={handleNotifyMe}
                  >
                    <Text style={[styles.notifyActionText, notifyEnabled && { color: "#fff" }]}>
                      {notifyEnabled ? '🔔 Already Notified' : 'Notify Me'}
                    </Text>
                  </TouchableOpacity>
                </LinearGradient>
              </View>
            )}
          </View>

          {/* Reviews Section */}
          <View style={[styles.section, { marginBottom: 20 }]}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Patient Reviews</Text>
              <TouchableOpacity style={styles.addReviewLink} onPress={() => setShowAddReview(true)}>
                <Text style={styles.addReviewLinkText}>{myExistingReview ? "Edit My Review" : "Add Review"}</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.reviewsList}>
              {reviews.length === 0 ? (
                <View style={styles.emptyState}>
                  <Ionicons name="chatbubbles-outline" size={40} color="#CBD5E1" />
                  <Text style={styles.emptyStateText}>No reviews yet. Be the first to share your experience!</Text>
                </View>
              ) : (
                reviews.slice(0, 3).map((r, i) => (
                  <View key={i} style={styles.reviewItem}>
                    <View style={styles.reviewHeaderRow}>
                      <View style={styles.reviewerAvatar}>
                        <Text style={styles.avatarInitial}>{r.author.charAt(0)}</Text>
                      </View>
                      <View style={{ flex: 1 }}>
                        <Text style={styles.reviewerName}>{r.author}</Text>
                        <View style={styles.starsContainer}>
                          <StarRow value={r.rating} size={12} />
                        </View>
                      </View>
                      <Text style={styles.reviewDateText}>{r.date ? new Date(r.date).toLocaleDateString() : ""}</Text>
                    </View>
                    <Text style={styles.reviewContent}>{r.comment}</Text>
                  </View>
                ))
              )}
            </View>
          </View>
        </View>
      </Animated.ScrollView>

      {/* FLOATING ACTION BUTTON */}
      <View style={styles.bottomBar}>
        <TouchableOpacity 
          style={styles.messageTextBtn}
          onPress={() => router.push({ pathname: "/(patient)/messages", params: { doctorId: doctorId, doctorName: doctor?.name } } as any)}
        >
          <Ionicons name="chatbubble-ellipses" size={20} color="#059669" />
          <Text style={styles.messageBtnText}>Message</Text>
        </TouchableOpacity>

        <TouchableOpacity
          activeOpacity={0.8}
          style={[styles.premiumBookBtn, (!selectedTime || !selectedDayData) && styles.disabledBookBtn]}
          onPress={handleProceed}
          disabled={booking || !selectedTime || !selectedDayData}
        >
          <LinearGradient
            colors={(!selectedTime || !selectedDayData) ? ["#CBD5E1", "#94A3B8"] : ["#059669", "#047857"]}
            style={styles.bookGradient}
          >
            <Text style={styles.bookText}>
              {booking ? "Wait..." : "Book Now"}
            </Text>
            <View style={styles.bookIconCircle}>
              <Ionicons name="chevron-forward" size={16} color={(!selectedTime || !selectedDayData) ? "#94A3B8" : "#059669"} />
            </View>
          </LinearGradient>
        </TouchableOpacity>
      </View>

      {/* MODALS (PAYMENT, VISA, SUCCESS, REVIEW) */}
      <Modal visible={modalStep === "payment"} transparent animationType="slide">
        <TouchableOpacity style={styles.modalBackdrop} activeOpacity={1} onPress={() => setModalStep(null)}>
          <View style={styles.modalSheet}>
            <View style={styles.modalIndicator} />
            <Text style={styles.modalHeaderTitle}>Payment Method</Text>
            <Text style={styles.modalHeaderSub}>Choose how you want to pay for your visit</Text>

            <TouchableOpacity style={[styles.payMethodCard, payMethod === "visa" && styles.payMethodActive]} onPress={() => handlePaymentChoice("visa")}>
              <View style={[styles.payIconBox, { backgroundColor: '#EFF6FF' }]}><Ionicons name="card" size={24} color="#2563EB" /></View>
              <View style={{ flex: 1 }}>
                <Text style={styles.payTitle}>Credit / Debit Card</Text>
                <Text style={styles.paySub}>Visa, Mastercard, etc.</Text>
              </View>
              <View style={[styles.radioCircle, payMethod === "visa" && styles.radioActive]} />
            </TouchableOpacity>

            <TouchableOpacity style={[styles.payMethodCard, payMethod === "cash" && styles.payMethodActive]} onPress={() => handlePaymentChoice("cash")}>
              <View style={[styles.payIconBox, { backgroundColor: '#F0FDF4' }]}><Ionicons name="cash" size={24} color="#059669" /></View>
              <View style={{ flex: 1 }}>
                <Text style={styles.payTitle}>Cash at Clinic</Text>
                <Text style={styles.paySub}>Pay after consultation</Text>
              </View>
              <View style={[styles.radioCircle, payMethod === "cash" && styles.radioActive]} />
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Visa Form Modal */}
      <Modal visible={modalStep === "visa_form"} transparent animationType="slide">
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.modalBackdrop}>
          <View style={styles.modalSheet}>
            <View style={styles.modalIndicator} />
            <View style={styles.modalHeaderWithBack}>
              <TouchableOpacity onPress={() => setModalStep("payment")}><Ionicons name="arrow-back" size={24} color="#1E293B" /></TouchableOpacity>
              <Text style={styles.modalHeaderTitle}>Card Details</Text>
              <View style={{ width: 24 }} />
            </View>

            <View style={styles.visaFormContainer}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabelText}>Cardholder Name</Text>
                <TextInput style={styles.premiumInput} placeholder="e.g. John Doe" value={cardName} onChangeText={setCardName} />
                {cardErrors.cardName && <Text style={styles.formError}>{cardErrors.cardName}</Text>}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabelText}>Card Number</Text>
                <TextInput style={styles.premiumInput} placeholder="0000 0000 0000 0000" value={cardNumber} onChangeText={(v) => setCardNumber(formatCardNumber(v))} keyboardType="numeric" maxLength={19} />
                {cardErrors.cardNumber && <Text style={styles.formError}>{cardErrors.cardNumber}</Text>}
              </View>

              <View style={styles.visaInlineRow}>
                <View style={[styles.inputGroup, { flex: 1 }]}>
                  <Text style={styles.inputLabelText}>Expiry Date</Text>
                  <TextInput style={styles.premiumInput} placeholder="MM/YY" value={expiry} onChangeText={(v) => setExpiry(formatExpiry(v))} keyboardType="numeric" maxLength={5} />
                </View>
                <View style={[styles.inputGroup, { flex: 1 }]}>
                  <Text style={styles.inputLabelText}>CVV</Text>
                  <TextInput style={styles.premiumInput} placeholder="000" value={cvv} onChangeText={(v) => setCvv(v.slice(0, 4))} keyboardType="numeric" secureTextEntry />
                </View>
              </View>

              <TouchableOpacity style={styles.confirmPaymentBtn} onPress={handleVisaConfirm}>
                <LinearGradient colors={["#2563EB", "#1D4ED8"]} style={styles.confirmPaymentGradient}>
                  <Text style={styles.confirmPaymentText}>Pay ${doctor?.consultationFee}</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* Success Modal */}
      <Modal visible={showSuccess} transparent animationType="fade">
        <View style={styles.successContainer}>
          <View style={styles.successCard}>
            <View style={styles.successIconOuter}>
              <View style={styles.successIconInner}>
                <Ionicons name="checkmark" size={60} color="#fff" />
              </View>
            </View>
            <Text style={styles.successMainTitle}>Booking Requested!</Text>
            <Text style={styles.successMainSub}>Your appointment with Dr. {doctor?.name} is waiting for approval. You'll receive a notification soon.</Text>
            <TouchableOpacity style={styles.successActionBtn} onPress={() => { setShowSuccess(false); router.back(); }}>
              <Text style={styles.successActionText}>Go to Appointments</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Review Modal */}
      <Modal visible={showAddReview} transparent animationType="fade">
        <View style={styles.reviewModalContainer}>
          <View style={styles.reviewModalBox}>
            <Text style={styles.reviewModalTitle}>Rate Your Visit</Text>
            <Text style={styles.reviewModalSub}>How was your experience with Dr. {doctor?.name}?</Text>

            <View style={styles.ratingStarsWrap}>
              <StarRow value={myRating} onChange={setMyRating} size={42} />
            </View>

            <TextInput
              style={styles.reviewTextInput}
              placeholder="Tell us about the doctor, the clinic, or the service..."
              value={myComment}
              onChangeText={setMyComment}
              multiline
              numberOfLines={4}
            />

            <View style={styles.reviewBtnRow}>
              <TouchableOpacity style={styles.reviewCancelBtn} onPress={() => setShowAddReview(false)}>
                <Text style={styles.reviewCancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.reviewSubmitBtn} onPress={myExistingReview ? handleUpdateExistingReview : submitReview} disabled={savingReview}>
                <LinearGradient colors={["#059669", "#047857"]} style={styles.reviewSubmitGradient}>
                  <Text style={styles.reviewSubmitBtnText}>{savingReview ? "..." : "Submit"}</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: '#fff' },
  retryTxt: { color: "#64748B", fontWeight: "500", marginTop: 10 },
  errorTxt: { color: "#EF4444", fontSize: 16, fontWeight: '600', marginTop: 10 },
  retryBtn: { backgroundColor: "#059669", paddingHorizontal: 40, paddingVertical: 14, borderRadius: 30, marginTop: 20, elevation: 4 },

  magicHeader: { position: 'absolute', top: 0, left: 0, right: 0, height: 340, overflow: 'hidden', zIndex: 0 },
  fixedHeaderNav: { position: 'absolute', top: 50, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 20, zIndex: 1000 },
  liquidBlob: { position: 'absolute', borderRadius: 150 },
  headerNav: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 50, zIndex: 10 },
  glassBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)' },

  doctorHeroSection: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 24, paddingTop: 110 },
  profileImgContainer: { position: 'relative' },
  avatarGlassBorder: { padding: 4, borderRadius: 54, backgroundColor: 'rgba(255,255,255,0.2)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)' },
  profileImg: { width: 100, height: 100, borderRadius: 50 },
  activePulse: { position: 'absolute', bottom: 4, right: 4, width: 18, height: 18, borderRadius: 9, backgroundColor: '#10B981', borderWidth: 3, borderColor: '#064E3B' },

  heroInfo: { marginLeft: 20, flex: 1 },
  specBadge: { alignSelf: 'flex-start', backgroundColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, marginBottom: 6 },
  specBadgeText: { color: '#fff', fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  heroName: { fontSize: 26, fontWeight: '900', color: '#fff', letterSpacing: -0.5 },
  heroRatingRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  heroRatingText: { color: '#fff', fontSize: 14, fontWeight: '800', marginLeft: 4 },
  ratingDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: 'rgba(255,255,255,0.4)', marginHorizontal: 8 },
  heroReviewCount: { color: 'rgba(255,255,255,0.8)', fontSize: 13, fontWeight: '600' },

  contentCard: { backgroundColor: '#F8FAFC', borderTopLeftRadius: 45, borderTopRightRadius: 45, paddingHorizontal: 20, paddingTop: 30, marginTop: -45, minHeight: 600, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 20, elevation: 5 },
  statsContainer: { flexDirection: 'row', backgroundColor: '#fff', borderRadius: 30, padding: 20, marginBottom: 35, elevation: 15, shadowColor: '#064E3B', shadowOpacity: 0.1, shadowRadius: 25, borderWidth: 1, borderColor: '#F1F5F9' },
  statItem: { flex: 1, alignItems: 'center' },
  statIconWrap: { width: 44, height: 44, borderRadius: 16, justifyContent: 'center', alignItems: 'center', marginBottom: 10 },
  statLabel: { fontSize: 10, color: '#94A3B8', fontWeight: '800', textTransform: 'uppercase', letterSpacing: 0.5 },
  statValue: { fontSize: 15, fontWeight: '900', color: '#1E293B', marginTop: 2 },
  statDivider: { width: 1, height: '60%', backgroundColor: '#F1F5F9', alignSelf: 'center' },

  section: { marginBottom: 35 },
  sectionTitle: { fontSize: 18, fontWeight: '900', color: '#1E293B', marginBottom: 18, letterSpacing: -0.5 },
  bioCard: { backgroundColor: '#fff', padding: 20, borderRadius: 28, borderWidth: 1, borderColor: '#F1F5F9', elevation: 2, shadowColor: '#000', shadowOpacity: 0.02 },
  bioText: { fontSize: 15, color: '#64748B', lineHeight: 24, fontWeight: '500' },

  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 },
  loaderContainer: { paddingVertical: 40, alignItems: 'center' },
  daysScroll: { marginBottom: 25 },
  dayButton: { width: 68, height: 85, backgroundColor: '#fff', borderRadius: 20, justifyContent: 'center', alignItems: 'center', marginRight: 15, borderWidth: 1.5, borderColor: '#F1F5F9', elevation: 3 },
  dayButtonActive: { backgroundColor: '#059669', borderColor: '#059669', elevation: 12, shadowColor: '#059669', shadowOpacity: 0.4, shadowRadius: 15 },
  dayName: { fontSize: 12, color: '#94A3B8', fontWeight: '700', textTransform: 'uppercase' },
  dayNum: { fontSize: 18, fontWeight: '900', color: '#1E293B', marginTop: 4 },
  dayTextActive: { color: '#fff' },

  slotsContainer: { marginTop: 10 },
  slotsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  slotCard: { paddingHorizontal: 14, paddingVertical: 14, borderRadius: 18, backgroundColor: '#fff', borderWidth: 1.5, borderColor: '#F1F5F9', flex: 1, minWidth: '28%', alignItems: 'center', elevation: 2 },
  slotCardActive: { backgroundColor: '#ECFDF5', borderColor: '#059669', elevation: 6, shadowColor: '#059669', shadowOpacity: 0.2 },
  slotTime: { fontSize: 13, fontWeight: '700', color: '#475569' },
  slotTimeActive: { color: '#059669', fontWeight: '900' },
  emptyState: { alignItems: 'center', paddingVertical: 40, backgroundColor: '#fff', borderRadius: 28, borderWidth: 1, borderColor: '#F1F5F9', borderStyle: 'dashed' },
  emptyStateText: { color: '#94A3B8', marginTop: 12, fontSize: 14, fontWeight: '600' },

  waitingCard: { borderRadius: 24, overflow: 'hidden', elevation: 5 },
  waitingGradient: { padding: 25, alignItems: 'center' },
  waitingTitle: { fontSize: 18, fontWeight: '800', color: '#065F46', marginTop: 12 },
  waitingSub: { fontSize: 14, color: '#059669', textAlign: 'center', marginTop: 6, marginBottom: 20, opacity: 0.8 },
  notifyAction: { backgroundColor: '#fff', paddingHorizontal: 25, paddingVertical: 12, borderRadius: 15, borderWidth: 1, borderColor: '#059669' },
  notifyActionActive: { backgroundColor: '#059669' },
  notifyActionText: { color: '#059669', fontWeight: '700' },

  reviewsList: { gap: 18 },
  reviewItem: { backgroundColor: '#fff', padding: 20, borderRadius: 28, borderWidth: 1, borderColor: '#F1F5F9', elevation: 2, shadowColor: '#000', shadowOpacity: 0.02 },
  reviewHeaderRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 15 },
  reviewerAvatar: { width: 44, height: 44, borderRadius: 15, backgroundColor: '#F1F5F9', justifyContent: 'center', alignItems: 'center', marginRight: 15 },
  avatarInitial: { color: '#059669', fontWeight: '900', fontSize: 16 },
  reviewerName: { fontSize: 15, fontWeight: '800', color: '#1E293B' },
  starsContainer: { marginTop: 4 },
  reviewDateText: { fontSize: 12, color: '#94A3B8', fontWeight: '500' },
  reviewContent: { fontSize: 14, color: '#64748B', lineHeight: 22, fontWeight: '500' },
  addReviewLink: { backgroundColor: '#ECFDF5', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 12 },
  addReviewLinkText: { color: '#059669', fontSize: 13, fontWeight: '800' },

  bottomBar: { position: 'absolute', bottom: 0, left: 0, right: 0, backgroundColor: '#fff', paddingHorizontal: 20, paddingVertical: 18, paddingBottom: Platform.OS === 'ios' ? 34 : 20, flexDirection: 'row', alignItems: 'center', borderTopLeftRadius: 35, borderTopRightRadius: 35, shadowColor: '#000', shadowOffset: { width: 0, height: -10 }, shadowOpacity: 0.1, shadowRadius: 15, elevation: 25, gap: 12 },
  messageTextBtn: { flex: 0.45, height: 56, borderRadius: 20, backgroundColor: '#F0FDF4', flexDirection: 'row', justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: '#059669', gap: 8 },
  messageBtnText: { color: '#059669', fontSize: 15, fontWeight: '800' },
  premiumBookBtn: { flex: 1, height: 56, borderRadius: 20, overflow: 'hidden', elevation: 8, shadowColor: '#059669', shadowOpacity: 0.3, shadowRadius: 12 },
  disabledBookBtn: { elevation: 0, shadowOpacity: 0 },
  bookGradient: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 15 },
  bookText: { color: '#fff', fontSize: 16, fontWeight: '800', marginRight: 10 },
  bookIconCircle: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },

  modalBackdrop: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.6)', justifyContent: 'flex-end' },
  modalSheet: { backgroundColor: '#fff', borderTopLeftRadius: 35, borderTopRightRadius: 35, padding: 24, paddingBottom: 40 },
  modalIndicator: { width: 40, height: 5, backgroundColor: '#F1F5F9', borderRadius: 5, alignSelf: 'center', marginBottom: 20 },
  modalHeaderTitle: { fontSize: 20, fontWeight: '800', color: '#1E293B' },
  modalHeaderSub: { fontSize: 14, color: '#64748B', marginTop: 4, marginBottom: 25 },
  payMethodCard: { flexDirection: 'row', alignItems: 'center', padding: 16, backgroundColor: '#F8FAFC', borderRadius: 20, marginBottom: 12, borderWidth: 1.5, borderColor: '#F1F5F9' },
  payMethodActive: { borderColor: '#059669', backgroundColor: '#F0FDF4' },
  payIconBox: { width: 48, height: 48, borderRadius: 14, justifyContent: 'center', alignItems: 'center', marginRight: 15 },
  payTitle: { fontSize: 16, fontWeight: '700', color: '#1E293B' },
  paySub: { fontSize: 12, color: '#64748B' },
  radioCircle: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, borderColor: '#CBD5E1' },
  radioActive: { borderColor: '#059669', borderWidth: 6 },

  modalHeaderWithBack: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25 },
  visaFormContainer: { gap: 15 },
  inputGroup: { gap: 8 },
  inputLabelText: { fontSize: 13, fontWeight: '700', color: '#64748B', marginLeft: 4 },
  premiumInput: { backgroundColor: '#F8FAFC', height: 55, borderRadius: 16, paddingHorizontal: 16, borderWidth: 1.5, borderColor: '#F1F5F9', fontSize: 15, fontWeight: '600', color: '#1E293B' },
  visaInlineRow: { flexDirection: 'row', gap: 15 },
  confirmPaymentBtn: { marginTop: 10, height: 55, borderRadius: 16, overflow: 'hidden' },
  confirmPaymentGradient: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  confirmPaymentText: { color: '#fff', fontSize: 16, fontWeight: '800' },
  formError: { color: '#EF4444', fontSize: 12, marginLeft: 4 },

  successContainer: { flex: 1, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', padding: 30 },
  successCard: { alignItems: 'center', width: '100%' },
  successIconOuter: { width: 140, height: 140, borderRadius: 70, backgroundColor: '#ECFDF5', justifyContent: 'center', alignItems: 'center', marginBottom: 40 },
  successIconInner: { width: 100, height: 100, borderRadius: 50, backgroundColor: '#059669', justifyContent: 'center', alignItems: 'center', elevation: 15, shadowColor: '#059669', shadowOpacity: 0.4, shadowRadius: 20 },
  successMainTitle: { fontSize: 28, fontWeight: '900', color: '#1E293B', marginBottom: 15, textAlign: 'center' },
  successMainSub: { fontSize: 16, color: '#64748B', textAlign: 'center', lineHeight: 26, marginBottom: 50, paddingHorizontal: 10 },
  successActionBtn: { backgroundColor: '#059669', width: '100%', height: 60, borderRadius: 22, justifyContent: 'center', alignItems: 'center', elevation: 10, shadowColor: '#059669', shadowOpacity: 0.3, shadowRadius: 15 },
  successActionText: { color: '#fff', fontSize: 17, fontWeight: '800' },

  reviewModalContainer: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.6)', justifyContent: 'center', padding: 20 },
  reviewModalBox: { backgroundColor: '#fff', borderRadius: 30, padding: 25, alignItems: 'center' },
  reviewModalTitle: { fontSize: 20, fontWeight: '800', color: '#1E293B' },
  reviewModalSub: { fontSize: 14, color: '#64748B', textAlign: 'center', marginTop: 6, marginBottom: 20 },
  ratingStarsWrap: { marginVertical: 10 },
  reviewTextInput: { width: '100%', height: 120, backgroundColor: '#F8FAFC', borderRadius: 20, padding: 15, textAlignVertical: 'top', borderWidth: 1.5, borderColor: '#F1F5F9', marginTop: 15, fontSize: 14 },
  reviewBtnRow: { flexDirection: 'row', gap: 12, marginTop: 25, width: '100%' },
  reviewCancelBtn: { flex: 1, height: 55, borderRadius: 15, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F1F5F9' },
  reviewCancelBtnText: { color: '#64748B', fontWeight: '700' },
  reviewSubmitBtn: { flex: 2, height: 55, borderRadius: 15, overflow: 'hidden' },
  reviewSubmitGradient: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  reviewSubmitBtnText: { color: '#fff', fontSize: 15, fontWeight: '800' },
});