import React, { useState, useEffect, useMemo } from "react";
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  ActivityIndicator, Alert, Modal, TextInput, KeyboardAvoidingView,
  Platform, Image
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { COLORS } from "../../constants/colors";
import { getDoctorById, getReviewsByDoctor, addReview, updateMyReview, deleteMyReview, DoctorDetails, Review } from "../../services/doctorService";
import { bookAppointment } from "../../services/appointmentService";
import { apiFetch } from "../../services/http";
import { BASE_URL } from "../../constants/api";
import { onScheduleReady, onScheduleUpdated, startSignalRConnection, subscribeToDoctorSchedule } from "../../services/signalr";
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
      
      // If we follow, we definitely want notifications. 
      // If we unfollow, we only keep notifications if explicitly subscribed.
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
      
      // If we are currently following, we can't really "disable" notifications 
      // without unfollowing, unless the user wants to keep following but mute?
      // But the requirement says "notify me should not follow".
      // So if I'm not following, I can subscribe/unsubscribe.
      
      await setSubscribed(id, next)
      setNotifyEnabled(next || isFollowed)

      if (next) {
        await apiFetch(`${BASE_URL}/api/doctors/${doctorId}/notify-schedule`, { method: 'POST' }, true)
        await startSignalRConnection()
        await subscribeToDoctorSchedule(id)
        Alert.alert("🔔 Notifications On", "You will be notified when Dr. " + (doctor?.fullName || "the doctor") + " updates their schedule.")
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
      console.log('Availability raw data:', JSON.stringify(data, null, 2))

      // Handle both nested structure and direct array
      const availData = data?.days ?? data?.Days ?? (Array.isArray(data) ? data : [])
      const bookedData = data?.bookedSlots ?? data?.BookedSlots ?? []

      console.log('Availability data:', availData)
      console.log('Booked slots:', bookedData)

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
        // ignore subscription issues
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
      if ((e?.message ?? "").toLowerCase().includes("already reviewed")) {
        Alert.alert("Review Exists", "You already reviewed this doctor. You can edit or delete your old review.", [
          {
            text: "Edit Old Review",
            onPress: () => {
              handleUpdateExistingReview().catch(() => undefined);
            },
          },
          {
            text: "Delete Old Review",
            style: "destructive",
            onPress: () => {
              handleDeleteExistingReview().catch(() => undefined);
            },
          },
          { text: "Cancel", style: "cancel" },
        ]);
      } else {
        Alert.alert("Error", e.message);
      }
    } finally {
      setSavingReview(false);
    }
  };

  const handleUpdateExistingReview = async () => {
    if (myRating === 0) {
      Alert.alert("Rating required", "Please select a star rating.");
      return;
    }
    if (!myComment.trim()) {
      Alert.alert("Comment required", "Please write a comment.");
      return;
    }

    setSavingReview(true);
    try {
      await updateMyReview(Number(doctorId), myRating, myComment.trim(), myExistingReview?.id);
      await fetchAll();
      setShowAddReview(false);
      Alert.alert("Updated", "Your review was updated successfully.");
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
      setMyRating(0);
      setMyComment("");
      setShowAddReview(false);
      Alert.alert("Deleted", "Your review was deleted.");
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? "Failed to delete review.");
    } finally {
      setSavingReview(false);
    }
  };

  const myExistingReview = useMemo(() => {
    const byMineFlag = reviews.find((review) => review.isMine === true)
    if (byMineFlag) return byMineFlag

    const myName = normalizeText(myDisplayName)
    if (!myName) return undefined

    return reviews.find((review) => normalizeText(review.author) === myName)
  }, [reviews, myDisplayName])

  const confirmDeleteMyReview = () => {
    if (!myExistingReview) return

    Alert.alert(
      "Delete Review",
      "Are you sure you want to delete your review?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: () => {
            handleDeleteExistingReview().catch(() => undefined)
          },
        },
      ]
    )
  }

  const openReviewModal = () => {
    if (myExistingReview) {
      setMyRating(Number(myExistingReview.rating || 0));
      setMyComment(myExistingReview.comment || "");
    } else {
      setMyRating(0);
      setMyComment("");
    }
    setShowAddReview(true);
  };

  const avgRating = reviews.length > 0
    ? (reviews.reduce((sum, r) => sum + Number(r.rating || 0), 0) / reviews.length).toFixed(1)
    : (doctor?.rating ? doctor.rating.toFixed(1) : "0.0");
  const reviewsCount = reviews.length > 0 ? reviews.length : (doctor?.reviewCount ?? 0);

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color={COLORS.primary} /></View>;
  if (error || !doctor) return (
    <View style={styles.center}>
      <Text style={styles.errorTxt}>⚠️ {error || "Doctor not found"}</Text>
      <TouchableOpacity style={styles.retryBtn} onPress={fetchAll}>
        <Text style={styles.retryTxt}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  const selectedDayData = days[selectedDay] ?? null;
  const currentSlots = selectedDayData ? getTimeSlotsForDay(selectedDayData.isoDate, availability, bookedSlots) : [];

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={22} color="#1A1A1A" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.headerBtn, { right: 16 }]}
          onPress={() => toggleFollow(Number(doctorId))}
          disabled={following}
        >
          {following ? (
            <ActivityIndicator size="small" color={COLORS.primary} />
          ) : (
            <Ionicons name={isFollowed ? "heart" : "heart-outline"} size={24} color={isFollowed ? "#e53935" : "#1A1A1A"} />
          )}
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.profileCard}>
          <View style={styles.avatarWrap}>
            <View style={styles.avatarFallback}>
              {doctor.photoUrl || (doctor as any).imageUrl ? (
                <Image source={{ uri: doctor.photoUrl || (doctor as any).imageUrl }} style={styles.avatarImg} />
              ) : (
                <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.avatarImg} />
              )}
            </View>
            <View style={[styles.onlineDot, { backgroundColor: doctor.isAvailable ? "#4CAF50" : "#CBD5E1" }]} />
          </View>
          <Text style={styles.docName}>{doctor.name}</Text>
          <Text style={styles.docSpec}>{doctor.specialty}</Text>
          <Text style={[styles.docStatus, { color: doctor.isAvailable ? "#16A34A" : "#94A3B8" }]}>
            {doctor.isAvailable ? "Online" : "Offline"}
          </Text>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <RatingStars rating={Number(avgRating)} showText={false} size={14} />
              <Text style={[styles.statVal, { marginTop: 4 }]}>{avgRating}</Text>
              <Text style={styles.statLbl}>Rating</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="time-outline" size={14} color={COLORS.primary} />
              <Text style={styles.statVal}>{doctor.experience || doctor.yearsExperience || 0} yrs</Text>
              <Text style={styles.statLbl}>Experience</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="people-outline" size={14} color={COLORS.primary} />
              <Text style={styles.statVal}>{reviewsCount}</Text>
              <Text style={styles.statLbl}>Reviews</Text>
            </View>
          </View>
        </View>

        {doctor.bio && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>About</Text>
            <Text style={styles.bio}>{doctor.bio}</Text>
          </View>
        )}

        <View style={styles.pills}>
          <View style={styles.pill}>
            <Ionicons name="location-outline" size={14} color={COLORS.primary} />
            <Text style={styles.pillTxt}>{doctor.location || "Clinic Location"}</Text>
          </View>
          <View style={styles.pill}>
            <Ionicons name="cash-outline" size={14} color={COLORS.primary} />
            <Text style={styles.pillTxt}>${doctor.consultationFee} per visit</Text>
          </View>
        </View>

        <View style={styles.quickActions}>
          <TouchableOpacity
            style={styles.messageDoctorBtn}
            onPress={() =>
              router.push({
                pathname: "/(patient)/messages",
                params: {
                  doctorId: String(doctor.id),
                  doctorName: doctor.name,
                },
              })
            }
          >
            <Ionicons name="chatbubble-ellipses-outline" size={15} color="#fff" />
            <Text style={styles.messageDoctorTxt}>Message Doctor</Text>
          </TouchableOpacity>
        </View>

        {loadingAvailability ? (
          <View style={[styles.section, { alignItems: 'center', paddingVertical: 40 }]}>
            <ActivityIndicator color={COLORS.primary} size="large" />
          </View>
        ) : hasSchedule ? (
          <>
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Select Date</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10 }}>
                {days.map((day, i) => (
                  <TouchableOpacity
                    key={i}
                    style={[styles.dayChip, selectedDay === i && styles.dayChipActive]}
                    onPress={() => { setSelectedDay(i); setSelectedTime(null); }}
                  >
                    <Text style={[styles.dayLabel, selectedDay === i && styles.dayLabelActive]}>{day.label}</Text>
                    <Text style={[styles.dayDate, selectedDay === i && styles.dayDateActive]}>{day.date}</Text>
                  </TouchableOpacity>
                ))}
                {days.length === 0 && (
                  <Text style={{ color: "#AAA", marginTop: 8 }}>No available dates configured yet.</Text>
                )}
              </ScrollView>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Select Time</Text>
              <View style={styles.timeGrid}>
                {currentSlots.map((slot) => (
                  <TouchableOpacity
                    key={slot}
                    style={[styles.timeChip, selectedTime === slot && styles.timeChipActive]}
                    onPress={() => setSelectedTime(slot)}
                  >
                    <Text style={[styles.timeTxt, selectedTime === slot && styles.timeTxtActive]}>{slot}</Text>
                  </TouchableOpacity>
                ))}
                {currentSlots.length === 0 && (
                  <Text style={{ color: "#AAA", marginTop: 8 }}>No available slots on this day.</Text>
                )}
              </View>
            </View>
          </>
        ) : (
          <View style={styles.waitingContainer}>
            <Text style={styles.waitingIcon}>⏳</Text>
            <Text style={styles.waitingTitle}>Schedule Coming Soon</Text>
            <Text style={styles.waitingText}>This doctor has not set their schedule yet.</Text>
            <TouchableOpacity style={[styles.notifyBtn, notifyEnabled && styles.notifyBtnActive]} onPress={handleNotifyMe}>
              <Text style={[styles.notifyBtnText, notifyEnabled && { color: "#fff" }]}>
                {notifyEnabled ? '🔔 Notified!' : '🔔 Notify Me When Ready'}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.section}>
          <View style={styles.reviewHeader}>
            <Text style={styles.sectionTitle}>Reviews ({reviews.length})</Text>
            <View style={styles.reviewActionsRow}>
              {myExistingReview && (
                <>
                  <TouchableOpacity style={styles.editReviewBtn} onPress={openReviewModal}>
                    <Ionicons name="create-outline" size={14} color={COLORS.primary} />
                    <Text style={styles.editReviewTxt}>Edit</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.deleteReviewBtn} onPress={confirmDeleteMyReview}>
                    <Ionicons name="trash-outline" size={14} color="#DC2626" />
                    <Text style={styles.deleteReviewTxt}>Delete</Text>
                  </TouchableOpacity>
                </>
              )}
              <TouchableOpacity style={styles.addReviewBtn} onPress={openReviewModal}>
                <Ionicons name="add" size={14} color="#fff" />
                <Text style={styles.addReviewTxt}>{myExistingReview ? "Edit Review" : "Add Review"}</Text>
              </TouchableOpacity>
            </View>
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
                  <View style={styles.reviewAvatar}><Text style={styles.reviewAvatarTxt}>{r.author?.charAt(0).toUpperCase() || 'U'}</Text></View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.reviewAuthor}>{r.author}</Text>
                    <Text style={styles.reviewDate}>{r.date ? new Date(r.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "--"}</Text>
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

      <View style={styles.bottomBar}>
        <View style={styles.feePill}>
          <Text style={styles.feeLbl}>Fee</Text>
          <Text style={styles.feeAmt}>${doctor.consultationFee}</Text>
        </View>
        <TouchableOpacity style={[styles.bookBtn, (!selectedTime || !selectedDayData) && styles.bookBtnOff]} onPress={handleProceed} disabled={booking || !selectedTime || !selectedDayData}>
          {booking ? <ActivityIndicator color="#fff" /> : <><Text style={styles.bookBtnTxt}>{!selectedDayData ? "No Dates" : selectedTime ? "Continue" : "Pick a Time"}</Text>{selectedTime && selectedDayData && <Ionicons name="arrow-forward" size={15} color="#fff" />}</>}
        </TouchableOpacity>
      </View>

      <Modal visible={modalStep === "payment"} transparent animationType="slide">
        <TouchableOpacity style={styles.overlay} activeOpacity={1} onPress={() => setModalStep(null)}>
          <View style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Choose Payment</Text>
            <Text style={styles.sheetSub}>{selectedDayData?.date ?? "--"} · {selectedTime} · ${doctor.consultationFee}</Text>
            <TouchableOpacity style={styles.payOption} onPress={() => handlePaymentChoice("visa")}>
              <View style={[styles.payIconBg, { backgroundColor: "#EEF2FF" }]}><Ionicons name="card" size={22} color="#4F46E5" /></View>
              <View style={{ flex: 1 }}><Text style={styles.payTitle}>Pay with Visa</Text><Text style={styles.paySub}>Credit / Debit card</Text></View>
              <Ionicons name="chevron-forward" size={18} color="#CCC" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.payOption} onPress={() => handlePaymentChoice("cash")}>
              <View style={[styles.payIconBg, { backgroundColor: "#ECFDF5" }]}><Ionicons name="cash" size={22} color={COLORS.primary} /></View>
              <View style={{ flex: 1 }}><Text style={styles.payTitle}>Pay on Arrival</Text><Text style={styles.paySub}>Cash at clinic</Text></View>
              <Ionicons name="chevron-forward" size={18} color="#CCC" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.cancelBtn} onPress={() => setModalStep(null)}><Text style={styles.cancelTxt}>Cancel</Text></TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      <Modal visible={modalStep === "visa_form"} transparent animationType="slide">
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.overlay}>
          <View style={[styles.sheet, { paddingBottom: 36 }]}>
            <View style={styles.sheetHandle} />
            <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 6 }}>
              <TouchableOpacity onPress={() => setModalStep("payment")} style={{ marginRight: 10 }}><Ionicons name="chevron-back" size={22} color="#333" /></TouchableOpacity>
              <Text style={styles.sheetTitle}>Card Details</Text>
            </View>
            <Text style={styles.sheetSub}>All transactions are secure & encrypted</Text>
            <Text style={styles.fieldLabel}>Cardholder Name</Text>
            <TextInput style={[styles.input, cardErrors.cardName && styles.inputError]} placeholder="Name on card" placeholderTextColor="#BBB" value={cardName} onChangeText={setCardName} autoCapitalize="words" />
            {cardErrors.cardName && <Text style={styles.fieldError}>{cardErrors.cardName}</Text>}
            <Text style={styles.fieldLabel}>Card Number</Text>
            <View style={[styles.inputRow, cardErrors.cardNumber && styles.inputError]}><Ionicons name="card-outline" size={18} color="#AAA" /><TextInput style={[styles.input, { flex: 1, borderWidth: 0, marginBottom: 0, padding: 0 }]} placeholder="0000 0000 0000 0000" placeholderTextColor="#BBB" value={cardNumber} onChangeText={(v) => setCardNumber(formatCardNumber(v))} keyboardType="numeric" maxLength={19} /></View>
            {cardErrors.cardNumber && <Text style={styles.fieldError}>{cardErrors.cardNumber}</Text>}
            <View style={{ flexDirection: "row", gap: 12 }}>
              <View style={{ flex: 1 }}><Text style={styles.fieldLabel}>Expiry Date</Text><TextInput style={[styles.input, cardErrors.expiry && styles.inputError]} placeholder="MM/YY" placeholderTextColor="#BBB" value={expiry} onChangeText={(v) => setExpiry(formatExpiry(v))} keyboardType="numeric" maxLength={5} />{cardErrors.expiry && <Text style={styles.fieldError}>{cardErrors.expiry}</Text>}</View>
              <View style={{ flex: 1 }}><Text style={styles.fieldLabel}>CVV</Text><TextInput style={[styles.input, cardErrors.cvv && styles.inputError]} placeholder="•••" placeholderTextColor="#BBB" value={cvv} onChangeText={(v) => setCvv(v.replace(/\D/g, "").slice(0, 4))} keyboardType="numeric" secureTextEntry maxLength={4} />{cardErrors.cvv && <Text style={styles.fieldError}>{cardErrors.cvv}</Text>}</View>
            </View>
            <TouchableOpacity style={styles.confirmPayBtn} onPress={handleVisaConfirm}><Ionicons name="lock-closed" size={15} color="#fff" /><Text style={styles.confirmPayTxt}>Pay ${doctor.consultationFee} Securely</Text></TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      <Modal visible={showAddReview} transparent animationType="slide">
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.overlay}>
          <View style={[styles.sheet, { paddingBottom: 36 }]}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Write a Review</Text>
            <Text style={styles.sheetSub}>Share your experience with {doctor.name}</Text>
            <Text style={styles.fieldLabel}>Your Rating</Text>
            <StarRow value={myRating} onChange={setMyRating} size={32} />
            {myRating > 0 && <Text style={[styles.fieldLabel, { color: "#FFB300", marginTop: 6 }]}>{["", "Poor", "Fair", "Good", "Very Good", "Excellent"][myRating]}</Text>}
            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Your Comment</Text>
            <TextInput style={[styles.input, styles.textarea]} placeholder="Tell others about your experience..." placeholderTextColor="#BBB" value={myComment} onChangeText={setMyComment} multiline numberOfLines={4} textAlignVertical="top" />
            <View style={{ flexDirection: "row", gap: 10, marginTop: 8 }}>
              <TouchableOpacity style={[styles.confirmPayBtn, { flex: 1, backgroundColor: "#F0F0F0" }]} onPress={() => setShowAddReview(false)}><Text style={[styles.confirmPayTxt, { color: "#555" }]}>Cancel</Text></TouchableOpacity>
              <TouchableOpacity style={[styles.confirmPayBtn, { flex: 2 }]} onPress={myExistingReview ? handleUpdateExistingReview : submitReview} disabled={savingReview}>{savingReview ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.confirmPayTxt}>{myExistingReview ? "Update Review" : "Submit Review"}</Text>}</TouchableOpacity>
            </View>
            {myExistingReview && (
              <TouchableOpacity style={[styles.confirmPayBtn, { backgroundColor: "#DC2626", marginTop: 8 }]} onPress={handleDeleteExistingReview} disabled={savingReview}>
                <Text style={styles.confirmPayTxt}>Delete My Review</Text>
              </TouchableOpacity>
            )}
          </View>
        </KeyboardAvoidingView>
      </Modal>

      <Modal visible={showSuccess} transparent animationType="fade">
        <View style={styles.successOverlay}>
          <View style={styles.successCard}>
            <View style={styles.successIconWrap}><Ionicons name="checkmark-circle" size={64} color={COLORS.primary} /></View>
            <Text style={styles.successTitle}>Booking Confirmed!</Text>
            <Text style={styles.successDoc}>{doctor.name}</Text>
            <View style={styles.successDetails}>
              <View style={styles.successRow}><Ionicons name="calendar-outline" size={15} color={COLORS.primary} /><Text style={styles.successDetailTxt}>{selectedDayData?.date ?? "--"} · {selectedTime}</Text></View>
              <View style={styles.successRow}><Ionicons name={payMethod === "visa" ? "card-outline" : "cash-outline"} size={15} color={COLORS.primary} /><Text style={styles.successDetailTxt}>{payMethod === "visa" ? "Paid by Visa" : "Pay on Arrival"}</Text></View>
            </View>
            <TouchableOpacity style={styles.doneBtn} onPress={() => { setShowSuccess(false); router.back(); }}><Text style={styles.doneBtnTxt}>Done</Text></TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F4F6FA" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", gap: 12 },
  errorTxt: { color: "#e53935", fontSize: 14, textAlign: "center" },
  retryBtn: { backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 9, borderRadius: 18 },
  retryTxt: { color: "#fff", fontWeight: "600" },
  headerRow: { position: "absolute", top: 48, left: 16, right: 16, zIndex: 10, flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  backBtn: { backgroundColor: "#fff", borderRadius: 14, padding: 10, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 6, elevation: 4 },
  headerBtn: { backgroundColor: "#fff", borderRadius: 14, padding: 10, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 6, elevation: 4 },
  profileCard: { backgroundColor: "#fff", alignItems: "center", paddingTop: 88, paddingBottom: 24, paddingHorizontal: 20, borderBottomLeftRadius: 28, borderBottomRightRadius: 28, shadowColor: "#000", shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.06, shadowRadius: 10, elevation: 4 },
  avatarWrap: { position: "relative", marginBottom: 12 },
  avatarFallback: { width: 84, height: 84, borderRadius: 42, backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: COLORS.primary + "30", overflow: "hidden" },
  avatarImg: { width: "100%", height: "100%", borderRadius: 42 },
  avatarTxt: { fontSize: 32, fontWeight: "800", color: COLORS.primary },
  onlineDot: { position: "absolute", bottom: 4, right: 4, width: 15, height: 15, borderRadius: 8, backgroundColor: "#4CAF50", borderWidth: 3, borderColor: "#fff" },
  docName: { fontSize: 20, fontWeight: "800", color: "#1A1A1A" },
  docSpec: { fontSize: 13, color: COLORS.primary, fontWeight: "600", marginTop: 3 },
  docStatus: { fontSize: 12, fontWeight: "700", marginTop: 6 },
  statsRow: { flexDirection: "row", alignItems: "center", marginTop: 18, paddingTop: 18, borderTopWidth: 1, borderTopColor: "#F0F0F0", width: "100%", justifyContent: "center" },
  statItem: { flex: 1, alignItems: "center", gap: 3 },
  statDivider: { width: 1, height: 32, backgroundColor: "#F0F0F0" },
  statVal: { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  statLbl: { fontSize: 10, color: "#AAA" },
  section: { marginTop: 20, paddingHorizontal: 18 },
  sectionTitle: { fontSize: 16, fontWeight: "700", color: "#1A1A1A", marginBottom: 12 },
  bio: { fontSize: 13, color: "#666", lineHeight: 20 },
  pills: { flexDirection: "row", flexWrap: "wrap", gap: 10, paddingHorizontal: 18, marginTop: 14 },
  pill: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: COLORS.primary + "12", paddingHorizontal: 12, paddingVertical: 7, borderRadius: 20 },
  pillTxt: { fontSize: 12, color: "#444", fontWeight: "500" },
  quickActions: { paddingHorizontal: 18, marginTop: 12 },
  messageDoctorBtn: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: COLORS.primary,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 18,
  },
  messageDoctorTxt: { color: "#fff", fontSize: 12, fontWeight: "700" },
  dayChip: { alignItems: "center", paddingHorizontal: 14, paddingVertical: 10, borderRadius: 14, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#EEE", minWidth: 68 },
  dayChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  dayLabel: { fontSize: 10, color: "#AAA", fontWeight: "500" },
  dayLabelActive: { color: "rgba(255,255,255,0.8)" },
  dayDate: { fontSize: 13, color: "#1A1A1A", fontWeight: "700", marginTop: 2 },
  dayDateActive: { color: "#fff" },
  timeGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  timeChip: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 12, backgroundColor: "#fff", borderWidth: 1.5, borderColor: "#EEE" },
  timeChipActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  timeTxt: { fontSize: 12, color: "#333", fontWeight: "500" },
  timeTxtActive: { color: "#fff" },
  reviewHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  reviewActionsRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  editReviewBtn: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "#ECFEFF", paddingHorizontal: 10, paddingVertical: 6, borderRadius: 14, borderWidth: 1, borderColor: "#99F6E4" },
  editReviewTxt: { color: COLORS.primary, fontSize: 12, fontWeight: "700" },
  deleteReviewBtn: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "#FEF2F2", paddingHorizontal: 10, paddingVertical: 6, borderRadius: 14, borderWidth: 1, borderColor: "#FECACA" },
  deleteReviewTxt: { color: "#DC2626", fontSize: 12, fontWeight: "700" },
  addReviewBtn: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: COLORS.primary, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 18 },
  addReviewTxt: { color: "#fff", fontSize: 12, fontWeight: "600" },
  noReviews: { alignItems: "center", paddingVertical: 24, gap: 8 },
  noReviewsTxt: { fontSize: 13, color: "#BBB" },
  reviewCard: { backgroundColor: "#fff", borderRadius: 14, padding: 14, marginBottom: 10, shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 4, elevation: 1 },
  reviewTop: { flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 8 },
  reviewAvatar: { width: 36, height: 36, borderRadius: 18, backgroundColor: COLORS.primary + "20", justifyContent: "center", alignItems: "center" },
  reviewAvatarTxt: { fontSize: 15, fontWeight: "700", color: COLORS.primary },
  reviewAuthor: { fontSize: 13, fontWeight: "700", color: "#1A1A1A" },
  reviewDate: { fontSize: 11, color: "#BBB", marginTop: 1 },
  reviewComment: { fontSize: 13, color: "#555", lineHeight: 20 },
  bottomBar: { position: "absolute", bottom: 0, left: 0, right: 0, backgroundColor: "#fff", paddingHorizontal: 18, paddingTop: 12, paddingBottom: 28, borderTopWidth: 1, borderTopColor: "#F0F0F0", flexDirection: "row", alignItems: "center", gap: 12 },
  feePill: { backgroundColor: COLORS.primary + "12", paddingHorizontal: 14, paddingVertical: 10, borderRadius: 14, alignItems: "center" },
  feeLbl: { fontSize: 9, color: COLORS.primary, fontWeight: "600" },
  feeAmt: { fontSize: 15, fontWeight: "800", color: COLORS.primary },
  bookBtn: { flex: 1, backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 15, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 },
  bookBtnOff: { backgroundColor: "#DDD" },
  bookBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  overlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.45)", justifyContent: "flex-end" },
  sheet: { backgroundColor: "#fff", borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 22, paddingTop: 14, paddingBottom: 28 },
  sheetHandle: { width: 40, height: 4, borderRadius: 2, backgroundColor: "#DDD", alignSelf: "center", marginBottom: 18 },
  sheetTitle: { fontSize: 19, fontWeight: "800", color: "#1A1A1A", marginBottom: 3 },
  sheetSub: { fontSize: 12, color: "#AAA", marginBottom: 20 },
  payOption: { flexDirection: "row", alignItems: "center", gap: 14, backgroundColor: "#FAFAFA", borderRadius: 16, padding: 14, marginBottom: 10, borderWidth: 1.5, borderColor: "#F0F0F0" },
  payIconBg: { width: 44, height: 44, borderRadius: 14, justifyContent: "center", alignItems: "center" },
  payTitle: { fontSize: 14, fontWeight: "700", color: "#1A1A1A" },
  paySub: { fontSize: 11, color: "#AAA", marginTop: 2 },
  cancelBtn: { alignItems: "center", paddingVertical: 12, marginTop: 4 },
  cancelTxt: { color: "#AAA", fontSize: 14, fontWeight: "500" },
  fieldLabel: { fontSize: 12, fontWeight: "600", color: "#555", marginBottom: 6, marginTop: 12 },
  input: { backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 14, color: "#1A1A1A", borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2 },
  inputRow: { flexDirection: "row", alignItems: "center", gap: 10, backgroundColor: "#F7F7F7", borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, borderWidth: 1.5, borderColor: "#EFEFEF", marginBottom: 2 },
  inputError: { borderColor: "#e53935" },
  fieldError: { fontSize: 11, color: "#e53935", marginBottom: 4 },
  textarea: { height: 90, paddingTop: 12 },
  confirmPayBtn: { backgroundColor: COLORS.primary, borderRadius: 16, paddingVertical: 15, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 16 },
  confirmPayTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  successOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "center", alignItems: "center", padding: 24 },
  successCard: { backgroundColor: "#fff", borderRadius: 28, padding: 28, alignItems: "center", width: "100%" },
  successIconWrap: { width: 90, height: 90, borderRadius: 45, backgroundColor: COLORS.primary + "15", justifyContent: "center", alignItems: "center", marginBottom: 14 },
  successTitle: { fontSize: 20, fontWeight: "800", color: "#1A1A1A" },
  successDoc: { fontSize: 14, color: COLORS.primary, fontWeight: "600", marginTop: 3, marginBottom: 18 },
  successDetails: { backgroundColor: "#F8F8F8", borderRadius: 14, padding: 14, width: "100%", gap: 10 },
  successRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  successDetailTxt: { fontSize: 13, color: "#555", fontWeight: "500" },
  doneBtn: { backgroundColor: COLORS.primary, borderRadius: 18, paddingHorizontal: 44, paddingVertical: 13, marginTop: 18 },
  doneBtnTxt: { color: "#fff", fontSize: 15, fontWeight: "700" },
  waitingContainer: { alignItems: "center", paddingVertical: 40, paddingHorizontal: 20 },
  waitingIcon: { fontSize: 40, marginBottom: 12 },
  waitingTitle: { fontSize: 18, fontWeight: "700", color: "#1A1A1A", marginBottom: 6 },
  waitingText: { fontSize: 13, color: "#666", textAlign: "center", marginBottom: 20 },
  notifyBtn: { backgroundColor: "#EEF2FF", paddingHorizontal: 24, paddingVertical: 12, borderRadius: 20, borderWidth: 1, borderColor: "#4F46E5" },
  notifyBtnActive: { backgroundColor: "#4F46E5" },
  notifyBtnText: { color: "#4F46E5", fontWeight: "700", fontSize: 14 },
});
