import { View, TouchableOpacity, Text, Modal, ScrollView, StyleSheet } from 'react-native'
import { Ionicons } from '@expo/vector-icons'
import { useState, useEffect } from 'react'
import { COLORS } from '../constants/colors'
import { Notification, markAllAsRead, getNotifications, deleteNotification } from '../services/notificationService'

interface NotificationBellProps {
  onPress?: () => void
  isRTL?: boolean
}

export function NotificationBell({ onPress, isRTL = false }: NotificationBellProps) {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [modalVisible, setModalVisible] = useState(false)

  useEffect(() => {
    loadNotifications()
    const interval = setInterval(loadNotifications, 2000)
    return () => clearInterval(interval)
  }, [])

  const loadNotifications = async () => {
    const notifs = await getNotifications()
    setNotifications(notifs)
  }

  const handleDelete = async (id: string) => {
    await deleteNotification(id)
    await loadNotifications()
  }

  const unreadCount = notifications.filter(n => !n.isRead).length
  const openNotifications = async () => {
    setModalVisible(true)
    onPress?.()
    if (unreadCount > 0) {
      await markAllAsRead()
      const updated = await getNotifications()
      setNotifications(updated)
    }
  }

  return (
    <>
      <TouchableOpacity
        style={styles.bellButton}
        onPress={openNotifications}
      >
        <Ionicons name="notifications-outline" size={24} color={COLORS.primary} />
        {unreadCount > 0 && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>
              {unreadCount > 99 ? '99+' : unreadCount}
            </Text>
          </View>
        )}
      </TouchableOpacity>

      <Modal
        visible={modalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={[styles.header, isRTL && styles.headerRTL]}>
              <Text style={[styles.title, isRTL && styles.textRight]}>Notifications</Text>
              <TouchableOpacity onPress={() => setModalVisible(false)}>
                <Ionicons name="close" size={24} color={COLORS.primary} />
              </TouchableOpacity>
            </View>

            {notifications.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="notifications-off" size={48} color="#CBD5E1" />
                <Text style={styles.emptyText}>No notifications yet</Text>
              </View>
            ) : (
              <ScrollView style={styles.notificationsList} showsVerticalScrollIndicator={false}>
                {notifications.map((notif) => (
                  <View key={notif.id} style={[
                    styles.notificationItem, 
                    isRTL && styles.itemRTL,
                    !notif.isRead && styles.unreadItem
                  ]}>
                    <Text style={styles.icon}>{notif.icon}</Text>
                    <View style={{ flex: 1 }}>
                      <Text style={[styles.notifTitle, isRTL && styles.textRight]}>
                        {notif.title}
                      </Text>
                      <Text style={[styles.notifMessage, isRTL && styles.textRight]}>
                        {notif.message}
                      </Text>
                      <Text style={[styles.timestamp, isRTL && styles.textRight]}>
                        {formatTime(notif.timestamp)}
                      </Text>
                    </View>
                    <TouchableOpacity
                      onPress={() => handleDelete(notif.id)}
                      style={styles.deleteBtn}
                    >
                      <Ionicons name="trash-outline" size={18} color="#94A3B8" />
                    </TouchableOpacity>
                  </View>
                ))}
              </ScrollView>
            )}
          </View>
        </View>
      </Modal>
    </>
  )
}

function formatTime(timestamp: number): string {
  const now = Date.now()
  const diff = now - timestamp
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days < 7) return `${days}d ago`

  const date = new Date(timestamp)
  return date.toLocaleDateString()
}

const styles = StyleSheet.create({
  bellButton: {
    padding: 8,
    position: 'relative',
  },
  badge: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: '#FF5252',
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    height: '80%',
    paddingTop: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  headerRTL: {
    flexDirection: 'row-reverse',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1E293B',
  },
  textRight: {
    textAlign: 'right',
  },
  notificationsList: {
    flex: 1,
    paddingHorizontal: 12,
  },
  notificationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
    gap: 12,
    borderRadius: 16,
    marginVertical: 4,
  },
  itemRTL: {
    flexDirection: 'row-reverse',
  },
  unreadItem: {
    backgroundColor: '#F0F9FF',
  },
  icon: {
    fontSize: 24,
  },
  notifTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1E293B',
    marginBottom: 4,
  },
  notifMessage: {
    fontSize: 13,
    color: '#64748B',
    marginBottom: 4,
  },
  timestamp: {
    fontSize: 11,
    color: '#94A3B8',
  },
  deleteBtn: {
    padding: 8,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 100,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    color: '#64748B',
  },
})
