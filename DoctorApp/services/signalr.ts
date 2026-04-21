import * as signalR from '@microsoft/signalr'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { BASE_URL } from '../constants/api'

let connection: signalR.HubConnection | null = null

const waitForConnected = async (timeoutMs = 8000): Promise<signalR.HubConnection | null> => {
  const startedAt = Date.now()
  while (connection && Date.now() - startedAt < timeoutMs) {
    if (connection.state === signalR.HubConnectionState.Connected) return connection
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  return connection?.state === signalR.HubConnectionState.Connected ? connection : null
}

export async function startSignalRConnection(): Promise<signalR.HubConnection | null> {
  // Guard 1: Already connected
  if (connection?.state === signalR.HubConnectionState.Connected) return connection

  // Guard 2: Already connecting - wait for it instead of returning null
  if (connection?.state === signalR.HubConnectionState.Connecting) {
    return waitForConnected()
  }

  // Guard 3: No token yet - don't even try
  const token = await AsyncStorage.getItem('token')
  if (!token) {
    console.log('SignalR: No token, skipping connection')
    return null
  }

  // Stop any existing broken connection first
  if (connection) {
    try { await connection.stop() } catch {}
    connection = null
  }

  // Now create fresh connection
  connection = new signalR.HubConnectionBuilder()
    .withUrl(`${BASE_URL}/hubs/notifications`, {
      accessTokenFactory: async () =>
        await AsyncStorage.getItem('token') ?? '',
    })
    .withAutomaticReconnect([2000, 5000, 10000, 30000])
    .configureLogging(signalR.LogLevel.Warning)
    .build()

  try {
    await connection.start()
    console.log('SignalR connected successfully')
  } catch (err) {
    console.warn('SignalR failed to connect:', err)
    connection = null
  }

  return connection
}

export async function stopSignalRConnection(): Promise<void> {
  if (!connection) return
  try {
    await connection.stop()
  } catch (e) {
    // ignore
  } finally {
    connection = null
  }
}

export function onDoctorCreated(callback: (data: any) => void) {
  connection?.on('DoctorCreated', callback)
  return () => connection?.off('DoctorCreated', callback)
}

export function onDoctorUpdated(callback: (data: any) => void) {
  connection?.on('DoctorUpdated', callback)
  return () => connection?.off('DoctorUpdated', callback)
}

export function onAppointmentUpdated(callback: (data: any) => void) {
  connection?.on('AppointmentUpdated', callback)
  return () => connection?.off('AppointmentUpdated', callback)
}

export function onScheduleReady(callback: (data: any) => void) {
  connection?.on('ScheduleReady', callback)
  return () => connection?.off('ScheduleReady', callback)
}

export function onScheduleUpdated(callback: (data: any) => void) {
  connection?.on('ScheduleUpdated', callback)
  return () => connection?.off('ScheduleUpdated', callback)
}

export function onNotificationReceived(callback: (data: any) => void) {
  connection?.on('NotificationReceived', callback)
  return () => connection?.off('NotificationReceived', callback)
}

export async function subscribeToDoctorSchedule(doctorId: number) {
  let conn =
    connection?.state === signalR.HubConnectionState.Connected
      ? connection
      : await startSignalRConnection()

  if (conn?.state !== signalR.HubConnectionState.Connected) {
    conn = await waitForConnected()
  }

  if (conn?.state === signalR.HubConnectionState.Connected) {
    await conn.invoke('SubscribeToDoctorSchedule', doctorId)
  }
}
