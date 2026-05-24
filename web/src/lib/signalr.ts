import * as signalR from '@microsoft/signalr'

const HUB_URL = import.meta.env.VITE_SIGNALR_HUB_URL || 'http://localhost:5194/hubs/notifications'

let connection: signalR.HubConnection | null = null

export function getSignalRConnection(token: string): signalR.HubConnection {
  if (connection && connection.state === signalR.HubConnectionState.Connected) {
    return connection
  }

  connection = new signalR.HubConnectionBuilder()
    .withUrl(HUB_URL, {
      accessTokenFactory: () => token,
    })
    .withAutomaticReconnect([0, 2000, 5000, 10000, 30000])
    .configureLogging(signalR.LogLevel.Warning)
    .build()

  return connection
}

export async function startConnection(token: string): Promise<signalR.HubConnection> {
  const conn = getSignalRConnection(token)
  if (conn.state === signalR.HubConnectionState.Disconnected) {
    await conn.start()
  }
  return conn
}

export async function stopConnection(): Promise<void> {
  if (connection && connection.state !== signalR.HubConnectionState.Disconnected) {
    await connection.stop()
  }
}

export { connection }
