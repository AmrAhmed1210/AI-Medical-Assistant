import { describe, it, expect, beforeEach } from '@jest/globals';
import { useNotificationStore } from '../store/notificationStore';

describe('useNotificationStore', () => {
  beforeEach(() => {
    const store = useNotificationStore.getState();
    store.clearAllMessages();
  });

  it('should have initial state with zero notifications', () => {
    const state = useNotificationStore.getState();
    expect(state.unreadMessages).toBe(0);
    expect(state.unreadCounts).toEqual({});
    expect(state.latestMessagePayload).toBeNull();
  });

  it('should increment session messages correctly', () => {
    const store = useNotificationStore.getState();
    store.incrementSessionMessage(1);
    store.incrementSessionMessage(1);
    store.incrementSessionMessage(2);

    const state = useNotificationStore.getState();
    expect(state.unreadCounts[1]).toBe(2);
    expect(state.unreadCounts[2]).toBe(1);
    expect(state.unreadMessages).toBe(3);
  });

  it('should clear session messages correctly', () => {
    const store = useNotificationStore.getState();
    store.incrementSessionMessage(1);
    store.incrementSessionMessage(1);
    store.clearSessionMessages(1);

    const state = useNotificationStore.getState();
    expect(state.unreadCounts[1]).toBeUndefined();
    expect(state.unreadMessages).toBe(0);
  });

  it('should clear all messages', () => {
    const store = useNotificationStore.getState();
    store.incrementSessionMessage(1);
    store.incrementSessionMessage(2);
    store.clearAllMessages();

    const state = useNotificationStore.getState();
    expect(state.unreadMessages).toBe(0);
    expect(state.unreadCounts).toEqual({});
  });

  it('should set latest message payload', () => {
    const store = useNotificationStore.getState();
    const payload = { message: 'Hello', sender: 'Doctor' };
    store.setLatestMessagePayload(payload);

    const state = useNotificationStore.getState();
    expect(state.latestMessagePayload).toEqual(payload);
  });
});