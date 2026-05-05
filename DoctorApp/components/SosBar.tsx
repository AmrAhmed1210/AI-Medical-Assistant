import React from "react";
import { View, Text, StyleSheet } from "react-native";

export interface SosBarProps {
  bloodType?: string;
  allergies?: Array<{ allergenName: string; severity: string; reaction: string }>;
  emergencyContact?: { name: string; phone: string };
}

export function SosBar({ bloodType, allergies = [], emergencyContact }: SosBarProps) {
  const lifeThreatening = allergies.filter(
    (a) => a.severity.toLowerCase().includes("life") || a.severity.toLowerCase().includes("anaphylaxis")
  );
  const severe = allergies.filter(
    (a) => a.severity.toLowerCase().includes("severe") && !a.severity.toLowerCase().includes("life")
  );

  const hasCritical = lifeThreatening.length > 0 || severe.length > 0;

  return (
    <View style={[styles.container, hasCritical && styles.criticalContainer]}>
      <View style={styles.row}>
        {bloodType && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>🩸 {bloodType}</Text>
          </View>
        )}

        {lifeThreatening.length > 0 && (
          <View style={[styles.badge, styles.lifeThreateningBadge]}>
            <Text style={[styles.badgeText, styles.lifeThreateningText]}>
              ⚠ {lifeThreatening.map((a) => a.allergenName).join(", ")}
            </Text>
          </View>
        )}

        {severe.length > 0 && lifeThreatening.length === 0 && (
          <View style={[styles.badge, styles.severeBadge]}>
            <Text style={[styles.badgeText, styles.severeText]}>
              ⚠ {severe.map((a) => a.allergenName).join(", ")}
            </Text>
          </View>
        )}

        {emergencyContact && (
          <View style={styles.contactBadge}>
            <Text style={styles.contactText}>
              📞 {emergencyContact.name} — {emergencyContact.phone}
            </Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#E53935",
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  criticalContainer: {
    backgroundColor: "#C62828",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 8,
  },
  badge: {
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginRight: 6,
    marginBottom: 4,
  },
  badgeText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
  },
  lifeThreateningBadge: {
    backgroundColor: "#FFEBEE",
  },
  lifeThreateningText: {
    color: "#C62828",
  },
  severeBadge: {
    backgroundColor: "#FFF3E0",
  },
  severeText: {
    color: "#E65100",
  },
  contactBadge: {
    backgroundColor: "rgba(255,255,255,0.1)",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  contactText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "600",
  },
});
