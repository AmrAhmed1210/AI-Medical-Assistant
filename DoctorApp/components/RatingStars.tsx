import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface RatingStarsProps {
  rating: number;
  reviewCount?: number;
  size?: number;
  showText?: boolean;
  textColor?: string;
  countColor?: string;
}

export default function RatingStars({
  rating,
  reviewCount,
  size = 14,
  showText = true,
  textColor = '#1E293B',
  countColor = '#64748B'
}: RatingStarsProps) {
  const roundedRating = Number(rating || 0).toFixed(1);

  return (
    <View style={styles.container}>
      <View style={styles.starRow}>
        {[1, 2, 3, 4, 5].map((star) => (
          <Ionicons
            key={star}
            name={star <= Math.round(rating) ? "star" : (star - rating < 0.5 ? "star-half" : "star-outline")}
            size={size}
            color="#FFB300"
          />
        ))}
      </View>
      {showText && (
        <View style={styles.textRow}>
          <Text style={[styles.ratingText, { color: textColor, fontSize: size - 2 }]}>
            {roundedRating}
          </Text>
          {reviewCount !== undefined && (
            <Text style={[styles.countText, { color: countColor, fontSize: size - 3 }]}>
              ({reviewCount})
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  starRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 1,
  },
  textRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
  ratingText: {
    fontWeight: '700',
  },
  countText: {
    fontWeight: '500',
  },
});
