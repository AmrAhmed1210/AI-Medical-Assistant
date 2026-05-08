import React from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet } from 'react-native';
import { Star, Heart } from 'lucide-react-native';
import { COLORS } from '../../constants/colors';

export interface DoctorCardProps {
  doctor: {
    id: string | number;
    name: string;
    specialty: string;
    rating: number;
    reviewCount: number;
    imageUrl?: string;
    isAvailable?: boolean;
  };
  onPress: () => void;
  isFollowed?: boolean;
  onFollowPress?: () => void;
}

const DoctorCard: React.FC<DoctorCardProps> = ({ doctor, onPress, isFollowed, onFollowPress }) => {
  return (
    <TouchableOpacity
      style={styles.card}
      activeOpacity={0.85}
      onPress={onPress}
    >
      <View style={styles.avatarWrap}>
        <View style={styles.avatar}>
          {doctor.imageUrl && !doctor.imageUrl.includes('default') ? (
            <Image source={{ uri: doctor.imageUrl }} style={styles.image} />
          ) : (
            <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3774/3774299.png' }} style={styles.image} />
          )}
        </View>
        {doctor.isAvailable && <View style={styles.onlineDot} />}
      </View>

      <Text style={styles.name} numberOfLines={1}>
        {doctor.name}
      </Text>

      <View style={styles.specBadge}>
        <Text style={styles.specText} numberOfLines={1}>
          {doctor.specialty}
        </Text>
      </View>

      <View style={styles.ratingRow}>
        <Star size={11} color="#F59E0B" fill="#F59E0B" />
        <Text style={styles.ratingValue}>{doctor.rating}</Text>
        <Text style={styles.reviewCount}>({doctor.reviewCount})</Text>
      </View>

      <TouchableOpacity 
        style={styles.bookBtn}
        onPress={onPress}
      >
        <Text style={styles.bookText}>Book</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    width: 140,
    height: 190,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 12,
    alignItems: 'center',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 2,
    marginRight: 12,
  },
  avatarWrap: {
    position: 'relative',
    marginBottom: 4,
  },
  avatar: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#F1F5F9',
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  onlineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#10B981',
    position: 'absolute',
    bottom: 2,
    right: 2,
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  name: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1E293B',
    textAlign: 'center',
  },
  specBadge: {
    backgroundColor: '#1D9E7515',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
    maxWidth: '100%',
  },
  specText: {
    fontSize: 11,
    color: '#1D9E75',
    fontWeight: '600',
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  ratingValue: {
    fontSize: 11,
    fontWeight: '600',
    color: '#1E293B',
  },
  reviewCount: {
    fontSize: 11,
    color: '#6B7280',
  },
  bookBtn: {
    backgroundColor: '#1D9E75',
    width: '100%',
    height: 32,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bookText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '600',
  },
});

export default DoctorCard;
