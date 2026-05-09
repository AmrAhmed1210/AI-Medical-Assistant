import React from 'react';
import { render, screen } from '@testing-library/react-native';
import RatingStars from '../components/RatingStars';

describe('RatingStars Component', () => {
  it('renders with default props', () => {
    render(<RatingStars rating={4.5} />);
    expect(screen.getByText('4.5')).toBeTruthy();
  });

  it('renders rating with review count', () => {
    render(<RatingStars rating={4.5} reviewCount={25} />);
    expect(screen.getByText('4.5')).toBeTruthy();
    expect(screen.getByText('(25)')).toBeTruthy();
  });

  it('displays exact rating value', () => {
    render(<RatingStars rating={3.7} />);
    expect(screen.getByText('3.7')).toBeTruthy();
  });

  it('handles zero rating', () => {
    render(<RatingStars rating={0} />);
    expect(screen.getByText('0.0')).toBeTruthy();
  });

  it('handles full rating', () => {
    render(<RatingStars rating={5} />);
    expect(screen.getByText('5.0')).toBeTruthy();
  });

  it('rounds rating to one decimal', () => {
    render(<RatingStars rating={4.567} />);
    expect(screen.getByText('4.6')).toBeTruthy();
  });

  it('accepts custom size', () => {
    const { toJSON } = render(<RatingStars rating={4} size={20} />);
    expect(toJSON()).toBeTruthy();
  });

  it('hides text when showText is false', () => {
    render(<RatingStars rating={4.5} showText={false} />);
    expect(screen.queryByText('4.5')).toBeNull();
  });

  it('handles null/undefined rating', () => {
    const { toJSON } = render(<RatingStars rating={undefined as any} />);
    expect(toJSON()).toBeTruthy();
  });
});