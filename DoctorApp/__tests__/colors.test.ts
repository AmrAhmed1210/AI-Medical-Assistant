import { describe, it, expect } from '@jest/globals';
import { COLORS } from '../constants/colors';

describe('COLORS Constants', () => {
  it('should have primary color defined', () => {
    expect(COLORS.primary).toBe('#1E9E84');
  });

  it('should have background color defined', () => {
    expect(COLORS.background).toBe('#F4F6FA');
  });

  it('should have white and black colors defined', () => {
    expect(COLORS.white).toBe('#FFFFFF');
    expect(COLORS.black).toBe('#1C1C1C');
  });

  it('should have gray colors defined', () => {
    expect(COLORS.gray).toBe('#9E9E9E');
    expect(COLORS.lightGray).toBe('#EAEAEA');
  });
});