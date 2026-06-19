export interface ParsedAiReport {
  analysis_en: string
  analysis_ar: string
  needsDoctor?: boolean
}

export function parseAiDiagnosisSummary(raw?: string | null): ParsedAiReport | null {
  if (!raw?.trim()) return null

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const analysis_en = String(
      parsed.analysis_en ?? parsed.en ?? parsed.analysisEn ?? ''
    ).trim()
    const analysis_ar = String(
      parsed.analysis_ar ?? parsed.ar ?? parsed.analysisAr ?? ''
    ).trim()

    if (!analysis_en && !analysis_ar) return null

    return {
      analysis_en: analysis_en || analysis_ar,
      analysis_ar: analysis_ar || analysis_en,
      needsDoctor: parsed.needsDoctor as boolean | undefined,
    }
  } catch {
    return {
      analysis_en: raw.trim(),
      analysis_ar: raw.trim(),
    }
  }
}

export function getAiReportText(report: ParsedAiReport, lang: 'en' | 'ar'): string {
  if (lang === 'ar') {
    return report.analysis_ar || report.analysis_en || 'لم يتم إنشاء التقرير بالعربية بعد.'
  }
  return report.analysis_en || report.analysis_ar || 'No English report available yet.'
}

export function hasDistinctArabicReport(report: ParsedAiReport): boolean {
  const ar = report.analysis_ar?.trim()
  const en = report.analysis_en?.trim()
  if (!ar) return false
  if (!en) return true
  return ar !== en
}
