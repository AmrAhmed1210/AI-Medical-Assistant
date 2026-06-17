import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2, AlertTriangle } from 'lucide-react';
import axiosInstance from '@/api/axiosInstance';
import type { AppointmentDto } from '@/lib/types';
import { Button } from '@/components/ui';

interface PreVisitSummaryCardProps {
  appointment: AppointmentDto;
}

export default function PreVisitSummaryCard({ appointment }: PreVisitSummaryCardProps) {
  const [summary, setSummary] = useState<{ summary_en: string; summary_ar: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateSummary = async () => {
    setLoading(true);
    setError('');
    try {
      // Mocked detailed info to simulate what the C# backend would aggregate
      // STRICT ANONYMIZATION: We only send the patient ID, not the name, to the AI.
      const payload = {
        patient_id: String(appointment.patientId || "ANON-1234"),
        age: Math.floor(Math.random() * (65 - 25 + 1) + 25), // Mocked age
        gender: "Male", // Mocked gender (clinically relevant but anonymous)
        chief_complaint: "Routine checkup and follow up on blood pressure.",
        chronic_diseases: ["Hypertension"], // Mocked
        medications: ["Concor 5mg"], // Mocked
        allergies: ["Penicillin"], // Mocked
        vitals: ["BP 145/95", "HR 82"] // Mocked abnormal vitals
      };

      const { data } = await axiosInstance.post('/api/chat/pre-visit-summary', payload);
      setSummary(data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to generate AI summary.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 bg-primary-50/50 rounded-xl p-4 border border-primary-100">
      <div className="flex items-center justify-between">
        <h4 className="flex items-center gap-2 text-primary-800 font-bold text-sm">
          <Sparkles className="w-4 h-4 text-primary-500" />
          AI Pre-Visit Summary
        </h4>
        {!summary && !loading && (
          <Button variant="outline" size="sm" onClick={generateSummary} className="h-8 text-xs bg-white border-primary-200 hover:bg-primary-50 text-primary-700">
            Generate Summary
          </Button>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-primary-600 text-sm mt-3">
          <Loader2 className="w-4 h-4 animate-spin" />
          Analyzing patient history...
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 text-red-600 text-sm bg-red-50 p-3 rounded-lg mt-3">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      {summary && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4 mt-4">
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100" dir="ltr">
            <h5 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">English Summary</h5>
            <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
              {summary.summary_en}
            </p>
          </div>
          
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100" dir="rtl">
            <h5 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2" dir="ltr">Arabic Summary</h5>
            <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
              {summary.summary_ar}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
