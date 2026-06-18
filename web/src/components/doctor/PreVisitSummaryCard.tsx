import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2, AlertTriangle } from 'lucide-react';
import axiosInstance from '@/api/axiosInstance';
import type { AppointmentDto } from '@/lib/types';
import { Button } from '@/components/ui';
import { doctorApi } from '@/api/doctorApi';
import { patientRecordsApi } from '@/api/patientRecordsApi';

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
      let chronicDiseases: any[] = [];
      let medications: any[] = [];
      let allergies: any[] = [];
      let vitals: any[] = [];
      let patient: any = null;

      try {
        const patients = await doctorApi.getPatients();
        patient = patients.find((p: any) => String(p.id) === String(appointment.patientId));
      } catch (e) {
        console.error("Failed to load patient summary info", e);
      }

      try {
        [chronicDiseases, medications, allergies, vitals] = await Promise.all([
          patientRecordsApi.getChronicDiseases(appointment.patientId).catch(() => []),
          patientRecordsApi.getMedications(appointment.patientId).catch(() => []),
          patientRecordsApi.getAllergies(appointment.patientId).catch(() => []),
          patientRecordsApi.getVitals(appointment.patientId).catch(() => []),
        ]);
      } catch (e) {
        console.error("Failed to load patient record lists", e);
      }

      let age = 35; // Default fallback
      let gender = "Male"; // Default fallback
      if (patient) {
        gender = patient.gender || "Male";
        if (patient.dateOfBirth) {
          const birthDate = new Date(patient.dateOfBirth);
          const ageDifMs = Date.now() - birthDate.getTime();
          const ageDate = new Date(ageDifMs);
          age = Math.abs(ageDate.getUTCFullYear() - 1970);
        }
      }

      const diseases = chronicDiseases
        .filter((d: any) => d.isActive)
        .map((d: any) => d.diseaseName);

      const meds = medications
        .filter((m: any) => m.isActive)
        .map((m: any) => `${m.medicationName} ${m.dosage || ''}`);

      const allergyNames = allergies
        .filter((a: any) => a.isActive)
        .map((a: any) => a.allergenName);

      const vitalsList = vitals
        .slice(0, 5) // latest 5 vitals
        .map((v: any) => `${v.readingType}: ${v.value}${v.value2 ? '/' + v.value2 : ''} ${v.unit}`);

      const payload = {
        patient_id: String(appointment.patientId || "ANON-1234"),
        age,
        gender,
        chief_complaint: appointment.notes || "Routine checkup and follow up.",
        chronic_diseases: diseases.length > 0 ? diseases : ["None reported"],
        medications: meds.length > 0 ? meds : ["None reported"],
        allergies: allergyNames.length > 0 ? allergyNames : ["None reported"],
        vitals: vitalsList.length > 0 ? vitalsList : ["None reported"]
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
