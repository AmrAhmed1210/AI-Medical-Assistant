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
    <div className="py-2">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h4 className="flex items-center gap-2 text-slate-700 dark:text-slate-300 font-bold text-sm">
          <Sparkles className="w-4 h-4 text-primary-500" />
          AI Pre-Visit Summary
        </h4>
        {!summary && !loading && (
          <button 
            onClick={generateSummary} 
            className="self-start sm:self-auto text-xs font-bold text-primary-600 bg-primary-50 hover:bg-primary-100 dark:text-primary-400 dark:bg-primary-900/20 dark:hover:bg-primary-900/40 px-4 py-2 rounded-xl transition-colors"
          >
            Generate Summary
          </button>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-primary-500 text-sm mt-4 font-medium bg-primary-50/50 dark:bg-primary-900/10 p-3 rounded-xl border border-primary-100 dark:border-primary-800/30 w-fit">
          <Loader2 className="w-4 h-4 animate-spin" />
          Analyzing patient history...
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm bg-red-50 dark:bg-red-900/20 p-3 rounded-xl mt-4 border border-red-100 dark:border-red-900/30">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {summary && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="grid md:grid-cols-2 gap-4 mt-5">
          <div className="bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-5 border border-slate-100 dark:border-slate-700/50" dir="ltr">
            <h5 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400"></span>
              English Summary
            </h5>
            <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap leading-relaxed font-medium">
              {summary.summary_en}
            </p>
          </div>
          
          <div className="bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-5 border border-slate-100 dark:border-slate-700/50" dir="rtl">
            <h5 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2" dir="ltr">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              Arabic Summary
            </h5>
            <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap leading-relaxed font-medium font-tajawal">
              {summary.summary_ar}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
