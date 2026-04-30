using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientRecords;

namespace MedicalAssistant.Services.Services
{
    public class PatientRecordService : IPatientRecordService
    {
        private readonly IUnitOfWork _unitOfWork;

        public PatientRecordService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        // Medical profile
        public async Task<MedicalProfile?> GetMedicalProfileAsync(int patientId)
        {
            return (await _unitOfWork.Repository<MedicalProfile>().FindAsync(p => p.PatientId == patientId)).FirstOrDefault();
        }

        public async Task<MedicalProfile> CreateMedicalProfileAsync(int patientId, MedicalProfile profile)
        {
            profile.PatientId = patientId;
            profile.CreatedAt = DateTime.UtcNow;
            profile.UpdatedAt = DateTime.UtcNow;
            await _unitOfWork.Repository<MedicalProfile>().AddAsync(profile);
            await _unitOfWork.SaveChangesAsync();
            return profile;
        }

        public async Task<MedicalProfile?> UpdateMedicalProfileAsync(int patientId, MedicalProfile profileUpdates)
        {
            var existing = (await _unitOfWork.Repository<MedicalProfile>().FindAsync(p => p.PatientId == patientId)).FirstOrDefault();
            if (existing == null) return null;

            // apply updates if provided
            if (profileUpdates.BloodType != null) existing.BloodType = profileUpdates.BloodType;
            if (profileUpdates.WeightKg.HasValue) existing.WeightKg = profileUpdates.WeightKg;
            if (profileUpdates.HeightCm.HasValue) existing.HeightCm = profileUpdates.HeightCm;
            existing.IsSmoker = profileUpdates.IsSmoker;
            if (profileUpdates.SmokingDetails != null) existing.SmokingDetails = profileUpdates.SmokingDetails;
            existing.DrinksAlcohol = profileUpdates.DrinksAlcohol;
            if (profileUpdates.ExerciseHabits != null) existing.ExerciseHabits = profileUpdates.ExerciseHabits;
            if (profileUpdates.EmergencyContactName != null) existing.EmergencyContactName = profileUpdates.EmergencyContactName;
            if (profileUpdates.EmergencyContactPhone != null) existing.EmergencyContactPhone = profileUpdates.EmergencyContactPhone;
            if (profileUpdates.EmergencyContactRelation != null) existing.EmergencyContactRelation = profileUpdates.EmergencyContactRelation;

            existing.UpdatedAt = DateTime.UtcNow;
            _unitOfWork.Repository<MedicalProfile>().Update(existing);
            await _unitOfWork.SaveChangesAsync();

            return existing;
        }

        // Surgeries
        public async Task<IEnumerable<SurgeryHistory>> GetSurgeriesAsync(int patientId)
        {
            return await _unitOfWork.Repository<SurgeryHistory>().FindAsync(s => s.PatientId == patientId);
        }

        public async Task<SurgeryHistory> AddSurgeryAsync(int patientId, SurgeryHistory surgery)
        {
            surgery.PatientId = patientId;
            surgery.CreatedAt = DateTime.UtcNow;
            await _unitOfWork.Repository<SurgeryHistory>().AddAsync(surgery);
            await _unitOfWork.SaveChangesAsync();
            return surgery;
        }

        public async Task<SurgeryHistory?> UpdateSurgeryAsync(int surgeryId, SurgeryHistory surgeryUpdates)
        {
            var existing = await _unitOfWork.Repository<SurgeryHistory>().GetByIdAsync(surgeryId);
            if (existing == null) return null;

            if (surgeryUpdates.SurgeryName != null) existing.SurgeryName = surgeryUpdates.SurgeryName;
            if (surgeryUpdates.SurgeryDate != default) existing.SurgeryDate = surgeryUpdates.SurgeryDate;
            if (surgeryUpdates.HospitalName != null) existing.HospitalName = surgeryUpdates.HospitalName;
            if (surgeryUpdates.DoctorName != null) existing.DoctorName = surgeryUpdates.DoctorName;
            if (surgeryUpdates.Complications != null) existing.Complications = surgeryUpdates.Complications;
            if (surgeryUpdates.Notes != null) existing.Notes = surgeryUpdates.Notes;

            _unitOfWork.Repository<SurgeryHistory>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return existing;
        }

        public async Task<bool> DeleteSurgeryAsync(int surgeryId)
        {
            var existing = await _unitOfWork.Repository<SurgeryHistory>().GetByIdAsync(surgeryId);
            if (existing == null) return false;
            _unitOfWork.Repository<SurgeryHistory>().Delete(existing);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        // Allergies
        public async Task<IEnumerable<AllergyRecord>> GetAllergiesAsync(int patientId)
        {
            return await _unitOfWork.Repository<AllergyRecord>().FindAsync(a => a.PatientId == patientId);
        }

        public async Task<AllergyRecord> AddAllergyAsync(int patientId, AllergyRecord allergy)
        {
            allergy.PatientId = patientId;
            allergy.CreatedAt = DateTime.UtcNow;
            await _unitOfWork.Repository<AllergyRecord>().AddAsync(allergy);
            await _unitOfWork.SaveChangesAsync();
            return allergy;
        }

        public async Task<AllergyRecord?> UpdateAllergyAsync(int allergyId, AllergyRecord allergyUpdates)
        {
            var existing = await _unitOfWork.Repository<AllergyRecord>().GetByIdAsync(allergyId);
            if (existing == null) return null;

            if (allergyUpdates.AllergyType != null) existing.AllergyType = allergyUpdates.AllergyType;
            if (allergyUpdates.AllergenName != null) existing.AllergenName = allergyUpdates.AllergenName;
            if (allergyUpdates.Severity != null) existing.Severity = allergyUpdates.Severity;
            if (allergyUpdates.ReactionDescription != null) existing.ReactionDescription = allergyUpdates.ReactionDescription;
            if (allergyUpdates.FirstObservedDate != default) existing.FirstObservedDate = allergyUpdates.FirstObservedDate;
            existing.IsActive = allergyUpdates.IsActive;

            _unitOfWork.Repository<AllergyRecord>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return existing;
        }

        public async Task<bool> DeleteAllergyAsync(int allergyId)
        {
            var existing = await _unitOfWork.Repository<AllergyRecord>().GetByIdAsync(allergyId);
            if (existing == null) return false;
            _unitOfWork.Repository<AllergyRecord>().Delete(existing);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        // Chronic diseases
        public async Task<IEnumerable<ChronicDiseaseMonitor>> GetChronicDiseasesAsync(int patientId)
        {
            return await _unitOfWork.Repository<ChronicDiseaseMonitor>()
                .FindAsync(c => c.PatientId == patientId);
        }

        public async Task<ChronicDiseaseMonitor> AddChronicDiseaseAsync(int patientId, ChronicDiseaseMonitor monitor)
        {
            monitor.PatientId = patientId;
            monitor.CreatedAt = DateTime.UtcNow;
            monitor.UpdatedAt = DateTime.UtcNow;
            await _unitOfWork.Repository<ChronicDiseaseMonitor>().AddAsync(monitor);
            await _unitOfWork.SaveChangesAsync();
            return monitor;
        }

        public async Task<ChronicDiseaseMonitor?> UpdateChronicDiseaseAsync(int monitorId, ChronicDiseaseMonitor updates)
        {
            var existing = await _unitOfWork.Repository<ChronicDiseaseMonitor>().GetByIdAsync(monitorId);
            if (existing == null) return null;

            if (!string.IsNullOrWhiteSpace(updates.DiseaseName)) existing.DiseaseName = updates.DiseaseName;
            if (!string.IsNullOrWhiteSpace(updates.DiseaseType)) existing.DiseaseType = updates.DiseaseType;
            if (updates.DiagnosedDate.HasValue) existing.DiagnosedDate = updates.DiagnosedDate;
            if (!string.IsNullOrWhiteSpace(updates.Severity)) existing.Severity = updates.Severity;
            if (!string.IsNullOrWhiteSpace(updates.MonitoringFrequency)) existing.MonitoringFrequency = updates.MonitoringFrequency;
            if (updates.DoctorNotes != null) existing.DoctorNotes = updates.DoctorNotes;
            if (updates.TargetValues != null) existing.TargetValues = updates.TargetValues;
            if (updates.LastCheckDate.HasValue) existing.LastCheckDate = updates.LastCheckDate;

            existing.IsActive = updates.IsActive;
            existing.UpdatedAt = DateTime.UtcNow;

            _unitOfWork.Repository<ChronicDiseaseMonitor>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return existing;
        }

        public async Task<bool> DeactivateChronicDiseaseAsync(int monitorId)
        {
            var existing = await _unitOfWork.Repository<ChronicDiseaseMonitor>().GetByIdAsync(monitorId);
            if (existing == null) return false;

            existing.IsActive = false;
            existing.UpdatedAt = DateTime.UtcNow;
            _unitOfWork.Repository<ChronicDiseaseMonitor>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        // Vital readings
        public async Task<IEnumerable<VitalReading>> GetVitalsAsync(int patientId, string? type = null, DateTime? from = null, DateTime? to = null)
        {
            var vitals = await _unitOfWork.Repository<VitalReading>().FindAsync(v =>
                v.PatientId == patientId
                && (type == null || v.ReadingType == type)
                && (!from.HasValue || v.RecordedAt >= from.Value)
                && (!to.HasValue || v.RecordedAt <= to.Value));

            return vitals.OrderByDescending(v => v.RecordedAt);
        }

        public async Task<VitalReading> AddVitalAsync(int patientId, VitalReading reading)
        {
            reading.PatientId = patientId;
            reading.RecordedAt = reading.RecordedAt == default ? DateTime.UtcNow : reading.RecordedAt;

            await _unitOfWork.Repository<VitalReading>().AddAsync(reading);
            await _unitOfWork.SaveChangesAsync();
            return reading;
        }

        public async Task<VitalReading?> GetLatestVitalAsync(int patientId, string type)
        {
            var items = await _unitOfWork.Repository<VitalReading>().FindAsync(v => v.PatientId == patientId && v.ReadingType == type);
            return items.OrderByDescending(v => v.RecordedAt).FirstOrDefault();
        }

        public async Task<IEnumerable<VitalTrendPointDto>> GetVitalTrendAsync(int patientId, string type, int days)
        {
            var from = DateTime.UtcNow.AddDays(-Math.Max(1, days));
            var items = await _unitOfWork.Repository<VitalReading>().FindAsync(v =>
                v.PatientId == patientId && v.ReadingType == type && v.RecordedAt >= from);

            return items
                .OrderBy(v => v.RecordedAt)
                .Select(v => new VitalTrendPointDto(v.RecordedAt, v.Value, v.Value2))
                .ToList();
        }

        public async Task<bool> DeleteVitalAsync(int vitalId)
        {
            var existing = await _unitOfWork.Repository<VitalReading>().GetByIdAsync(vitalId);
            if (existing == null) return false;
            _unitOfWork.Repository<VitalReading>().Delete(existing);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        // Medication trackers
        public async Task<IEnumerable<MedicationTracker>> GetMedicationsAsync(int patientId, bool activeOnly = true)
        {
            var items = await _unitOfWork.Repository<MedicationTracker>().FindAsync(m =>
                m.PatientId == patientId && (!activeOnly || m.IsActive));

            return items.OrderByDescending(m => m.CreatedAt);
        }

        public async Task<MedicationTracker> AddMedicationAsync(int patientId, MedicationTracker tracker)
        {
            tracker.PatientId = patientId;
            tracker.CreatedAt = DateTime.UtcNow;
            await _unitOfWork.Repository<MedicationTracker>().AddAsync(tracker);
            await _unitOfWork.SaveChangesAsync();
            return tracker;
        }

        public async Task<MedicationTracker?> UpdateMedicationAsync(int medicationId, MedicationTracker updates)
        {
            var existing = await _unitOfWork.Repository<MedicationTracker>().GetByIdAsync(medicationId);
            if (existing == null) return null;

            if (!string.IsNullOrWhiteSpace(updates.MedicationName)) existing.MedicationName = updates.MedicationName;
            if (updates.GenericName != null) existing.GenericName = updates.GenericName;
            if (!string.IsNullOrWhiteSpace(updates.Dosage)) existing.Dosage = updates.Dosage;
            if (!string.IsNullOrWhiteSpace(updates.Form)) existing.Form = updates.Form;
            if (!string.IsNullOrWhiteSpace(updates.Frequency)) existing.Frequency = updates.Frequency;
            if (updates.TimesPerDay > 0) existing.TimesPerDay = updates.TimesPerDay;
            if (!string.IsNullOrWhiteSpace(updates.DoseTimes)) existing.DoseTimes = updates.DoseTimes;
            if (updates.StartDate != default) existing.StartDate = updates.StartDate;
            if (updates.EndDate.HasValue) existing.EndDate = updates.EndDate;
            if (updates.Instructions != null) existing.Instructions = updates.Instructions;
            if (updates.PillsRemaining.HasValue) existing.PillsRemaining = updates.PillsRemaining;
            if (updates.RefillThreshold > 0) existing.RefillThreshold = updates.RefillThreshold;

            existing.IsChronic = updates.IsChronic;
            existing.IsActive = updates.IsActive;
            existing.ChronicDiseaseMonitorId = updates.ChronicDiseaseMonitorId;

            _unitOfWork.Repository<MedicationTracker>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return existing;
        }

        public async Task<bool> DeactivateMedicationAsync(int medicationId)
        {
            var existing = await _unitOfWork.Repository<MedicationTracker>().GetByIdAsync(medicationId);
            if (existing == null) return false;

            existing.IsActive = false;
            _unitOfWork.Repository<MedicationTracker>().Update(existing);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<IEnumerable<MedicationScheduleItemDto>> GetTodayScheduleAsync(int patientId, DateTime? now = null)
        {
            var baseline = now ?? DateTime.UtcNow;
            var start = baseline.Date;
            var end = baseline.Date.AddDays(1);

            var logs = await _unitOfWork.Repository<MedicationLog>()
                .FindAsync(l => l.PatientId == patientId && l.ScheduledAt >= start && l.ScheduledAt < end);

            var trackerIds = logs.Select(l => l.MedicationTrackerId).Distinct().ToList();
            var trackers = await _unitOfWork.Repository<MedicationTracker>().FindAsync(m => trackerIds.Contains(m.Id));
            var trackerMap = trackers.ToDictionary(t => t.Id, t => t);

            return logs
                .OrderBy(l => l.ScheduledAt)
                .Select(l =>
                {
                    var tr = trackerMap.GetValueOrDefault(l.MedicationTrackerId);
                    return new MedicationScheduleItemDto(
                        l.MedicationTrackerId,
                        l.ScheduledAt,
                        tr?.MedicationName ?? string.Empty,
                        tr?.Dosage ?? string.Empty,
                        l.Status
                    );
                })
                .ToList();
        }

        public async Task<IEnumerable<MedicationTracker>> GetLowStockMedicationsAsync(int patientId)
        {
            var items = await _unitOfWork.Repository<MedicationTracker>().FindAsync(m =>
                m.PatientId == patientId
                && m.IsActive
                && m.PillsRemaining.HasValue
                && m.PillsRemaining.Value <= m.RefillThreshold);

            return items.OrderBy(m => m.PillsRemaining);
        }
    }
}
