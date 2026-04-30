using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Shared.DTOs.PatientRecords;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IPatientRecordService
    {
        // Medical profile
        Task<MedicalProfile?> GetMedicalProfileAsync(int patientId);
        Task<MedicalProfile> CreateMedicalProfileAsync(int patientId, MedicalProfile profile);
        Task<MedicalProfile?> UpdateMedicalProfileAsync(int patientId, MedicalProfile profileUpdates);

        // Surgeries
        Task<IEnumerable<SurgeryHistory>> GetSurgeriesAsync(int patientId);
        Task<SurgeryHistory> AddSurgeryAsync(int patientId, SurgeryHistory surgery);
        Task<SurgeryHistory?> UpdateSurgeryAsync(int surgeryId, SurgeryHistory surgeryUpdates);
        Task<bool> DeleteSurgeryAsync(int surgeryId);

        // Allergies
        Task<IEnumerable<AllergyRecord>> GetAllergiesAsync(int patientId);
        Task<AllergyRecord> AddAllergyAsync(int patientId, AllergyRecord allergy);
        Task<AllergyRecord?> UpdateAllergyAsync(int allergyId, AllergyRecord allergyUpdates);
        Task<bool> DeleteAllergyAsync(int allergyId);

        // Chronic diseases
        Task<IEnumerable<ChronicDiseaseMonitor>> GetChronicDiseasesAsync(int patientId);
        Task<ChronicDiseaseMonitor> AddChronicDiseaseAsync(int patientId, ChronicDiseaseMonitor monitor);
        Task<ChronicDiseaseMonitor?> UpdateChronicDiseaseAsync(int monitorId, ChronicDiseaseMonitor updates);
        Task<bool> DeactivateChronicDiseaseAsync(int monitorId);

        // Vital readings
        Task<IEnumerable<VitalReading>> GetVitalsAsync(int patientId, string? type = null, DateTime? from = null, DateTime? to = null);
        Task<VitalReading> AddVitalAsync(int patientId, VitalReading reading);
        Task<VitalReading?> GetLatestVitalAsync(int patientId, string type);
        Task<IEnumerable<VitalTrendPointDto>> GetVitalTrendAsync(int patientId, string type, int days);
        Task<bool> DeleteVitalAsync(int vitalId);

        // Medication trackers
        Task<IEnumerable<MedicationTracker>> GetMedicationsAsync(int patientId, bool activeOnly = true);
        Task<MedicationTracker> AddMedicationAsync(int patientId, MedicationTracker tracker);
        Task<MedicationTracker?> UpdateMedicationAsync(int medicationId, MedicationTracker updates);
        Task<bool> DeactivateMedicationAsync(int medicationId);
        Task<IEnumerable<MedicationScheduleItemDto>> GetTodayScheduleAsync(int patientId, DateTime? now = null);
        Task<IEnumerable<MedicationTracker>> GetLowStockMedicationsAsync(int patientId);
    }
}
