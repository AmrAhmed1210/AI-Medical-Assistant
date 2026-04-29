using MedicalAssistant.Domain.Entities.PatientModule;
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
    }
}
