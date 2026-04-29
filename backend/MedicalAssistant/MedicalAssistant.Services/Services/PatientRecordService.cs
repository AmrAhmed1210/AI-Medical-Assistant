using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;

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
    }
}
