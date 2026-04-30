using MedicalAssistant.Shared.DTOs.PatientVisits;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IPatientVisitService
    {
        Task<PatientVisitDto> OpenVisitAsync(int doctorUserId, CreateVisitDto dto);
        Task<PatientVisitDto?> GetVisitAsync(int visitId);
        Task<PatientVisitDto?> UpdateVisitAsync(int doctorUserId, int visitId, UpdateVisitDto dto);
        Task<bool> CloseVisitAsync(int doctorUserId, int visitId);
        Task<IEnumerable<PatientVisitDto>> GetVisitsForPatientAsync(int patientId);
        Task<IEnumerable<PatientVisitDto>> GetTodayVisitsForDoctorAsync(int doctorUserId);
        Task<VisitSummaryDto?> GetVisitSummaryAsync(int doctorUserId, int visitId);

        // Symptoms
        Task<VisitSymptomDto> AddSymptomAsync(int doctorOrNurseUserId, int visitId, CreateVisitSymptomDto dto);
        Task<IEnumerable<VisitSymptomDto>> GetSymptomsAsync(int visitId);
        Task<bool> DeleteSymptomAsync(int doctorUserId, int symptomId);
        Task<IEnumerable<VisitSymptomDto>> GetSymptomHistoryForPatientAsync(int doctorUserId, int patientId);

        // Clinical vitals
        Task<VisitVitalDto> AddVisitVitalAsync(int doctorOrNurseUserId, int visitId, CreateVisitVitalDto dto);
        Task<IEnumerable<VisitVitalDto>> GetVisitVitalsAsync(int visitId);
        Task<bool> DeleteClinicalVitalAsync(int vitalId);
    }
}
