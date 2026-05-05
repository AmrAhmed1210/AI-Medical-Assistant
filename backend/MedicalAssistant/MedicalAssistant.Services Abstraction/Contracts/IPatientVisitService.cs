using MedicalAssistant.Shared.DTOs.PatientVisits;
using Microsoft.AspNetCore.Http;
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
        Task<IEnumerable<PatientVisitDto>> GetMyVisitsAsync(int patientUserId);
        Task<VisitSummaryDto?> GetVisitSummaryAsync(int doctorUserId, int visitId);
        Task<VisitSummaryDto?> GetVisitSummaryForPatientAsync(int patientUserId, int visitId);
        Task<PatientHistoryDto?> GetPatientHistoryAsync(int doctorUserId, int patientId);

        // Symptoms
        Task<VisitSymptomDto> AddSymptomAsync(int doctorOrNurseUserId, int visitId, CreateVisitSymptomDto dto);
        Task<IEnumerable<VisitSymptomDto>> GetSymptomsAsync(int visitId);
        Task<bool> DeleteSymptomAsync(int doctorUserId, int symptomId);
        Task<IEnumerable<VisitSymptomDto>> GetSymptomHistoryForPatientAsync(int doctorUserId, int patientId);

        // Clinical vitals
        Task<VisitVitalDto> AddVisitVitalAsync(int doctorOrNurseUserId, int visitId, CreateVisitVitalDto dto);
        Task<IEnumerable<VisitVitalDto>> GetVisitVitalsAsync(int visitId);
        Task<bool> DeleteClinicalVitalAsync(int vitalId);

        // Prescriptions
        Task<VisitPrescriptionDto> AddPrescriptionAsync(int doctorUserId, int visitId, CreateVisitPrescriptionDto dto);
        Task<IEnumerable<VisitPrescriptionDto>> GetPrescriptionsAsync(int visitId);
        Task<VisitPrescriptionDto?> UpdatePrescriptionAsync(int doctorUserId, int prescriptionId, UpdateVisitPrescriptionDto dto);
        Task<bool> DeletePrescriptionAsync(int doctorUserId, int prescriptionId);
        Task<bool> DispensePrescriptionAsync(int pharmacistUserId, int prescriptionId);

        // Documents
        Task<VisitDocumentDto> UploadDocumentAsync(int uploaderUserId, int visitId, IFormFile file, string documentType, string title, string? description);
        Task<IEnumerable<VisitDocumentDto>> GetDocumentsAsync(int visitId);
        Task<bool> DeleteDocumentAsync(int requesterUserId, int documentId);
    }
}
