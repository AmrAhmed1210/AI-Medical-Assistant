using MedicalAssistant.Shared.DTOs.ConsultationDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IConsultationService
    {
        Task<ConsultationDto> CreateConsultationAsync(int doctorId, CreateConsultationDto dto);
        Task<ConsultationDto?> GetConsultationByIdAsync(int id);
        Task<IEnumerable<ConsultationDto>> GetConsultationsByPatientIdAsync(int patientId);
        Task<IEnumerable<ConsultationDto>> GetConsultationsByDoctorIdAsync(int doctorId);
        Task<ConsultationDto?> UpdateConsultationAsync(UpdateConsultationDto dto);
        Task<bool> DeleteConsultationAsync(int id);
        Task<bool> CompleteConsultationAsync(int id);
        Task<bool> CancelConsultationAsync(int id);
    }
}
