using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IAppointmentService
    {
        Task<AppointmentDto> CreateAppointmentAsync(CreateAppointmentDto dto);
        Task<AppointmentDto?> GetAppointmentByIdAsync(int id);
        Task<IEnumerable<AppointmentDto>> GetAppointmentsByPatientIdAsync(int patientId);
        Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorIdAsync(int doctorId);
        Task<AppointmentDto?> UpdateAppointmentAsync(UpdateAppointmentDto dto);
        Task<bool> DeleteAppointmentAsync(int id);
        Task<PaginatedResultDto<AppointmentDto>> GetPaginatedAppointmentsAsync(int pageNumber, int pageSize);
    }
}
