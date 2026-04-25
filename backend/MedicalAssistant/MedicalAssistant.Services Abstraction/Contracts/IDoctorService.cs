using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IDoctorService
{
    Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync();

    Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id);

    Task<IReadOnlyList<DoctorDTO>> GetAvailableDoctorsAsync();

    Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId);

    Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name);

    Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count);

    Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize);

    // Dashboard Methods
    Task<DoctorDashboardDto> GetDashboardStatsAsync(int doctorId);
    Task<DoctorDetailsDTO> GetDoctorProfileAsync(int doctorId);
    Task<IEnumerable<AppointmentDto>> GetDoctorAppointmentsAsync(int doctorId);
    Task<IEnumerable<MedicalAssistant.Shared.DTOs.PatientDTOs.PatientDto>> GetDoctorPatientsAsync(int doctorId);
    Task<IEnumerable<AIReportDto>> GetDoctorReportsAsync(int doctorId);
    Task<IEnumerable<AvailabilityDto>> GetDoctorAvailabilityAsync(int doctorId);
    Task<IEnumerable<AvailabilityDto>> GetPublicAvailabilityAsync(int doctorId);

    Task<DoctorDashboardDto> GetDashboardAsync(int doctorId);
    Task<DoctorDetailDto?> GetProfileAsync(int doctorId);
    Task UpdateProfileAsync(int doctorId, UpdateDoctorProfileRequest request);
    Task<IEnumerable<AppointmentDto>> GetAppointmentsAsync(int doctorId, string? status);
    Task<IEnumerable<PatientSummaryDto>> GetPatientsAsync(int doctorId, string? search);
    Task<IEnumerable<AIReportDto>> GetReportsAsync(int doctorId, string? urgency);
    Task<IEnumerable<AvailabilityDto>> GetAvailabilityAsync(int doctorId);
    Task UpdateAvailabilityAsync(int doctorId, List<AvailabilityDto> data);
    Task<DoctorScheduleDto?> GetScheduleAsync(int doctorId);
    Task<DoctorScheduleDto?> GetMyScheduleAsync(int doctorUserId);
    Task UpdateScheduleAsync(int doctorUserId, UpdateDoctorScheduleRequest request);
    Task<IEnumerable<SpecialtyDto>> GetSpecialtiesAsync();

    Task<IEnumerable<MedicalAssistant.Shared.DTOs.ReviewDTOs.ReviewDto>> GetMyReviewsAsync(int doctorUserId);

    Task UpdateScheduleVisibilityAsync(int doctorUserId, bool isVisible);

    // Clear history
    Task ClearAppointmentHistoryAsync(int doctorUserId);

    // Self-deactivate account
    Task<bool> SelfDeactivateAsync(int doctorUserId);

    Task ApplyForDoctorAccountAsync(ApplyDoctorRequest request);
    Task UpdatePhotoAsync(int doctorUserId, string photoUrl);
    Task UpdateApplicationDocumentAsync(int applicationId, string documentUrl);
}
