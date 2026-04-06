using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Http;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IDoctorService
{
    // --- Public Endpoints ---
    Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync();

    Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id);

    // التعديل هنا: البحث بـ int specialtyId بدلاً من string specialty
    // لأننا فصلنا التخصصات في جدول منفصل بـ IDs
    Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId);

    Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name);

    Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count);

    Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize);

    // --- Doctor Specific Endpoints ---
    Task<DoctorDashboardDto> GetDoctorDashboardAsync(int doctorId);

    Task<IEnumerable<AvailabilityDto>> GetAvailabilityAsync(int doctorId);

    Task UpdateAvailabilityAsync(int doctorId, IEnumerable<AvailabilityDto> slots);

    Task UpdateProfileAsync(int doctorId, DoctorUpdateDto dto);

    Task<string> UploadProfilePhotoAsync(int doctorId, IFormFile file);

    Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorAsync(int doctorId, string? status);

    Task<IEnumerable<PatientDto>> GetPatientsByDoctorAsync(int doctorId, string? search);

    Task<IEnumerable<AIReportDto>> GetAIReportsAsync(int doctorId, string? urgency, int? patientId);
}