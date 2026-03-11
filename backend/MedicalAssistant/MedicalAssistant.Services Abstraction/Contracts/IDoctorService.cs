using MedicalAssistant.Shared.DTOs.DoctorDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IDoctorService
{
    Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync();

    // تم التحديث ليعيد تفاصيل الطبيب الكاملة بدلاً من النوع العام
    Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id);

    Task<IReadOnlyList<DoctorDTO>> GetAvailableDoctorsAsync();

    Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId);

    Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name);

    Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count);

    Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize);
}