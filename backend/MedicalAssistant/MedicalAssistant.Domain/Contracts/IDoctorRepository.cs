using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Contracts;

public interface IDoctorRepository : IGenericRepository<Doctor>
{
    Task<IEnumerable<Doctor>> GetAvailableDoctorsAsync();

    Task<IEnumerable<Doctor>> GetUnavailableDoctorsAsync();

    Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId);

    Task<IEnumerable<Doctor>> SearchByNameAsync(string name);

    Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count);

    Task<(IEnumerable<Doctor> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);
}