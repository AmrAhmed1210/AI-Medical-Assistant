using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Contracts;

public interface IAdminRepository
{
    Task<int> GetCountAsync<T>() where T : class;
    Task<IEnumerable<User>> GetAllUsersAsync();
    Task<bool> UpdateUserStatusAsync(Guid userId, bool isActive);
}