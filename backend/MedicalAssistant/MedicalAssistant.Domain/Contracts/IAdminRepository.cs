using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IAdminRepository
    {
        Task<int> GetCountAsync<T>() where T : class;
        Task<(IEnumerable<User> Items, int TotalCount)> GetUsersAsync(int page, int pageSize, string? search, string? role);
        Task<bool> UpdateUserStatusAsync(int userId, bool isActive);
        Task<bool> DeleteUserAsync(int userId);
    }
}