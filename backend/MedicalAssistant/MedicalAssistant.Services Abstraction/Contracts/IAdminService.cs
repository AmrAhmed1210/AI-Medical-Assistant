using MedicalAssistant.Shared.DTOs.Admin;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IAdminService
{
    // 📊 Stats
    Task<SystemStatsDto> GetSystemStatsAsync();

    // 👥 Users
    Task<object> GetUsersAsync(
        int page = 1,
        int pageSize = 10,
        string? search = null,
        string? role = null);

    // 🔄 Toggle
    Task<bool> ToggleUserStatusAsync(int userId);

    // ❌ Delete
    Task<bool> DeleteUserAsync(int id);

    // ➕ Create
    Task<object> CreateUserAsync(CreateUserRequest request);
}