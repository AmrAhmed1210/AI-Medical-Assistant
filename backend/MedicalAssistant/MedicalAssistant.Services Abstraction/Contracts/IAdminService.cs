using MedicalAssistant.Shared.DTOs.Admin;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IAdminService
{
    Task<SystemStatsDto> GetSystemStatsAsync();
    Task<IEnumerable<UserManagementDto>> GetAllUsersAsync();
    Task<bool> ToggleUserStatusAsync(int userId);
}