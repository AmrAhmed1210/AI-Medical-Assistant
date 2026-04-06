using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.Common;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IAdminService
    {
        Task<SystemStatsDto> GetSystemStatsAsync();

        Task<PagedResult<UserManagementDto>> GetUsersAsync(
            int page = 1,
            int pageSize = 20,
            string? search = null,
            string? role = null); 

        Task<UserManagementDto> CreateUserAsync(CreateUserRequest request);

        Task<bool> ToggleUserStatusAsync(int userId);

        Task<bool> DeleteUserAsync(int id);

        Task<IEnumerable<ModelVersionDto>> ListModelVersionsAsync();

        Task ReloadAiModelAsync(string agentName); 
    }
}