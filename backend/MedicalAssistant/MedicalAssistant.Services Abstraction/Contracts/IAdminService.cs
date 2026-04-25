using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
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
        Task<bool> DeactivateUserAsync(int userId, string role);
        Task<bool> ActivateUserAsync(int userId, string role);

        Task<bool> DeleteUserAsync(int id, string role);

        Task<IEnumerable<ModelVersionDto>> ListModelVersionsAsync();

        Task ReloadAiModelAsync(string agentName);

        Task<IEnumerable<DoctorApplicationDto>> GetDoctorApplicationsAsync(string? status = null);
        Task<bool> ApproveDoctorApplicationAsync(int applicationId);
        Task<bool> RejectDoctorApplicationAsync(int applicationId, string? reason = null);
    }
}
