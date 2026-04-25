using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface ISessionService
    {
        Task<SessionDto> CreateSessionAsync(int userId, string? title = null, string type = "AI");
        Task<SessionDetailDto?> GetSessionByIdAsync(int id);
        Task<IEnumerable<SessionDto>> GetSessionsByUserIdAsync(int userId);
        Task<IEnumerable<SessionDto>> GetAllSessionsAsync();
        Task<(IEnumerable<SessionDto> Items, int TotalCount)> GetPaginatedSessionsAsync(int pageNumber, int pageSize);
        Task<bool> DeleteSessionAsync(int id);
        Task<SessionDto?> UpdateLastMessageTimeAsync(int id);
    }
}