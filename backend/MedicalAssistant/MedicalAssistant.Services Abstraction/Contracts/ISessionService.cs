using MedicalAssistant.Shared.DTOs.SessionDTOs;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface ISessionService
    {
        Task<SessionDto> CreateSessionAsync(int userId, string? title = null);
        Task<SessionDetailDto?> GetSessionByIdAsync(int id);
        Task<IEnumerable<SessionDto>> GetSessionsByUserIdAsync(int userId);
        Task<(IEnumerable<SessionDto> Items, int TotalCount)> GetPaginatedSessionsAsync(int pageNumber, int pageSize);
        Task<bool> DeleteSessionAsync(int id);
        Task<SessionDto?> UpdateLastMessageTimeAsync(int id);
    }
}
