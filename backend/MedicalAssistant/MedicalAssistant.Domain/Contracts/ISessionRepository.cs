using MedicalAssistant.Domain.Entities.SessionsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface ISessionRepository : IGenericRepository<Session>
    {
        Task<IEnumerable<Session>> GetByUserIdAsync(int userId);
        new Task<Session?> GetByIdAsync(int id);
        Task<(IEnumerable<Session> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);
    }
}
