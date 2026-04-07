using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class SessionRepository(MedicalAssistantDbContext context) : GenericRepository<Session>(context), ISessionRepository
    {
        public async Task<IEnumerable<Session>> GetByUserIdAsync(int userId)
        {
            return await _dbSet.Where(s => s.UserId == userId && !s.IsDeleted).ToListAsync();
        }

        public async Task<Session?> GetByIdAsync(int id)
        {
            return await _dbSet.FirstOrDefaultAsync(s => s.Id == id && !s.IsDeleted);
        }

        public async Task<(IEnumerable<Session> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
        {
            var totalCount = await _dbSet.Where(s => !s.IsDeleted).CountAsync();
            var items = await _dbSet
                .Where(s => !s.IsDeleted)
                .OrderByDescending(s => s.CreatedAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();
            return (items, totalCount);
        }
    }
}
