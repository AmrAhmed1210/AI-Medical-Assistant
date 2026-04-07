using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class MessageRepository : GenericRepository<Message>, IMessageRepository
    {
        public MessageRepository(MedicalAssistantDbContext context) : base(context) { }

        public async Task<IEnumerable<Message>> GetBySessionIdAsync(int sessionId)
        {
            return await _dbSet
                .Where(m => m.SessionId == sessionId)
                .OrderBy(m => m.Timestamp)
                .ToListAsync();
        }
    }
}
