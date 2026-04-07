using MedicalAssistant.Domain.Entities.SessionsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IMessageRepository : IGenericRepository<Message>
    {
        Task<IEnumerable<Message>> GetBySessionIdAsync(int sessionId);
    }
}
