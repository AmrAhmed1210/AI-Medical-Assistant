using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services.Services
{
    public class MessageService : IMessageService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMessageRepository _messages;

        public MessageService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
            _messages = (IMessageRepository)_unitOfWork.Repository<Message>();
        }

        public async Task<MessageDto> SendMessageAsync(int sessionId, int userId, string role, string content)
        {
            var entity = new Message
            {
                SessionId = sessionId,
                Role = role,
                Content = content,
                Timestamp = DateTime.UtcNow,
            };

            await _messages.AddAsync(entity);
            await _unitOfWork.SaveChangesAsync();

            return Map(entity);
        }

        public async Task<IReadOnlyList<MessageDto>> GetMessagesForSessionAsync(int sessionId)
        {
            var items = await _messages.GetBySessionIdAsync(sessionId);
            return items.Select(Map).ToList();
        }

        private static MessageDto Map(Message m) => new MessageDto
        {
            Id = m.Id,
            SessionId = m.SessionId,
            Role = m.Role,
            Content = m.Content,
            Timestamp = m.Timestamp,
        };
    }
}
