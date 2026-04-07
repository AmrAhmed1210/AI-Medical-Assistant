using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IMessageService
    {
        Task<MessageDto> SendMessageAsync(int sessionId, int userId, string role, string content);
        Task<IReadOnlyList<MessageDto>> GetMessagesForSessionAsync(int sessionId);
    }
}
