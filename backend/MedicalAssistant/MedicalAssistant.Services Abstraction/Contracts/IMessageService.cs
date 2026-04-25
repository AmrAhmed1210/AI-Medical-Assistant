using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IMessageService
    {
        Task<MessageDto> SendMessageAsync(int sessionId, int userId, string role, string content, string type = "text", string? attachmentUrl = null, string? fileName = null);
        Task<IReadOnlyList<MessageDto>> GetMessagesForSessionAsync(int sessionId);
    }
}
