using System.Linq;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services.Services
{
    public class MessageService : IMessageService
    {
        private readonly IUnitOfWork _unitOfWork;

        public MessageService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<MessageDto> SendMessageAsync(int sessionId, int userId, string role, string content, string type = "text", string? attachmentUrl = null, string? fileName = null)
        {
            var (senderName, senderPhotoUrl) = await ResolveSenderDetailsAsync(userId, role);
            
            var entity = new Message
            {
                SessionId = sessionId,
                Role      = role,
                Content   = content,
                MessageType = type,
                AttachmentUrl = attachmentUrl,
                FileName = fileName,
                SenderName = senderName,
                SenderPhotoUrl = senderPhotoUrl,
                Timestamp = DateTime.UtcNow,
            };

            await _unitOfWork.Messages.AddAsync(entity);
            await _unitOfWork.SaveChangesAsync();

            // Support Auto-reply logic
            var session = await _unitOfWork.Sessions.GetByIdAsync(sessionId);
            if (session?.Type == "SupportChat" && (role.Equals("user", StringComparison.OrdinalIgnoreCase) || role.Equals("doctor", StringComparison.OrdinalIgnoreCase)))
            {
                var messages = await _unitOfWork.Repository<Message>()
                    .FindAsync(m => m.SessionId == sessionId);
                    
                if (messages != null && messages.Count() == 1)
                {
                    var autoReply = new Message
                    {
                        SessionId = sessionId,
                        Role = "admin",
                        Content = "شكراً لتواصلك معنا. سيتم الرد عليك في أقرب وقت ممكن.",
                        SenderName = "System Admin",
                        Timestamp = DateTime.UtcNow.AddSeconds(1),
                        MessageType = "text"
                    };
                    await _unitOfWork.Messages.AddAsync(autoReply);
                    await _unitOfWork.SaveChangesAsync();
                }
            }

            return Map(entity);
        }

        public async Task<IReadOnlyList<MessageDto>> GetMessagesForSessionAsync(int sessionId)
        {
            var items = await _unitOfWork.Repository<Message>()
                .FindAsync(m => m.SessionId == sessionId);
                
            return (items ?? Enumerable.Empty<Message>())
                .OrderBy(m => m.Timestamp)
                .Select(Map)
                .ToList();
        }

        private static MessageDto Map(Message m) => new MessageDto
        {
            Id        = m.Id,
            SessionId = m.SessionId,
            Role      = m.Role,
            Content   = m.Content,
            MessageType = m.MessageType,
            AttachmentUrl = m.AttachmentUrl,
            FileName = m.FileName,
            SenderName = m.SenderName,
            SenderPhotoUrl = m.SenderPhotoUrl,
            Timestamp = m.Timestamp,
        };

        private async Task<(string Name, string? PhotoUrl)> ResolveSenderDetailsAsync(int userId, string role)
        {
            if (string.Equals(role, "doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctor = (await _unitOfWork.Repository<Doctor>()
                    .FindAsync(d => d.UserId == userId))
                    .FirstOrDefault();

                var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                string? photoUrl = doctor?.ImageUrl ?? user?.PhotoUrl;

                if (!string.IsNullOrWhiteSpace(doctor?.Name))
                {
                    return ($"Dr. {doctor.Name.Trim()}", photoUrl);
                }

                if (!string.IsNullOrWhiteSpace(user?.FullName))
                {
                    return ($"Dr. {user.FullName.Trim()}", photoUrl);
                }
            }

            if (string.Equals(role, "user", StringComparison.OrdinalIgnoreCase))
            {
                var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == userId)).FirstOrDefault();
                
                string? photoUrl = user?.PhotoUrl ?? patient?.ImageUrl;
                string name = user?.FullName ?? patient?.FullName ?? "User";

                return (name.Trim(), photoUrl);
            }

            return (role, null);
        }
    }
}