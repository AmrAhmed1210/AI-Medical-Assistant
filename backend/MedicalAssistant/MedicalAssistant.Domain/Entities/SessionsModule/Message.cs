namespace MedicalAssistant.Domain.Entities.SessionsModule
{
    public class Message : BaseEntity
    {
        public int SessionId { get; set; }
        public virtual Session Session { get; set; } = null!;

        public string Role { get; set; } = "user";

        public string Content { get; set; } = string.Empty;

        public string MessageType { get; set; } = "text";
        public string? AttachmentUrl { get; set; }
        public string? FileName { get; set; }

        public string SenderName { get; set; } = string.Empty;

        public DateTime Timestamp { get; set; }

        public string? SenderPhotoUrl { get; set; }
    }
}
