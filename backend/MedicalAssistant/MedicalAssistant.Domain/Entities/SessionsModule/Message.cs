namespace MedicalAssistant.Domain.Entities.SessionsModule
{
    public class Message : BaseEntity
    {
        // Parent session
        public int SessionId { get; set; }
        public virtual Session Session { get; set; } = default!;

        // Role: user | assistant | doctor
        public string Role { get; set; } = "user";

        // Message content
        public string Content { get; set; } = string.Empty;

        // Type: text | image | file
        public string MessageType { get; set; } = "text";
        public string? AttachmentUrl { get; set; }
        public string? FileName { get; set; }

        // Display name for sender on chat bubbles
        public string SenderName { get; set; } = string.Empty;

        // Sent timestamp
        public DateTime Timestamp { get; set; }

        // Snapshot of photo URL for fast display
        public string? SenderPhotoUrl { get; set; }
    }
}
