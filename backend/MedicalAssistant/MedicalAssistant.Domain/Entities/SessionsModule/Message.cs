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

        // Sent timestamp
        public DateTime Timestamp { get; set; }
    }
}
