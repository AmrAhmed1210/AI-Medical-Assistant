using System;

namespace MedicalAssistant.Shared.DTOs.SessionDTOs
{
    public class MessageDto
    {
        public int Id { get; set; }
        public int SessionId { get; set; }
        public string Role { get; set; } = string.Empty; // "user" | "assistant"
        public string Content { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
    }
}
