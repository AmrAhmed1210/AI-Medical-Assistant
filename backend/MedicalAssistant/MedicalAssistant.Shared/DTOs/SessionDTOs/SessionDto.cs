using System;

namespace MedicalAssistant.Shared.DTOs.SessionDTOs
{
    public class SessionDto
    {
        public int Id { get; set; }
        public int UserId { get; set; }
        public string? Title { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
        public int MessageCount { get; set; }
        public string? UrgencyLevel { get; set; }
        public bool IsDeleted { get; set; }
    }
}
