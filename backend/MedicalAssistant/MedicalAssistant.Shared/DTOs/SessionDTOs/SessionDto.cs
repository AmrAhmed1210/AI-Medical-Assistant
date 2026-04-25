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
        public string Type { get; set; } = "AI";
        public bool IsDeleted { get; set; }
        public string? LastMessage { get; set; }
        public DateTime? LastMessageAt { get; set; }
        public string? PatientName { get; set; }
        public string? PatientPhotoUrl { get; set; }
    }
}
