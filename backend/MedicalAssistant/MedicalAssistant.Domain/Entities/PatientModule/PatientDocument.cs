using System;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class PatientDocument : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public string DocumentType { get; set; } = string.Empty; // "scan", "lab", "report", "xray", "mri", "ct"
        public string Title { get; set; } = string.Empty;
        public string FileUrl { get; set; } = string.Empty;
        public string FileType { get; set; } = string.Empty; // image/jpeg, application/pdf
        public string? Description { get; set; }
        public DateTime DocumentDate { get; set; } = DateTime.UtcNow;
        public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    }
}
