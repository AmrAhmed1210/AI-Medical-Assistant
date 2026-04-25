namespace MedicalAssistant.Shared.DTOs.DoctorDTOs
{
    public class DoctorApplicationDto
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Phone { get; set; } = string.Empty;
        public int SpecialtyId { get; set; }
        public string SpecialtyName { get; set; } = string.Empty;
        public int Experience { get; set; }
        public string Bio { get; set; } = string.Empty;
        public string LicenseNumber { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string DocumentUrl { get; set; } = string.Empty;
        public string? PhotoUrl { get; set; }
        public string Status { get; set; } = string.Empty;
        public DateTime SubmittedAt { get; set; }
        public DateTime? ProcessedAt { get; set; }
    }

    public class ApplyDoctorRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Phone { get; set; } = string.Empty;
        public int SpecialtyId { get; set; }
        public int Experience { get; set; }
        public string Bio { get; set; } = string.Empty;
        public string LicenseNumber { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string DocumentUrl { get; set; } = string.Empty;
        public string? PhotoUrl { get; set; }
    }
}
