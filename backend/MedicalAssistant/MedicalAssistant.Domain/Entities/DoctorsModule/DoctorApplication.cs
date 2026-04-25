using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class DoctorApplication : BaseEntity
{
    public string Name { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Phone { get; set; } = string.Empty;
    public int SpecialtyId { get; set; }
    public Specialty Specialty { get; set; } = null!;
    public int Experience { get; set; }
    public string Bio { get; set; } = string.Empty;
    public string LicenseNumber { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string DocumentUrl { get; set; } = string.Empty;
    public string? PhotoUrl { get; set; }
    
    public string Status { get; set; } = "Pending";
    
    public DateTime SubmittedAt { get; set; } = DateTime.UtcNow;
    public DateTime? ProcessedAt { get; set; }
}
