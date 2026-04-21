namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class DoctorDetailDto
{
    public int Id { get; set; }
    public int? UserId { get; set; }
    public string FullName { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Specialty { get; set; } = string.Empty;
    public string? SpecialityNameAr { get; set; }
    public string? Bio { get; set; }
    public string? PhotoUrl { get; set; }
    public decimal? ConsultFee { get; set; }
    public int? YearsExperience { get; set; }
    public bool IsAvailable { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? UpdatedAt { get; set; }
}
