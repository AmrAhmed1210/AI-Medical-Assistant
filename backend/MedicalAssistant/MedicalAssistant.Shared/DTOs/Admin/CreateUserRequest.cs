namespace MedicalAssistant.Shared.DTOs.Admin;

public class CreateUserRequest
{
    public string FullName { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public string Role { get; set; } = string.Empty;
    public string? PhoneNumber { get; set; }
    public string? SpecialityName { get; set; }
    public string? SpecialityNameAr { get; set; }
    public int? YearsExperience { get; set; }
    public decimal? ConsultationFee { get; set; }
    public string? Bio { get; set; }
    public string? PhotoUrl { get; set; }
}