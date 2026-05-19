namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class UpdateDoctorProfileRequest
{
    public string FullName { get; set; } = string.Empty;
    public string? Bio { get; set; }
    public int YearsExperience { get; set; }
    public decimal ConsultationFee { get; set; }
    public bool IsAvailable { get; set; }
    public string? Location { get; set; }
    public string? PhoneNumber { get; set; }
}
