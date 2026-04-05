namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class DoctorUpdateDto
{
    public string Specialty { get; set; } = string.Empty;
    public string? Bio { get; set; }
    public decimal? ConsultFee { get; set; }
    public int? YearsExperience { get; set; }
}