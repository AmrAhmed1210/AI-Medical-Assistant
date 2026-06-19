namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class SymptomDto
{
    public string Term { get; set; } = string.Empty;
    public string? TermAr { get; set; }
    public string Icd11 { get; set; } = string.Empty;
    public int? Severity { get; set; }
}
