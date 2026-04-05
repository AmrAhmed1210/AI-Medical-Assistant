namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AIReportDto
{
    public Guid Id { get; set; }
    public Guid PatientId { get; set; }
    public string PatientName { get; set; } = string.Empty;
    public string UrgencyLevel { get; set; } = string.Empty;
    public List<SymptomDto> Symptoms { get; set; } = new();
    public string Disclaimer { get; set; } = "This is not medical advice.";
    public DateTime CreatedAt { get; set; }
}