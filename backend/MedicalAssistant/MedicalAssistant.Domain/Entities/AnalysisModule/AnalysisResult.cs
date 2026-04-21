using MedicalAssistant.Domain.Entities.PatientModule;

namespace MedicalAssistant.Domain.Entities.AnalysisModule;

public class AnalysisResult : BaseEntity
{
    public int PatientId { get; set; }
    public Patient Patient { get; set; } = null!;

    public int SessionId { get; set; }
    public int MessageId { get; set; }

    public string? SymptomsJson { get; set; }
    public string? UrgencyLevel { get; set; } // HIGH, MEDIUM, LOW
    public decimal? UrgencyScore { get; set; }

    public string Disclaimer { get; set; } = "This is not medical advice.";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}