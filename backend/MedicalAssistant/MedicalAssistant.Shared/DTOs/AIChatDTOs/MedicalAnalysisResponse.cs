using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs;

public record MedicalAnalysisResponseDTO
{
    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    [JsonPropertyName("analysis_ar")]
    public string? Analysis { get; init; }

    [JsonPropertyName("technical_details")]
    public string? TechnicalDetails { get; init; }

    [JsonPropertyName("model_used")]
    public string? ModelUsed { get; init; }

    [JsonPropertyName("disclaimer")]
    public string? Disclaimer { get; init; }
}