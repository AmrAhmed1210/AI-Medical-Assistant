// MedicalAnalysisResponseDTO.cs

using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs;

public sealed record MedicalAnalysisResponseDTO
{
    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    [JsonPropertyName("analysis")]
    public string? Analysis { get; init; }

    [JsonPropertyName("model_used")]
    public string? ModelUsed { get; init; }

    [JsonPropertyName("disclaimer")]
    public string? Disclaimer { get; init; }
}