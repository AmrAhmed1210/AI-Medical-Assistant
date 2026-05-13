using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatBotDTOs;

public record MedicalAnalysisResponseDTO(
    [property: JsonPropertyName("status")]
    string Status,

    [property: JsonPropertyName("analysis")]
    string Analysis,

    [property: JsonPropertyName("model_used")]
    string? ModelUsed,

    [property: JsonPropertyName("disclaimer")]
    string? Disclaimer
);
