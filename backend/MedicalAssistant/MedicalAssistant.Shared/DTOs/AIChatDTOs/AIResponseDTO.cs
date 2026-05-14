// AIResponseDTO.cs

using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs;

public sealed record MatchResultDto
{
    [JsonPropertyName("question")]
    public string Question { get; init; } = string.Empty;

    [JsonPropertyName("answer")]
    public string Answer { get; init; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }
}

public sealed record AIResponseDTO
{
    [JsonPropertyName("query")]
    public string Query { get; init; } = string.Empty;

    [JsonPropertyName("reply")]
    public string Reply { get; init; } = string.Empty;

    [JsonPropertyName("model_used")]
    public string ModelUsed { get; init; } = string.Empty;

    [JsonPropertyName("matches")]
    public List<MatchResultDto> Matches { get; init; } = [];

    [JsonPropertyName("is_medical")]
    public bool IsMedical { get; init; }

    [JsonPropertyName("found_in_database")]
    public bool FoundInDatabase { get; init; }

    [JsonPropertyName("low_confidence")]
    public bool LowConfidence { get; init; }

    [JsonPropertyName("language")]
    public string Language { get; init; } = string.Empty;

    [JsonPropertyName("disclaimer")]
    public string Disclaimer { get; init; } = string.Empty;
}