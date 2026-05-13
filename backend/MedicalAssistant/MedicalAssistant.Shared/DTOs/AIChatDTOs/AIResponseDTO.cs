using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs;

public record MatchResult
{
    [JsonPropertyName("symptom")]
    public string Symptom { get; init; } = string.Empty;

    [JsonPropertyName("reply")]
    public string Reply { get; init; } = string.Empty;

    [JsonPropertyName("category")]
    public string Category { get; init; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }
}

public record AIResponseDTO
{
    [JsonPropertyName("query")]
    public string Query { get; init; } = string.Empty;

    [JsonPropertyName("reply")]
    public string Reply { get; init; } = string.Empty;

    [JsonPropertyName("model_used")]
    public string? ModelUsed { get; init; }

    [JsonPropertyName("matches")]
    public List<MatchResult> Matches { get; init; } = new();

    [JsonPropertyName("low_confidence")]
    public bool LowConfidence { get; init; }

    [JsonPropertyName("is_medical")]
    public bool IsMedical { get; init; }

    [JsonPropertyName("found_in_database")]
    public bool FoundInDatabase { get; init; }

    [JsonPropertyName("disclaimer")]
    public string? Disclaimer { get; init; }

    [JsonPropertyName("language")]
    public string? Language { get; init; }
}