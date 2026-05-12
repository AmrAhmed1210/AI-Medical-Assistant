using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatBotDTOs;

public record MatchResult(
    [property: JsonPropertyName("symptom")]
    string Symptom,

    [property: JsonPropertyName("reply")]
    string Reply,

    [property: JsonPropertyName("category")]
    string Category,

    [property: JsonPropertyName("confidence")]
    double Confidence
);

public record AIResponseDTO(
    [property: JsonPropertyName("query")]
    string Query,

    [property: JsonPropertyName("gemini_reply")]
    string GeminiReply,

    [property: JsonPropertyName("matches")]
    List<MatchResult> Matches,

    [property: JsonPropertyName("low_confidence")]
    bool LowConfidence
);