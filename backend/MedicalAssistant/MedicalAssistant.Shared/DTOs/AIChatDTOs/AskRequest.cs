using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs;

public sealed record MessageDto(
    [property: JsonPropertyName("role")]
    string Role,

    [property: JsonPropertyName("content")]
    string Content
);

public sealed record AskRequest(
    [property: JsonPropertyName("question")]
    string Question,

    [property: JsonPropertyName("history")]
    List<MessageDto>? History
);