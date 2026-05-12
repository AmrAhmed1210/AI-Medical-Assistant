namespace MedicalAssistant.Shared.DTOs.AIChatBotDTOs;

public record MatchResult(
    string Symptom,
    string Reply,
    string Category,
    double Confidence);

public record AIResponseDTO(
    string Query,
    string GeminiReply,
    List<MatchResult> Matches,
    bool LowConfidence);
