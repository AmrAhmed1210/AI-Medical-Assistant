using System.Text.Json.Serialization;

namespace MedicalAssistant.Shared.DTOs.AIChatDTOs
{
    public record MedicalAnalysisResponseDTO(
        [property: JsonPropertyName("status")]
    string Status,

        [property: JsonPropertyName("analysis_ar")]   // ← كان "analysis"
    string? Analysis,

        [property: JsonPropertyName("technical_details")]  // ← field جديد
    string? TechnicalDetails,

        [property: JsonPropertyName("model_used")]
    string? ModelUsed,

        [property: JsonPropertyName("disclaimer")]
    string? Disclaimer
    );
}