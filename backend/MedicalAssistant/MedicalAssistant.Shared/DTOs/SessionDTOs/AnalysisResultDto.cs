using System.Collections.Generic;

namespace MedicalAssistant.Shared.DTOs.SessionDTOs
{
    public class AnalysisResultDto
    {
        public List<SymptomDto> Symptoms { get; set; } = new();
        public string UrgencyLevel { get; set; } = "MEDIUM";
        public string Disclaimer { get; set; } = "This is not medical advice.";
    }
}
