using System.Collections.Generic;

namespace MedicalAssistant.Shared.DTOs.SessionDTOs
{
    public class SessionDetailDto : SessionDto
    {
        public List<MessageDto> Messages { get; set; } = new();
        public AnalysisResultDto? AnalysisResult { get; set; }
    }
}
