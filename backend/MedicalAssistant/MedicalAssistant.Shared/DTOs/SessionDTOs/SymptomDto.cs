namespace MedicalAssistant.Shared.DTOs.SessionDTOs
{
    public class SymptomDto
    {
        public string Term { get; set; } = string.Empty;
        public string Icd11 { get; set; } = string.Empty;
        public int Severity { get; set; }
    }
}
