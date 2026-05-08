using System;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    public class UpdatePatientDocumentDto
    {
        public string? DocumentType { get; set; }
        public string? Title { get; set; }
        public string? Description { get; set; }
        public DateTime? DocumentDate { get; set; }
    }
}
