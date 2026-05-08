using System;
using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    public class CreatePatientDocumentDto
    {
        [Required]
        public string DocumentType { get; set; } = string.Empty; // scan, lab, report, xray, mri, ct

        [Required]
        public string Title { get; set; } = string.Empty;

        public string? Description { get; set; }

        public DateTime? DocumentDate { get; set; }
    }
}
