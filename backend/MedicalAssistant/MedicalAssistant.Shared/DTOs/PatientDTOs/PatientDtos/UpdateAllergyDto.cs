using System;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    public class UpdateAllergyDto
    {
        public string? AllergyType { get; set; }
        public string? AllergenName { get; set; }
        public string? Severity { get; set; }
        public string? ReactionDescription { get; set; }
        public DateOnly? FirstObservedDate { get; set; }
        public bool? IsActive { get; set; }
    }
}
