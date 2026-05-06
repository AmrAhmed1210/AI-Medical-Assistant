using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs;

public class CreateWalkInPatientDto
{
    [Required]
    public string FullName { get; set; } = string.Empty;

    public string? Email { get; set; }

    [Required]
    public string PhoneNumber { get; set; } = string.Empty;
}
