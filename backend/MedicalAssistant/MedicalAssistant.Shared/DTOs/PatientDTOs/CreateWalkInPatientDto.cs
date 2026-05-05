using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs;

public class CreateWalkInPatientDto
{
    [Required]
    public string FullName { get; set; } = string.Empty;

    [Required]
    [EmailAddress]
    public string Email { get; set; } = string.Empty;

    [Required]
    public string PhoneNumber { get; set; } = string.Empty;
}
