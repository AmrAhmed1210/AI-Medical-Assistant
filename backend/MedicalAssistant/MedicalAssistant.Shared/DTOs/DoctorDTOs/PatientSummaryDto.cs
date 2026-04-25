namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class PatientSummaryDto
{
    public int Id { get; set; }
    public string FullName { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string? PhoneNumber { get; set; }
    public DateTime? DateOfBirth { get; set; }
    public string? Gender { get; set; }
    public string? BloodType { get; set; }
    public string? Allergies { get; set; }
    public int TotalAppointments { get; set; }
    public string? LastVisit { get; set; }
    public string? PhotoUrl { get; set; }
}
