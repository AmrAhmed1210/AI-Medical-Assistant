namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AvailabilityDto
{
    public byte DayOfWeek { get; set; }
    public string StartTime { get; set; } = string.Empty; 
    public string EndTime { get; set; } = string.Empty; 
    public bool IsAvailable { get; set; } = true;
}