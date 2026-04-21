namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AvailabilityDto
{
    // From 0 (Sunday) to 6 (Saturday) to match the Entity
    public byte DayOfWeek { get; set; }

    public string DayName { get; set; } = string.Empty;

    // Prefer receiving time as string in format "HH:mm" (e.g., "09:00")
    // AutoMapper will convert it to TimeSpan in the Service
    public string StartTime { get; set; } = string.Empty;

    public string EndTime { get; set; } = string.Empty;

    public bool IsAvailable { get; set; } = true;

    public int SlotDurationMinutes { get; set; } = 30;
    
    public List<string> TimeSlots { get; set; } = new();
}
