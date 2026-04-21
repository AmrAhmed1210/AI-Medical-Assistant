namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class DoctorAvailability : BaseEntity
{
    public int DoctorId { get; set; }
    public Doctor Doctor { get; set; } = null!;
    public byte DayOfWeek { get; set; }
    public TimeSpan StartTime { get; set; }
    public TimeSpan EndTime { get; set; }
    public bool IsAvailable { get; set; } = true;
    public int SlotDurationMinutes { get; set; } = 30;
}