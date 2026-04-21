namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class BookedSlotDto
{
    public string Date { get; set; } = string.Empty;
    public string Time { get; set; } = string.Empty;
}

public class DoctorScheduleDto
{
    public int DoctorId { get; set; }
    public string DoctorName { get; set; } = string.Empty;
    public bool IsMobileEnabled { get; set; }
    public bool IsProfileComplete { get; set; }
    public bool HasSchedule { get; set; }
    public List<AvailabilityDto> Days { get; set; } = new();
    public List<BookedSlotDto> BookedSlots { get; set; } = new();
}
