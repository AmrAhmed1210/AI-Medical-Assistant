namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class UpdateDoctorScheduleRequest
{
    public bool IsMobileEnabled { get; set; }
    public List<AvailabilityDto> Days { get; set; } = new();
}
