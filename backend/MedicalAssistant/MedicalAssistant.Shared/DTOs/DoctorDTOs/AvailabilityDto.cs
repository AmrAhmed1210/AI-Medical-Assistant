namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AvailabilityDto
{
    // من 0 (الأحد) إلى 6 (السبت) ليتوافق مع الـ Entity
    public byte DayOfWeek { get; set; }

    // يفضل استقبال الوقت كـ string بتنسيق "HH:mm" (مثل "09:00")
    // والـ AutoMapper هيحولها لـ TimeSpan في الـ Service
    public string StartTime { get; set; } = string.Empty;

    public string EndTime { get; set; } = string.Empty;

    public bool IsAvailable { get; set; } = true;
}