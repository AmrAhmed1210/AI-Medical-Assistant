namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class DoctorDashboardDto
{
    public int TodayAppointments { get; set; }
    public int PendingAppointments { get; set; }
    public int TotalPatients { get; set; }
    public int UnreadReports { get; set; }
}